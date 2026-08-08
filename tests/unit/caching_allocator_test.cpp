#include "memory/caching_allocator.h"

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>

// CachingAllocator tests (M2-T06; the pool survives the CPU-first pivot,
// ADR-004). The pool is device-agnostic, so the size-class, stats, reuse,
// OOM-retry, and concurrency contracts are all exercised over host-memory
// upstreams — scriptable fakes plus the real CpuAllocator.

namespace {

using engine::core::IsInternal;
using engine::core::IsResourceExhausted;
using engine::core::StatusOr;
using engine::memory::Allocator;
using engine::memory::Buffer;
using engine::memory::CachingAllocator;
using engine::memory::CpuAllocator;
using engine::tensor::Device;

constexpr std::size_t kMiB = std::size_t{1} << 20U;

[[nodiscard]] bool IsAligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// Host-memory upstream that counts allocations and frees, records the last
// request, and can be scripted to fail — the instrument for "reuse means no
// upstream call" and for the OOM-retry courtesy.
class FakeUpstream final : public Allocator {
 public:
  enum class FailureKind : std::uint8_t {
    kResourceExhausted,
    kOutOfMemory,
    kInternal
  };

  [[nodiscard]] StatusOr<Buffer> Allocate(std::size_t bytes,
                                          std::size_t alignment) override {
    ++alloc_count_;
    last_bytes_ = bytes;
    last_alignment_ = alignment;
    if (failures_remaining_ > 0) {
      --failures_remaining_;
      switch (failure_kind_) {
        case FailureKind::kResourceExhausted:
          return engine::core::ResourceExhaustedError("scripted exhaustion");
        case FailureKind::kOutOfMemory:
          return engine::core::OutOfMemoryError("scripted exhaustion");
        case FailureKind::kInternal:
          return engine::core::InternalError("scripted failure");
      }
    }
    int* free_count = &free_count_;
    void* data = bytes == 0 ? nullptr : std::malloc(bytes);
    return Buffer(data, bytes, device(), [free_count](void* ptr) {
      ++*free_count;
      std::free(ptr);
    });
  }

  [[nodiscard]] Device device() const override { return Device::Cpu(); }

  void FailNext(int count, FailureKind kind) {
    failures_remaining_ = count;
    failure_kind_ = kind;
  }

  [[nodiscard]] int alloc_count() const { return alloc_count_; }
  [[nodiscard]] int free_count() const { return free_count_; }
  [[nodiscard]] std::size_t last_bytes() const { return last_bytes_; }
  [[nodiscard]] std::size_t last_alignment() const { return last_alignment_; }

 private:
  int alloc_count_ = 0;
  int free_count_ = 0;
  std::size_t last_bytes_ = 0;
  std::size_t last_alignment_ = 0;
  int failures_remaining_ = 0;
  FailureKind failure_kind_ = FailureKind::kResourceExhausted;
};

void ExpectStats(const CachingAllocator& pool, std::size_t bytes_allocated,
                 std::size_t bytes_reserved, std::uint64_t hit_count,
                 std::uint64_t miss_count) {
  const CachingAllocator::Stats stats = pool.stats();
  EXPECT_EQ(stats.bytes_allocated, bytes_allocated);
  EXPECT_EQ(stats.bytes_reserved, bytes_reserved);
  EXPECT_EQ(stats.hit_count, hit_count);
  EXPECT_EQ(stats.miss_count, miss_count);
}

// --- Reuse ---

TEST(CachingAllocatorTest, SameSizeReuseIsServedFromCache) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  void* first = nullptr;
  {
    const Buffer buffer = pool.Allocate(1000, 64).value();
    first = buffer.data();
    ASSERT_NE(first, nullptr);
  }
  const Buffer reused = pool.Allocate(1000, 64).value();
  EXPECT_EQ(reused.data(), first);
  EXPECT_EQ(upstream.alloc_count(), 1);  // the second Allocate never went up
  EXPECT_EQ(upstream.free_count(), 0);
  ExpectStats(pool, 1024, 1024, /*hit_count=*/1, /*miss_count=*/1);
}

TEST(CachingAllocatorTest, DifferentSizesInOneClassReuseTheBlock) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  void* first = nullptr;
  {
    const Buffer buffer = pool.Allocate(600, 64).value();
    first = buffer.data();
  }
  // 600 and 700 both round to the 1024 class.
  const Buffer reused = pool.Allocate(700, 64).value();
  EXPECT_EQ(reused.data(), first);
  EXPECT_EQ(upstream.alloc_count(), 1);
  ExpectStats(pool, 1024, 1024, /*hit_count=*/1, /*miss_count=*/1);
}

TEST(CachingAllocatorTest, DifferentClassesDoNotReuse) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  {
    const Buffer buffer = pool.Allocate(512, 64).value();
  }
  const Buffer other = pool.Allocate(513, 64).value();  // 1024 class
  EXPECT_EQ(upstream.alloc_count(), 2);
  ExpectStats(pool, 1024, 512 + 1024, /*hit_count=*/0, /*miss_count=*/2);
}

// --- Size classes & sizes reported ---

TEST(CachingAllocatorTest, RequestedSizeReportedClassSizeRequestedUpstream) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  const Buffer buffer = pool.Allocate(1000, 64).value();
  EXPECT_EQ(buffer.size_bytes(), 1000U);       // caller sees the request
  EXPECT_EQ(upstream.last_bytes(), 1024U);     // upstream sees the class
  EXPECT_EQ(upstream.last_alignment(), 256U);  // fixed pool-wide alignment
  EXPECT_EQ(buffer.device(), Device::Cpu());
  ExpectStats(pool, 1024, 1024, /*hit_count=*/0, /*miss_count=*/1);
}

TEST(CachingAllocatorTest, SizeClassRoundingObservedViaStats) {
  // The class table is an implementation constant, but exact stats are an
  // acceptance criterion, so the rounding is pinned here: 512B minimum,
  // powers of two to 1MiB, 1MiB steps above.
  struct Case {
    std::size_t requested;
    std::size_t class_bytes;
  };
  for (const Case& c :
       {Case{.requested = 1, .class_bytes = 512},
        Case{.requested = 511, .class_bytes = 512},
        Case{.requested = 512, .class_bytes = 512},
        Case{.requested = 513, .class_bytes = 1024},
        Case{.requested = 100000, .class_bytes = std::size_t{128} * 1024},
        Case{.requested = kMiB, .class_bytes = kMiB},
        Case{.requested = kMiB + 1, .class_bytes = 2 * kMiB},
        Case{.requested = (3 * kMiB) - 5, .class_bytes = 3 * kMiB}}) {
    FakeUpstream upstream;
    CachingAllocator pool(&upstream);
    const Buffer buffer = pool.Allocate(c.requested, 64).value();
    EXPECT_EQ(upstream.last_bytes(), c.class_bytes) << c.requested;
    EXPECT_EQ(pool.stats().bytes_allocated, c.class_bytes) << c.requested;
  }
}

// --- The scripted acceptance sequence: exact stats at every step ---

TEST(CachingAllocatorTest, ScriptedSequenceStatsAreExact) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  ExpectStats(pool, 0, 0, 0, 0);

  Buffer a = pool.Allocate(300, 64).value();  // 512 class, miss
  ExpectStats(pool, 512, 512, /*hit_count=*/0, /*miss_count=*/1);

  Buffer b = pool.Allocate(2000, 64).value();  // 2048 class, miss
  ExpectStats(pool, 512 + 2048, 512 + 2048, /*hit_count=*/0, /*miss_count=*/2);

  a = Buffer();  // free the 512 block into the cache
  ExpectStats(pool, 2048, 512 + 2048, /*hit_count=*/0, /*miss_count=*/2);

  const Buffer c = pool.Allocate(400, 64).value();  // 512 class, hit
  ExpectStats(pool, 512 + 2048, 512 + 2048, /*hit_count=*/1, /*miss_count=*/2);
  EXPECT_EQ(upstream.alloc_count(), 2);

  b = Buffer();  // both blocks live in the cache or with c
  ExpectStats(pool, 512, 512 + 2048, /*hit_count=*/1, /*miss_count=*/2);
}

// --- release_cached ---

TEST(CachingAllocatorTest, ReleaseCachedReturnsBlocksUpstream) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  {
    const Buffer buffer = pool.Allocate(300, 64).value();
  }  // 512, cached
  {
    const Buffer buffer = pool.Allocate(2000, 64).value();
  }  // 2048, cached
  ExpectStats(pool, 0, 512 + 2048, /*hit_count=*/0, /*miss_count=*/2);
  EXPECT_EQ(upstream.free_count(), 0);

  EXPECT_EQ(pool.release_cached(), 512U + 2048U);
  EXPECT_EQ(upstream.free_count(), 2);
  ExpectStats(pool, 0, 0, /*hit_count=*/0, /*miss_count=*/2);

  EXPECT_EQ(pool.release_cached(), 0U);  // idempotent on an empty cache
}

TEST(CachingAllocatorTest, ReleaseCachedLeavesLiveBuffersAlone) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  const Buffer live = pool.Allocate(100, 64).value();
  {
    const Buffer buffer = pool.Allocate(100, 64).value();
  }  // cached
  EXPECT_EQ(pool.release_cached(), 512U);
  EXPECT_EQ(upstream.free_count(), 1);
  ExpectStats(pool, 512, 512, /*hit_count=*/0, /*miss_count=*/2);
  // The live allocation is untouched and still writable.
  auto* bytes = static_cast<unsigned char*>(live.data());
  for (std::size_t i = 0; i < live.size_bytes(); ++i) {
    bytes[i] = static_cast<unsigned char>(i);
  }
}

TEST(CachingAllocatorTest, DestructorReleasesCachedBlocks) {
  FakeUpstream upstream;
  {
    CachingAllocator pool(&upstream);
    {
      const Buffer buffer = pool.Allocate(300, 64).value();
    }
    EXPECT_EQ(upstream.free_count(), 0);  // cached, not freed
  }
  EXPECT_EQ(upstream.free_count(), 1);  // pool destruction returned it
}

// --- Exhaustion: release_cached + one retry ---

TEST(CachingAllocatorTest, ExhaustionReleasesCacheAndRetries) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  {
    const Buffer buffer = pool.Allocate(600, 64).value();
  }  // 1024, cached
  upstream.FailNext(1, FakeUpstream::FailureKind::kResourceExhausted);

  const StatusOr<Buffer> buffer = pool.Allocate(5000, 64);  // 8192 class
  ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
  EXPECT_EQ(upstream.alloc_count(), 3);  // seed + failed attempt + retry
  EXPECT_EQ(upstream.free_count(), 1);   // the cache was released in between
  ExpectStats(pool, 8192, 8192, /*hit_count=*/0,
              /*miss_count=*/2);  // one miss per request
}

TEST(CachingAllocatorTest, HostOutOfMemoryAlsoTriggersTheRetry) {
  // The pool is device-agnostic: host exhaustion (kOutOfMemory) gets the
  // same courtesy as device exhaustion (kResourceExhausted).
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  {
    const Buffer buffer = pool.Allocate(600, 64).value();
  }
  upstream.FailNext(1, FakeUpstream::FailureKind::kOutOfMemory);

  const StatusOr<Buffer> buffer = pool.Allocate(5000, 64);
  ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
  EXPECT_EQ(upstream.free_count(), 1);
}

TEST(CachingAllocatorTest, PersistentExhaustionPropagates) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  upstream.FailNext(2, FakeUpstream::FailureKind::kResourceExhausted);
  const StatusOr<Buffer> buffer = pool.Allocate(100, 64);
  ASSERT_FALSE(buffer.ok());
  EXPECT_TRUE(IsResourceExhausted(buffer.status()))
      << buffer.status().ToString();
  EXPECT_EQ(upstream.alloc_count(), 2);  // attempt + one retry, no more
  ExpectStats(pool, 0, 0, /*hit_count=*/0, /*miss_count=*/1);
}

TEST(CachingAllocatorTest, NonExhaustionErrorPropagatesWithoutRetry) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  {
    const Buffer buffer = pool.Allocate(600, 64).value();
  }  // cached
  upstream.FailNext(1, FakeUpstream::FailureKind::kInternal);
  const StatusOr<Buffer> buffer = pool.Allocate(5000, 64);
  ASSERT_FALSE(buffer.ok());
  EXPECT_TRUE(IsInternal(buffer.status())) << buffer.status().ToString();
  EXPECT_EQ(upstream.alloc_count(), 2);  // no retry
  EXPECT_EQ(upstream.free_count(), 0);   // cache kept
  ExpectStats(pool, 0, 1024, /*hit_count=*/0, /*miss_count=*/2);
}

// --- Zero-size, device, real host memory ---

TEST(CachingAllocatorTest, ZeroSizeBypassesCacheAndStats) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  {
    const Buffer buffer = pool.Allocate(0, 64).value();
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size_bytes(), 0U);
    EXPECT_EQ(buffer.device(), Device::Cpu());
  }
  EXPECT_EQ(upstream.alloc_count(), 0);
  ExpectStats(pool, 0, 0, 0, 0);
}

TEST(CachingAllocatorTest, DeviceForwardsUpstream) {
  FakeUpstream upstream;
  const CachingAllocator pool(&upstream);
  EXPECT_EQ(pool.device(), Device::Cpu());
}

TEST(CachingAllocatorTest, WorksOverRealCpuAllocator) {
  CpuAllocator upstream;
  CachingAllocator pool(&upstream);
  void* first = nullptr;
  {
    const Buffer buffer = pool.Allocate(1000, 64).value();
    first = buffer.data();
    EXPECT_TRUE(IsAligned(first, CachingAllocator::kMaxAlignment));
    auto* bytes = static_cast<unsigned char*>(buffer.data());
    for (std::size_t i = 0; i < buffer.size_bytes(); ++i) {
      bytes[i] = static_cast<unsigned char>(i % 251);
    }
    for (std::size_t i = 0; i < buffer.size_bytes(); ++i) {
      ASSERT_EQ(bytes[i], static_cast<unsigned char>(i % 251)) << i;
    }
  }
  const Buffer reused = pool.Allocate(900, 64).value();
  EXPECT_EQ(reused.data(), first);
}

// --- CHECK violations ---

TEST(CachingAllocatorDeathTest, NullUpstreamAborts) {
  EXPECT_DEATH(const CachingAllocator pool(nullptr), "requires an upstream");
}

TEST(CachingAllocatorDeathTest, NonPowerOfTwoAlignmentAborts) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  EXPECT_DEATH((void)pool.Allocate(100, 24),
               "alignment must be a power of two");
}

TEST(CachingAllocatorDeathTest, AlignmentAboveTheCeilingAborts) {
  FakeUpstream upstream;
  CachingAllocator pool(&upstream);
  EXPECT_DEATH((void)pool.Allocate(100, 512), "exceeds the pool's fixed");
}

TEST(CachingAllocatorDeathTest, DestructionWithLiveBufferAborts) {
  EXPECT_DEATH(
      {
        FakeUpstream upstream;
        Buffer leaked;  // outlives the pool below — the lifetime violation
        CachingAllocator pool(&upstream);
        leaked = pool.Allocate(64, 64).value();
      },
      "destroyed with 1 live Buffer");
}

// --- Concurrency (acceptance criterion) ---

TEST(CachingAllocatorTest, ConcurrentAllocationStress) {
  CpuAllocator upstream;
  CachingAllocator pool(&upstream);
  constexpr int kThreads = 8;
  // static: the worker lambdas index the array without capturing it.
  static constexpr int kIterations = 300;
  static constexpr std::size_t kSizes[] = {1,    300,  512,   600,
                                           2048, 5000, 100000};
  static constexpr std::size_t kNumSizes = std::size(kSizes);
  std::vector<std::thread> threads;
  std::vector<int> successes(kThreads, 0);
  threads.reserve(kThreads);
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool, &successes, t] {
      // A small per-thread working set forces alloc/free interleavings so
      // blocks migrate between threads through the free lists.
      std::vector<Buffer> held;
      for (int i = 0; i < kIterations; ++i) {
        const std::size_t bytes =
            kSizes[(t + static_cast<std::size_t>(i)) % kNumSizes];
        StatusOr<Buffer> buffer = pool.Allocate(bytes, 64);
        if (!buffer.ok()) {
          continue;
        }
        auto* data = static_cast<unsigned char*>(buffer->data());
        data[0] = static_cast<unsigned char>(t);
        data[buffer->size_bytes() - 1] = static_cast<unsigned char>(i);
        held.push_back(*std::move(buffer));
        if (held.size() > 4) {
          held.clear();
        }
        ++successes[t];
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (std::size_t t = 0; t < kThreads; ++t) {
    EXPECT_EQ(successes[t], kIterations);
  }
  const CachingAllocator::Stats stats = pool.stats();
  EXPECT_EQ(stats.bytes_allocated, 0U);  // every Buffer was destroyed
  EXPECT_EQ(stats.hit_count + stats.miss_count,
            static_cast<std::uint64_t>(kThreads) * kIterations);
  // Everything reserved is cached, and release accounts for all of it.
  EXPECT_EQ(pool.release_cached(), stats.bytes_reserved);
  EXPECT_EQ(pool.stats().bytes_reserved, 0U);
}

}  // namespace
