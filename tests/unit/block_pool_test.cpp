#include "kvcache/block_pool.h"

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "memory/allocator.h"
#include "memory/caching_allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <thread>
#include <utility>
#include <vector>

// BlockPool tests (M8-T02; design: docs/design/paged-kv-cache.md §6). Pure
// bookkeeping over pre-allocated per-layer K/V slabs — construction validation,
// free-list LIFO allocate/release, refcount share, exhaustion →
// ResourceExhausted, double-free / share-on-free → CHECK, stats accuracy
// through scripted sequences, blocks_needed arithmetic, the §5.2 capacity
// helpers, and thread-safety. No model, no kernels, so no SCALAR_PASS.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::kvcache::BlockPool;
using engine::kvcache::BlockPoolStats;
using engine::kvcache::CacheGeometry;
using engine::memory::Allocator;
using engine::memory::Buffer;
using engine::memory::CachingAllocator;
using engine::memory::CpuAllocator;
using engine::tensor::DataType;
using engine::tensor::Device;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] CacheGeometry TinyGeom() {
  // tiny-llama: L=2, Hkv=2, d=16.
  return CacheGeometry{.num_layers = 2,
                       .num_kv_heads = 2,
                       .head_dim = 16,
                       .dtype = DataType::kFloat32};
}

[[nodiscard]] bool IsAligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// ----------------------------- construction -----------------------------

TEST(BlockPoolCreate, RejectsNonPositiveGeometry) {
  CacheGeometry g = TinyGeom();
  g.num_kv_heads = 0;
  EXPECT_TRUE(IsInvalidArgument(BlockPool::Create(g, 16, 4, nullptr).status()));
  g = TinyGeom();
  g.head_dim = -1;
  EXPECT_TRUE(IsInvalidArgument(BlockPool::Create(g, 16, 4, nullptr).status()));
  g = TinyGeom();
  g.num_layers = 0;
  EXPECT_TRUE(IsInvalidArgument(BlockPool::Create(g, 16, 4, nullptr).status()));
}

TEST(BlockPoolCreate, RejectsNonFloat32Dtype) {
  CacheGeometry g = TinyGeom();
  g.dtype = DataType::kInt8;
  EXPECT_TRUE(IsUnimplemented(BlockPool::Create(g, 16, 4, nullptr).status()));
}

TEST(BlockPoolCreate, RejectsInvalidBlockSize) {
  for (const int bs : {0, 1, 4, 12, 24, 48, 128, -16}) {
    EXPECT_TRUE(IsInvalidArgument(
        BlockPool::Create(TinyGeom(), bs, 4, nullptr).status()))
        << "block_size=" << bs;
  }
}

TEST(BlockPoolCreate, AcceptsValidBlockSizes) {
  for (const int bs : {8, 16, 32, 64}) {
    auto pool = BlockPool::Create(TinyGeom(), bs, 3, nullptr);
    EXPECT_TRUE(pool.ok()) << "block_size=" << bs;
    EXPECT_EQ(pool->block_size(), bs);
  }
}

TEST(BlockPoolCreate, RejectsNonPositiveNumBlocks) {
  EXPECT_TRUE(IsInvalidArgument(
      BlockPool::Create(TinyGeom(), 16, 0, nullptr).status()));
  EXPECT_TRUE(IsInvalidArgument(
      BlockPool::Create(TinyGeom(), 16, -5, nullptr).status()));
}

TEST(BlockPoolCreate, NullAllocatorUsesDefault) {
  const BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 4, nullptr));
  EXPECT_EQ(pool.num_blocks(), 4);
  EXPECT_EQ(pool.stats().free, 4);
}

TEST(BlockPoolCreate, StridesMatchGeometry) {
  const CacheGeometry g = TinyGeom();
  const int bs = 16;
  const BlockPool pool = Unwrap(BlockPool::Create(g, bs, 4, nullptr));
  EXPECT_EQ(pool.block_stride(),
            static_cast<std::int64_t>(g.num_kv_heads) * bs * g.head_dim);
  EXPECT_EQ(pool.head_stride(), static_cast<std::int64_t>(bs) * g.head_dim);
  EXPECT_EQ(pool.row_stride(), g.head_dim);
}

TEST(BlockPoolCreate, SlabPointersAlignedDistinctAndZeroFilled) {
  const CacheGeometry g = TinyGeom();
  const int bs = 16;
  const std::int64_t num_blocks = 4;
  BlockPool pool = Unwrap(BlockPool::Create(g, bs, num_blocks, nullptr));
  const std::int64_t slab_elems = num_blocks * g.num_kv_heads * bs * g.head_dim;

  std::set<const float*> bases;
  for (int layer = 0; layer < g.num_layers; ++layer) {
    for (float* base : {pool.k_slab(layer), pool.v_slab(layer)}) {
      EXPECT_TRUE(IsAligned(base, BlockPool::kSlabAlignment));
      EXPECT_TRUE(bases.insert(base).second)
          << "slab pointers must be distinct";
      for (std::int64_t i = 0; i < slab_elems; ++i) {
        ASSERT_EQ(base[i], 0.0F);
      }
    }
  }
  EXPECT_EQ(pool.slab_bytes(),
            slab_elems * static_cast<std::int64_t>(sizeof(float)));
  EXPECT_EQ(pool.total_bytes(), pool.slab_bytes() * 2 * g.num_layers);
}

TEST(BlockPoolCreate, BackedByCachingAllocator) {
  CpuAllocator upstream;
  CachingAllocator caching(&upstream);
  {
    const BlockPool pool =
        Unwrap(BlockPool::Create(TinyGeom(), 16, 4, &caching));
    // 2·L = 4 slabs handed out; the pool holds them live.
    EXPECT_GT(caching.stats().bytes_allocated, 0U);
    (void)pool;
  }
  // Pool destroyed → its slab Buffers return to the cache; nothing live.
  EXPECT_EQ(caching.stats().bytes_allocated, 0U);
}

// Scriptable upstream that fails after N successful allocations — proves
// Create propagates allocation failure and leaks nothing.
class FlakyAllocator final : public Allocator {
 public:
  explicit FlakyAllocator(int succeed_then_fail)
      : remaining_(succeed_then_fail) {}

  [[nodiscard]] StatusOr<Buffer> Allocate(std::size_t bytes,
                                          std::size_t /*alignment*/) override {
    if (remaining_ <= 0) {
      return engine::core::OutOfMemoryError("scripted OOM");
    }
    --remaining_;
    ++live_;
    int* live = &live_;
    void* data = bytes == 0 ? nullptr : std::malloc(bytes);
    return Buffer(data, bytes, device(), [live](void* ptr) {
      --*live;
      std::free(ptr);
    });
  }
  [[nodiscard]] Device device() const override { return Device::Cpu(); }
  [[nodiscard]] int live() const { return live_; }

 private:
  int remaining_;
  int live_ = 0;
};

TEST(BlockPoolCreate, PropagatesAllocationFailureNoLeak) {
  // tiny-llama needs 2·L = 4 slabs; fail on the 3rd.
  FlakyAllocator alloc(/*succeed_then_fail=*/2);
  auto pool = BlockPool::Create(TinyGeom(), 16, 4, &alloc);
  EXPECT_FALSE(pool.ok());
  // The two slabs allocated before the failure were dropped by RAII.
  EXPECT_EQ(alloc.live(), 0);
}

// ----------------------------- allocate / release
// -----------------------------

TEST(BlockPoolAllocate, IdsUniqueInRangeRefcountOne) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 3, nullptr));
  std::set<std::int32_t> ids;
  for (int i = 0; i < 3; ++i) {
    const std::int32_t b = Unwrap(pool.Allocate());
    EXPECT_GE(b, 0);
    EXPECT_LT(b, 3);
    EXPECT_TRUE(ids.insert(b).second);
    EXPECT_EQ(pool.refcount(b), 1);
  }
  EXPECT_EQ(pool.stats().used, 3);
  EXPECT_EQ(pool.stats().free, 0);
}

TEST(BlockPoolAllocate, ExhaustionReturnsResourceExhausted) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 2, nullptr));
  const std::int32_t a = Unwrap(pool.Allocate());
  (void)Unwrap(pool.Allocate());
  auto exhausted = pool.Allocate();
  EXPECT_TRUE(IsResourceExhausted(exhausted.status()));
  // Stats unchanged by the failed allocation.
  EXPECT_EQ(pool.stats().used, 2);
  EXPECT_EQ(pool.stats().free, 0);
  // Releasing one recovers capacity.
  pool.Release(a);
  EXPECT_TRUE(pool.Allocate().ok());
}

TEST(BlockPoolAllocate, FreeListIsLifo) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 4, nullptr));
  const std::int32_t b0 = Unwrap(pool.Allocate());
  const std::int32_t b1 = Unwrap(pool.Allocate());
  const std::int32_t b2 = Unwrap(pool.Allocate());
  // Release b1 then b2; next Allocate returns b2 (last freed), then b1.
  pool.Release(b1);
  pool.Release(b2);
  EXPECT_EQ(Unwrap(pool.Allocate()), b2);
  EXPECT_EQ(Unwrap(pool.Allocate()), b1);
  // Single release/alloc round-trip returns the same cache-warm block.
  pool.Release(b0);
  EXPECT_EQ(Unwrap(pool.Allocate()), b0);
}

TEST(BlockPoolAllocate, FirstAllocateReturnsBlockZero) {
  // Implementation detail pinned for readability of other tests: descending
  // free-list push means block 0 comes out first.
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 4, nullptr));
  EXPECT_EQ(Unwrap(pool.Allocate()), 0);
  EXPECT_EQ(Unwrap(pool.Allocate()), 1);
}

// ----------------------------- share -----------------------------

TEST(BlockPoolShare, BumpsRefcountAndKeepsBlockOwnedUntilLastRelease) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 2, nullptr));
  const std::int32_t b = Unwrap(pool.Allocate());
  pool.Share(b);
  pool.Share(b);
  EXPECT_EQ(pool.refcount(b), 3);
  EXPECT_EQ(pool.stats().used, 1);  // one physical block, however shared

  pool.Release(b);
  pool.Release(b);
  EXPECT_EQ(pool.refcount(b), 1);
  // Still owned: a full pool can't hand this block back out.
  (void)Unwrap(pool.Allocate());  // the other block
  EXPECT_TRUE(IsResourceExhausted(pool.Allocate().status()));

  pool.Release(b);  // rc 1 -> 0, back to free list
  EXPECT_EQ(pool.refcount(b), 0);
  EXPECT_EQ(Unwrap(pool.Allocate()), b);
}

// ----------------------------- scripted stats -----------------------------

TEST(BlockPoolStatsCheck, AccurateThroughScriptedSequence) {
  const std::int64_t n = 5;
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, n, nullptr));

  auto expect_stats = [&](std::int64_t used) {
    const BlockPoolStats s = pool.stats();
    EXPECT_EQ(s.total, n);
    EXPECT_EQ(s.used, used);
    EXPECT_EQ(s.free, n - used);
    EXPECT_EQ(s.used + s.free, s.total);
    EXPECT_DOUBLE_EQ(s.utilization,
                     static_cast<double>(used) / static_cast<double>(n));
    EXPECT_EQ(pool.free_blocks(), n - used);
  };

  expect_stats(0);
  const std::int32_t a = Unwrap(pool.Allocate());
  expect_stats(1);
  const std::int32_t b = Unwrap(pool.Allocate());
  expect_stats(2);
  pool.Share(a);  // sharing does not change used/free
  expect_stats(2);
  const std::int32_t c = Unwrap(pool.Allocate());
  expect_stats(3);
  pool.Release(b);  // rc 1->0, frees
  expect_stats(2);
  pool.Release(a);  // rc 2->1, still owned
  expect_stats(2);
  pool.Release(a);  // rc 1->0, frees
  expect_stats(1);
  pool.Release(c);
  expect_stats(0);
}

// ----------------------------- blocks_needed -----------------------------

TEST(BlockPoolBlocksNeeded, MatchesHandWorkedBoundaryCrossings) {
  // bs = 8 (the §7.2 examples re-derived at a valid block size).
  const BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 8, 8, nullptr));
  EXPECT_EQ(pool.blocks_needed(0, 6), 1);   // prefill 6 from empty
  EXPECT_EQ(pool.blocks_needed(6, 1), 0);   // decode within block 0
  EXPECT_EQ(pool.blocks_needed(7, 1), 0);   // fills block 0's last slot (pos 7)
  EXPECT_EQ(pool.blocks_needed(8, 1), 1);   // pos 8 crosses into block 1
  EXPECT_EQ(pool.blocks_needed(0, 16), 2);  // exactly two full blocks
  EXPECT_EQ(pool.blocks_needed(0, 17), 3);  // one token into a third
  EXPECT_EQ(pool.blocks_needed(16, 0), 0);  // no new tokens
  EXPECT_EQ(pool.blocks_needed(0, 0), 0);
}

// ----------------------------- capacity helpers (§5.2)
// -----------------------------

TEST(BlockPoolCapacity, BytesPerBlockWorkedExamples) {
  // tiny-llama, bs=16: 2·L·Hkv·bs·d·4 = 2·2·2·16·16·4 = 8 KiB (§3.3).
  EXPECT_EQ(Unwrap(BlockPool::BytesPerBlock(TinyGeom(), 16)), 8 * 1024);

  // Qwen2-0.5B: L=24, Hkv=2, d=64, bs=16 → 384 KiB (§3.3).
  const CacheGeometry qwen{.num_layers = 24,
                           .num_kv_heads = 2,
                           .head_dim = 64,
                           .dtype = DataType::kFloat32};
  EXPECT_EQ(Unwrap(BlockPool::BytesPerBlock(qwen, 16)), 384 * 1024);

  // Llama-3-8B class: L=32, Hkv=8, d=128, bs=16 → 4 MiB (§3.3).
  const CacheGeometry llama3{.num_layers = 32,
                             .num_kv_heads = 8,
                             .head_dim = 128,
                             .dtype = DataType::kFloat32};
  EXPECT_EQ(Unwrap(BlockPool::BytesPerBlock(llama3, 16)), 4 * 1024 * 1024);
}

TEST(BlockPoolCapacity, BytesPerBlockValidates) {
  EXPECT_TRUE(
      IsInvalidArgument(BlockPool::BytesPerBlock(TinyGeom(), 5).status()));
  CacheGeometry bad = TinyGeom();
  bad.num_layers = 0;
  EXPECT_TRUE(IsInvalidArgument(BlockPool::BytesPerBlock(bad, 16).status()));
}

TEST(BlockPoolCapacity, NumBlocksForBudgetWorkedExamples) {
  // Qwen2-0.5B, 1 GiB budget → ⌊1073741824 / 393216⌋ = 2730 blocks (§3.3).
  const CacheGeometry qwen{.num_layers = 24,
                           .num_kv_heads = 2,
                           .head_dim = 64,
                           .dtype = DataType::kFloat32};
  EXPECT_EQ(
      Unwrap(BlockPool::NumBlocksForBudget(qwen, 16, std::int64_t{1} << 30)),
      2730);

  // Llama-3-8B, 40 GiB → 10240 blocks (§3.3).
  const CacheGeometry llama3{.num_layers = 32,
                             .num_kv_heads = 8,
                             .head_dim = 128,
                             .dtype = DataType::kFloat32};
  EXPECT_EQ(
      Unwrap(BlockPool::NumBlocksForBudget(llama3, 16, std::int64_t{40} << 30)),
      10240);
}

TEST(BlockPoolCapacity, NumBlocksForBudgetTooSmall) {
  // Budget below one block → ResourceExhausted (model does not fit).
  EXPECT_TRUE(IsResourceExhausted(
      BlockPool::NumBlocksForBudget(TinyGeom(), 16, 1024).status()));
  EXPECT_TRUE(IsInvalidArgument(
      BlockPool::NumBlocksForBudget(TinyGeom(), 16, 0).status()));
}

// ----------------------------- capacity helper vs Create consistency
// -----------------------------

TEST(BlockPoolCapacity, CreateAcceptsDerivedNumBlocks) {
  const std::int64_t budget = std::int64_t{8} * 1024 * 10;  // 10 tiny blocks
  const std::int64_t n =
      Unwrap(BlockPool::NumBlocksForBudget(TinyGeom(), 16, budget));
  EXPECT_EQ(n, 10);
  const BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, n, nullptr));
  EXPECT_EQ(pool.total_bytes(), budget);
}

// ----------------------------- death tests -----------------------------

TEST(BlockPoolDeath, DoubleReleaseAborts) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 2, nullptr));
  const std::int32_t b = Unwrap(pool.Allocate());
  pool.Release(b);
  EXPECT_DEATH(pool.Release(b), "double free");
}

TEST(BlockPoolDeath, ShareFreeBlockAborts) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 2, nullptr));
  EXPECT_DEATH(pool.Share(0), "nobody owns");
}

TEST(BlockPoolDeath, OutOfRangeAborts) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 2, nullptr));
  EXPECT_DEATH(pool.Share(5), "out of range");
  EXPECT_DEATH(pool.Release(-1), "out of range");
  EXPECT_DEATH((void)pool.refcount(9), "out of range");
  EXPECT_DEATH((void)pool.k_slab(3), "out of range");
  EXPECT_DEATH((void)pool.v_slab(-1), "out of range");
}

TEST(BlockPoolDeath, MoveAfterHandoutAborts) {
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, 2, nullptr));
  (void)Unwrap(pool.Allocate());  // hand out a block
  EXPECT_DEATH(const BlockPool moved(std::move(pool)), "moved after");
}

// ----------------------------- concurrency -----------------------------

TEST(BlockPoolConcurrency, ConcurrentAllocateReleaseKeepsInvariant) {
  const std::int64_t n = 64;
  BlockPool pool = Unwrap(BlockPool::Create(TinyGeom(), 16, n, nullptr));
  constexpr int kThreads = 8;
  constexpr int kIters = 2000;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool] {
      for (int i = 0; i < kIters; ++i) {
        auto b = pool.Allocate();
        if (b.ok()) {
          pool.Release(*b);
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  // All blocks returned; invariant holds.
  const BlockPoolStats s = pool.stats();
  EXPECT_EQ(s.used, 0);
  EXPECT_EQ(s.free, n);
  EXPECT_EQ(s.used + s.free, s.total);
}

}  // namespace
