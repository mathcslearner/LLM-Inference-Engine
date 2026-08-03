#include "memory/allocator.h"

#include "core/status.h"
#include "tensor/device.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace {

using engine::core::IsOutOfMemory;
using engine::core::StatusOr;
using engine::memory::Allocator;
using engine::memory::Buffer;
using engine::memory::CpuAllocator;
using engine::memory::DefaultCpuAllocator;
using engine::tensor::Device;

[[nodiscard]] bool IsAligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// Test allocator that counts allocations and deleter invocations — the
// instrument for the "deleter runs exactly once" contract. The deleter
// captures only the counter pointer (self-contained, per the design), so
// buffers may outlive the allocator.
class CountingAllocator final : public Allocator {
 public:
  [[nodiscard]] StatusOr<Buffer> Allocate(std::size_t bytes,
                                          std::size_t alignment) override {
    (void)alignment;
    ++alloc_count_;
    int* free_count = &free_count_;
    void* data = bytes == 0 ? nullptr : std::malloc(bytes);
    return Buffer(data, bytes, device(), [free_count](void* ptr) {
      ++*free_count;
      std::free(ptr);
    });
  }

  [[nodiscard]] Device device() const override { return Device::Cpu(); }

  [[nodiscard]] int alloc_count() const { return alloc_count_; }
  [[nodiscard]] int free_count() const { return free_count_; }

 private:
  int alloc_count_ = 0;
  int free_count_ = 0;
};

// --- CpuAllocator: alignment ---

TEST(CpuAllocatorTest, AlignmentHonored) {
  CpuAllocator allocator;
  for (const std::size_t alignment : {8UL, 16UL, 64UL, 256UL, 4096UL}) {
    StatusOr<Buffer> buffer = allocator.Allocate(100, alignment);
    ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
    EXPECT_NE(buffer->data(), nullptr);
    EXPECT_TRUE(IsAligned(buffer->data(), alignment)) << alignment;
    EXPECT_EQ(buffer->size_bytes(), 100U);
    EXPECT_EQ(buffer->device(), Device::Cpu());
  }
}

TEST(CpuAllocatorTest, DefaultAlignmentIs64) {
  CpuAllocator allocator;
  EXPECT_EQ(allocator.default_alignment(), 64U);
  StatusOr<Buffer> buffer = allocator.Allocate(10);
  ASSERT_TRUE(buffer.ok());
  EXPECT_TRUE(IsAligned(buffer->data(), 64));
}

TEST(CpuAllocatorTest, ConfiguredDefaultAlignmentUsedByOneArgOverload) {
  CpuAllocator allocator(/*default_alignment=*/4096);
  EXPECT_EQ(allocator.default_alignment(), 4096U);
  StatusOr<Buffer> buffer = allocator.Allocate(10);
  ASSERT_TRUE(buffer.ok());
  EXPECT_TRUE(IsAligned(buffer->data(), 4096));
}

TEST(CpuAllocatorTest, SmallAlignmentsAreValid) {
  // Sub-fundamental alignments must work even though std::aligned_alloc
  // itself may not accept them (the implementation over-aligns).
  CpuAllocator allocator;
  for (const std::size_t alignment : {1UL, 2UL, 4UL}) {
    StatusOr<Buffer> buffer = allocator.Allocate(3, alignment);
    ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
    EXPECT_NE(buffer->data(), nullptr);
    EXPECT_EQ(buffer->size_bytes(), 3U);
  }
}

TEST(CpuAllocatorTest, RequestedSizeIsFullyWritable) {
  // Odd size at large alignment exercises the round-up-to-multiple math;
  // write and read back every requested byte.
  CpuAllocator allocator;
  StatusOr<Buffer> buffer = allocator.Allocate(1001, 256);
  ASSERT_TRUE(buffer.ok());
  auto* bytes = static_cast<unsigned char*>(buffer->data());
  for (std::size_t i = 0; i < buffer->size_bytes(); ++i) {
    bytes[i] = static_cast<unsigned char>(i % 251);
  }
  for (std::size_t i = 0; i < buffer->size_bytes(); ++i) {
    ASSERT_EQ(bytes[i], static_cast<unsigned char>(i % 251)) << i;
  }
}

TEST(CpuAllocatorTest, DeviceIsCpu) {
  const CpuAllocator allocator;
  EXPECT_EQ(allocator.device(), Device::Cpu());
}

// --- CpuAllocator: zero-size and failure ---

TEST(CpuAllocatorTest, ZeroSizeAllocationIsEngagedWithNullData) {
  CpuAllocator allocator;
  StatusOr<Buffer> buffer = allocator.Allocate(0, 64);
  ASSERT_TRUE(buffer.ok());
  EXPECT_EQ(buffer->data(), nullptr);
  EXPECT_EQ(buffer->size_bytes(), 0U);
  EXPECT_EQ(buffer->device(), Device::Cpu());
  // Destruction (end of scope) must be well-defined; the deleter-exactly-once
  // half of the contract is counted in BufferTest.ZeroSizeDeleterRunsOnce.
}

TEST(CpuAllocatorTest, HugeAllocationReturnsOutOfMemory) {
  CpuAllocator allocator;
  // Half the address space: std::aligned_alloc itself fails.
  const StatusOr<Buffer> buffer =
      allocator.Allocate(std::numeric_limits<std::size_t>::max() / 2, 64);
  ASSERT_FALSE(buffer.ok());
  EXPECT_TRUE(IsOutOfMemory(buffer.status())) << buffer.status().ToString();
}

TEST(CpuAllocatorTest, SizeOverflowingRoundUpReturnsOutOfMemory) {
  CpuAllocator allocator;
  // So large that rounding up to the alignment would overflow size_t.
  const StatusOr<Buffer> buffer =
      allocator.Allocate(std::numeric_limits<std::size_t>::max() - 1, 4096);
  ASSERT_FALSE(buffer.ok());
  EXPECT_TRUE(IsOutOfMemory(buffer.status())) << buffer.status().ToString();
}

TEST(CpuAllocatorDeathTest, NonPowerOfTwoAlignmentAborts) {
  CpuAllocator allocator;
  EXPECT_DEATH((void)allocator.Allocate(100, 24),
               "alignment must be a power of two");
  EXPECT_DEATH((void)allocator.Allocate(100, 0),
               "alignment must be a power of two");
}

TEST(CpuAllocatorDeathTest, NonPowerOfTwoDefaultAlignmentAborts) {
  EXPECT_DEATH(const CpuAllocator allocator(24),
               "default_alignment must be a power of two");
}

// --- Buffer: engagement, moves, deleter-exactly-once ---

TEST(BufferTest, DefaultConstructedIsDisengaged) {
  const Buffer buffer;
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 0U);
  EXPECT_EQ(buffer.device(), Device::Cpu());
  // Destruction runs no deleter (there is none) — must not crash.
}

TEST(BufferTest, DestructionRunsDeleterExactlyOnce) {
  int count = 0;
  int value = 0;
  {
    const Buffer buffer(&value, sizeof(value), Device::Cpu(),
                        [&count](void*) { ++count; });
    EXPECT_EQ(count, 0);
  }
  EXPECT_EQ(count, 1);
}

TEST(BufferTest, ZeroSizeDeleterRunsOnce) {
  int count = 0;
  {
    const Buffer buffer(nullptr, 0, Device::Cpu(),
                        [&count](void*) { ++count; });
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size_bytes(), 0U);
  }
  EXPECT_EQ(count, 1);
}

TEST(BufferTest, MoveConstructionTransfersOwnership) {
  int count = 0;
  int value = 0;
  {
    Buffer source(&value, sizeof(value), Device::Cpu(),
                  [&count](void*) { ++count; });
    const Buffer dest(std::move(source));
    EXPECT_EQ(dest.data(), &value);
    EXPECT_EQ(dest.size_bytes(), sizeof(value));
    // Reading the moved-from Buffer is deliberate: its state is specified
    // (disengaged, null data, zero size).
    // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size_bytes(), 0U);
    // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(count, 0);  // ownership moved, nothing destroyed yet
  }
  EXPECT_EQ(count, 1);  // both are gone; deleter ran exactly once
}

TEST(BufferTest, MoveAssignmentDestroysCurrentThenTransfers) {
  int count_a = 0;
  int count_b = 0;
  int value_a = 0;
  int value_b = 0;
  {
    Buffer a(&value_a, sizeof(value_a), Device::Cpu(),
             [&count_a](void*) { ++count_a; });
    Buffer b(&value_b, sizeof(value_b), Device::Cpu(),
             [&count_b](void*) { ++count_b; });
    a = std::move(b);
    EXPECT_EQ(count_a, 1);  // a's old allocation freed at assignment
    EXPECT_EQ(count_b, 0);  // b's allocation now owned by a
    EXPECT_EQ(a.data(), &value_b);
  }
  EXPECT_EQ(count_a, 1);
  EXPECT_EQ(count_b, 1);
}

TEST(BufferTest, MoveAssignmentIntoDisengagedBuffer) {
  int count = 0;
  int value = 0;
  {
    Buffer dest;
    {
      Buffer source(&value, sizeof(value), Device::Cpu(),
                    [&count](void*) { ++count; });
      dest = std::move(source);
    }
    EXPECT_EQ(count, 0);  // source (moved-from) destroyed without deleting
    EXPECT_EQ(dest.data(), &value);
  }
  EXPECT_EQ(count, 1);
}

TEST(BufferTest, SelfMoveAssignmentIsSafe) {
  int count = 0;
  int value = 0;
  {
    Buffer buffer(&value, sizeof(value), Device::Cpu(),
                  [&count](void*) { ++count; });
    Buffer& ref = buffer;  // via reference, to sidestep -Wself-move
    buffer = std::move(ref);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(buffer.data(), &value);
    EXPECT_EQ(buffer.size_bytes(), sizeof(value));
  }
  EXPECT_EQ(count, 1);
}

TEST(BufferDeathTest, NullDeleterAborts) {
  int value = 0;
  EXPECT_DEATH(Buffer(&value, sizeof(value), Device::Cpu(), nullptr),
               "engaged Buffer requires a deleter");
}

TEST(BufferDeathTest, DataSizeMismatchAborts) {
  int value = 0;
  EXPECT_DEATH(Buffer(&value, 0, Device::Cpu(), [](void*) {}),
               "data must be null iff size_bytes is 0");
  EXPECT_DEATH(Buffer(nullptr, 4, Device::Cpu(), [](void*) {}),
               "data must be null iff size_bytes is 0");
}

// --- Deleter-exactly-once through the Allocator interface ---

TEST(CountingAllocatorTest, DeleterInvokedExactlyOncePerAllocation) {
  CountingAllocator allocator;
  {
    std::vector<Buffer> buffers;
    for (int i = 0; i < 10; ++i) {
      StatusOr<Buffer> buffer = allocator.Allocate(16, 8);
      ASSERT_TRUE(buffer.ok());
      buffers.push_back(*std::move(buffer));
    }
    EXPECT_EQ(allocator.alloc_count(), 10);
    EXPECT_EQ(allocator.free_count(), 0);
  }
  EXPECT_EQ(allocator.free_count(), 10);
}

TEST(CountingAllocatorTest, MovesDoNotDoubleInvokeDeleter) {
  CountingAllocator allocator;
  {
    StatusOr<Buffer> allocated = allocator.Allocate(16, 8);
    ASSERT_TRUE(allocated.ok());
    Buffer a = *std::move(allocated);
    Buffer b = std::move(a);
    Buffer c;
    c = std::move(b);
    EXPECT_EQ(allocator.free_count(), 0);
  }
  EXPECT_EQ(allocator.free_count(), 1);
}

TEST(CountingAllocatorTest, BufferMayOutliveAllocator) {
  // Deleters are self-contained (design §6): freeing after the allocator is
  // gone is legal for allocators whose deleters don't reference them. Here
  // the counter outlives the allocator, standing in for that pattern.
  int count = 0;
  Buffer buffer;
  {
    int* counter = &count;
    Buffer local(nullptr, 0, Device::Cpu(), [counter](void*) { ++*counter; });
    buffer = std::move(local);
  }
  EXPECT_EQ(count, 0);
  buffer = Buffer();  // destroys the held allocation
  EXPECT_EQ(count, 1);
}

// --- DefaultCpuAllocator ---

TEST(DefaultCpuAllocatorTest, ReturnsStableInstance) {
  Allocator* first = DefaultCpuAllocator();
  Allocator* second = DefaultCpuAllocator();
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first->device(), Device::Cpu());
}

TEST(DefaultCpuAllocatorTest, AllocatesUsableMemory) {
  StatusOr<Buffer> buffer = DefaultCpuAllocator()->Allocate(64, 64);
  ASSERT_TRUE(buffer.ok());
  EXPECT_TRUE(IsAligned(buffer->data(), 64));
  std::memset(buffer->data(), 0xAB, buffer->size_bytes());
}

// --- Thread safety (contract smoke test; CpuAllocator is trivially safe) ---

TEST(CpuAllocatorTest, ConcurrentAllocationIsSafe) {
  CpuAllocator allocator;
  constexpr int kThreads = 4;
  constexpr int kAllocationsPerThread = 100;
  std::vector<std::thread> threads;
  std::vector<int> successes(kThreads, 0);
  threads.reserve(kThreads);
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&allocator, &successes, t] {
      for (int i = 0; i < kAllocationsPerThread; ++i) {
        StatusOr<Buffer> buffer = allocator.Allocate(64, 64);
        if (buffer.ok() && IsAligned(buffer->data(), 64)) {
          ++successes[t];
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (std::size_t t = 0; t < kThreads; ++t) {
    EXPECT_EQ(successes[t], kAllocationsPerThread);
  }
}

}  // namespace
