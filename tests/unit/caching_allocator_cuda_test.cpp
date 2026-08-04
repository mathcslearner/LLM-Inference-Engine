// CUDA-build-only tests for memory/caching_allocator.h (M2-T06): the
// acceptance criterion that needs the toolkit directly — release_cached()
// returns memory to the driver, observed as a cudaMemGetInfo free-memory
// delta — plus proof that a reused block is real, writable device memory.
// Registered only when ENGINE_ENABLE_CUDA (tests/unit/CMakeLists.txt); every
// test skips (never fails) without a device. The portable pooling tests that
// compile on all configurations live in caching_allocator_test.cpp.

#include "common/cuda.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "memory/caching_allocator.h"
#include "memory/cuda_allocator.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace {

using engine::memory::Allocator;
using engine::memory::Buffer;
using engine::memory::CachingAllocator;
using engine::memory::DefaultCudaAllocator;

// The M2-T06 acceptance criterion: cached blocks are held back from the
// driver until release_cached(), which actually returns them. Observed via
// cudaMemGetInfo with a generous tolerance — the driver rounds allocations
// and may lazily grow context-internal state (the warm-up triggers the
// latter first), so the block size dominates only when it is large.
// On CudaTestFixture (M2-T09) for the skip guard and device selection; the
// pools under test are built locally.
class CachingAllocatorDriverGpuTest : public engine::testing::CudaTestFixture {
};

TEST_F(CachingAllocatorDriverGpuTest, ReleaseCachedReturnsMemoryToTheDriver) {
  Allocator* const upstream = DefaultCudaAllocator(0).value();
  CachingAllocator pool(upstream);
  constexpr std::size_t kBlockBytes = std::size_t{64} << 20U;  // 64 MiB
  constexpr std::size_t kToleranceBytes = std::size_t{8} << 20U;
  {
    const Buffer warmup = pool.Allocate(kBlockBytes, 256).value();
    ASSERT_NE(warmup.data(), nullptr);
  }
  EXPECT_EQ(pool.release_cached(), kBlockBytes);

  std::size_t free_before = 0;
  std::size_t total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);

  {
    const Buffer buffer = pool.Allocate(kBlockBytes, 256).value();
  }
  // The freed block is cached, not returned: driver-visible free memory is
  // still down by the full block.
  std::size_t free_cached = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_cached, &total), cudaSuccess);
  EXPECT_GE(free_before, free_cached + kBlockBytes - kToleranceBytes);

  EXPECT_EQ(pool.release_cached(), kBlockBytes);
  // Now the memory is back with the driver.
  std::size_t free_released = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_released, &total), cudaSuccess);
  EXPECT_GE(free_released + kToleranceBytes, free_before);
}

// A cache-hit block is real, writable device memory: memset a fresh pattern
// after reuse and round-trip it through a D2H copy.
TEST_F(CachingAllocatorDriverGpuTest, ReusedBlockIsRealDeviceMemory) {
  Allocator* const upstream = DefaultCudaAllocator(0).value();
  CachingAllocator pool(upstream);
  constexpr std::size_t kBytes = 4096;
  void* first = nullptr;
  {
    const Buffer buffer = pool.Allocate(kBytes, 256).value();
    first = buffer.data();
    ASSERT_EQ(cudaMemset(buffer.data(), 0xAB, kBytes), cudaSuccess);
  }
  const Buffer reused = pool.Allocate(kBytes, 256).value();
  ASSERT_EQ(reused.data(), first);
  EXPECT_EQ(pool.stats().hit_count, 1U);
  ASSERT_EQ(cudaMemset(reused.data(), 0x5C, kBytes), cudaSuccess);
  std::vector<unsigned char> host(kBytes, 0);
  ASSERT_EQ(
      cudaMemcpy(host.data(), reused.data(), kBytes, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (const unsigned char byte : host) {
    ASSERT_EQ(byte, 0x5C);
  }
}

}  // namespace
