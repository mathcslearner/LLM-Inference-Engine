// CUDA-build-only tests for memory/cuda_allocator.h (M2-T05): the
// acceptance criteria that need the toolkit directly — the cudaMemGetInfo
// leak-delta check and proof that returned pointers are real device memory.
// Registered only when ENGINE_ENABLE_CUDA (tests/unit/CMakeLists.txt); every
// test skips (never fails) without a device. Portable CudaAllocator tests
// that compile on all configurations live in cuda_allocator_test.cpp.

#include "common/cuda.h"

#include "core/status.h"
#include "memory/allocator.h"
#include "memory/cuda_allocator.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace {

using engine::memory::Buffer;
using engine::memory::CudaAllocator;

// The M2-T05 acceptance criterion: allocate/free cycles are leak-free,
// verified as a cudaMemGetInfo free-memory delta of zero (with a small
// tolerance — the driver may lazily grow context-internal allocations, which
// the warm-up allocation is there to trigger first).
TEST(CudaAllocatorCudaTest, AllocateFreeCyclesAreLeakFree) {
  ENGINE_SKIP_WITHOUT_CUDA();
  CudaAllocator allocator = CudaAllocator::Create(0).value();
  constexpr std::size_t kBlockBytes = std::size_t{1} << 20U;  // 1 MiB
  {
    const Buffer warmup = allocator.Allocate(kBlockBytes, 256).value();
    ASSERT_NE(warmup.data(), nullptr);
  }
  std::size_t free_before = 0;
  std::size_t total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);
  for (int i = 0; i < 64; ++i) {
    const Buffer buffer = allocator.Allocate(kBlockBytes, 256).value();
    ASSERT_NE(buffer.data(), nullptr);
  }
  std::size_t free_after = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
  const std::size_t delta = free_before > free_after ? free_before - free_after
                                                     : free_after - free_before;
  EXPECT_LE(delta, 4 * kBlockBytes)
      << "free memory drifted by " << delta << " bytes across 64 cycles";
}

// The returned pointer is real, writable device memory on the allocator's
// device: a device-side memset round-trips through a D2H copy.
TEST(CudaAllocatorCudaTest, ReturnedMemoryIsRealDeviceMemory) {
  ENGINE_SKIP_WITHOUT_CUDA();
  CudaAllocator allocator = CudaAllocator::Create(0).value();
  constexpr std::size_t kBytes = 4096;
  const Buffer buffer = allocator.Allocate(kBytes, 256).value();
  ASSERT_EQ(cudaMemset(buffer.data(), 0xAB, kBytes), cudaSuccess);
  std::vector<unsigned char> host(kBytes, 0);
  ASSERT_EQ(
      cudaMemcpy(host.data(), buffer.data(), kBytes, cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (const unsigned char byte : host) {
    ASSERT_EQ(byte, 0xAB);
  }
}

}  // namespace
