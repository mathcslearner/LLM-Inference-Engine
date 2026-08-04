#include "memory/cuda_allocator.h"

#include "common/cuda.h"
#include "core/status.h"
#include "cuda/cuda_utils.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

// CudaAllocator tests (M2-T05). This TU compiles on every configuration:
// CPU-only builds exercise the stub taxonomy, CUDA builds exercise index
// validation without a GPU, and the allocation tests skip (never fail)
// without a device. The toolkit-touching acceptance tests (cudaMemGetInfo
// leak check, device-memory round-trip) live in cuda_allocator_cuda_test.cpp,
// which only CUDA builds compile.

namespace {

using engine::core::StatusOr;
using engine::memory::Allocator;
using engine::memory::CudaAllocator;
using engine::memory::DefaultCudaAllocator;

#ifndef ENGINE_ENABLE_CUDA

using engine::core::IsUnimplemented;

// CPU-only builds (design §2.2 taxonomy): the factories are the only
// producers of instances and both report the build, not a missing device.
TEST(CudaAllocatorStubTest, CreateIsUnimplemented) {
  const StatusOr<CudaAllocator> allocator = CudaAllocator::Create(0);
  ASSERT_FALSE(allocator.ok());
  EXPECT_TRUE(IsUnimplemented(allocator.status()))
      << allocator.status().ToString();
  EXPECT_NE(allocator.status().message().find("without CUDA"),
            std::string_view::npos)
      << allocator.status().ToString();
}

TEST(CudaAllocatorStubTest, DefaultCudaAllocatorIsUnimplemented) {
  const StatusOr<Allocator*> allocator = DefaultCudaAllocator(0);
  ASSERT_FALSE(allocator.ok());
  EXPECT_TRUE(IsUnimplemented(allocator.status()))
      << allocator.status().ToString();
}

#else  // ENGINE_ENABLE_CUDA

using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::memory::Buffer;
using engine::tensor::Device;

[[nodiscard]] bool IsAligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// No GPU required: out-of-range indices are InvalidArgument naming the index
// and the visible-device count (design §5.2). On a GPU-less machine
// device_count() is 0, so index 0 is itself out of range.
TEST(CudaAllocatorTest, CreateRejectsOutOfRangeIndex) {
  for (const int index : {-1, engine::cuda::device_count()}) {
    const StatusOr<CudaAllocator> allocator = CudaAllocator::Create(index);
    ASSERT_FALSE(allocator.ok()) << "index " << index;
    EXPECT_TRUE(IsInvalidArgument(allocator.status()))
        << allocator.status().ToString();
  }
}

TEST(CudaAllocatorTest, DefaultCudaAllocatorRejectsOutOfRangeIndex) {
  const StatusOr<Allocator*> allocator =
      DefaultCudaAllocator(engine::cuda::device_count());
  ASSERT_FALSE(allocator.ok());
  EXPECT_TRUE(IsInvalidArgument(allocator.status()))
      << allocator.status().ToString();
}

// GPU tests on CudaTestFixture (M2-T09): the fixture supplies the skip
// guard and device selection; the allocators under test are built locally,
// and the fixture's own pool is deliberately unused.
class CudaAllocatorGpuTest : public engine::testing::CudaTestFixture {};
class CudaAllocatorGpuDeathTest : public engine::testing::CudaTestFixture {};

TEST_F(CudaAllocatorGpuTest, AllocateReturnsEngagedDeviceTaggedBuffer) {
  CudaAllocator allocator = CudaAllocator::Create(0).value();
  EXPECT_TRUE(allocator.device() == Device::Cuda(0));
  const Buffer buffer = allocator.Allocate(1024, 256).value();
  EXPECT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 1024U);
  EXPECT_TRUE(buffer.device() == Device::Cuda(0));
  EXPECT_TRUE(IsAligned(buffer.data(), CudaAllocator::kGuaranteedAlignment));
}

TEST_F(CudaAllocatorGpuTest, ZeroByteAllocationIsEngagedAndNull) {
  CudaAllocator allocator = CudaAllocator::Create(0).value();
  const Buffer buffer = allocator.Allocate(0, 64).value();
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 0U);
  EXPECT_TRUE(buffer.device() == Device::Cuda(0));
  // Destruction runs the deleter on a null pointer — must be a benign no-op.
}

// The M2-T05 acceptance criterion: an impossible allocation is a Status, not
// a crash — and the allocator stays usable afterward.
TEST_F(CudaAllocatorGpuTest, HugeAllocationIsResourceExhausted) {
  CudaAllocator allocator = CudaAllocator::Create(0).value();
  const StatusOr<Buffer> huge = allocator.Allocate(std::size_t{1} << 62U, 256);
  ASSERT_FALSE(huge.ok());
  EXPECT_TRUE(IsResourceExhausted(huge.status())) << huge.status().ToString();
  const Buffer small = allocator.Allocate(256, 256).value();
  EXPECT_NE(small.data(), nullptr);
}

// Deleters are self-contained (capture only the device index), so a Buffer
// may outlive the allocator that produced it — the M1 Allocator contract.
TEST_F(CudaAllocatorGpuTest, BufferOutlivesAllocator) {
  Buffer buffer;
  {
    CudaAllocator allocator = CudaAllocator::Create(0).value();
    buffer = allocator.Allocate(512, 256).value();
  }
  EXPECT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 512U);
  // `buffer` is destroyed after the allocator — the deleter must still free.
}

TEST_F(CudaAllocatorGpuTest, DefaultCudaAllocatorIsPerDeviceSingleton) {
  Allocator* const first = DefaultCudaAllocator(0).value();
  Allocator* const second = DefaultCudaAllocator(0).value();
  EXPECT_EQ(first, second);
  ASSERT_NE(first, nullptr);
  EXPECT_TRUE(first->device() == Device::Cuda(0));
}

TEST_F(CudaAllocatorGpuDeathTest, AlignmentMisuseAborts) {
  CudaAllocator allocator = CudaAllocator::Create(0).value();
  EXPECT_DEATH((void)allocator.Allocate(16, 3), "power of two");
  EXPECT_DEATH((void)allocator.Allocate(16, 512), "exceeds");
}

#endif  // ENGINE_ENABLE_CUDA

}  // namespace
