#include "memory/pinned_allocator.h"

#include "common/cuda.h"
#include "core/status.h"
#include "cuda/cuda_utils.h"
#include "memory/allocator.h"
#include "tensor/device.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

// PinnedCpuAllocator tests (M2-T07). This TU compiles on every
// configuration: CPU-only builds exercise the stub taxonomy, CUDA builds
// without a GPU exercise the no-device taxonomy, and the allocation tests
// skip (never fail) without a device.

namespace {

using engine::core::StatusOr;
using engine::memory::PinnedCpuAllocator;

#ifndef ENGINE_ENABLE_CUDA

using engine::core::IsUnimplemented;

// CPU-only builds (design §2.2 taxonomy): Create is the only producer of
// instances and reports the build, not a missing device.
TEST(PinnedCpuAllocatorStubTest, CreateIsUnimplemented) {
  const StatusOr<PinnedCpuAllocator> allocator = PinnedCpuAllocator::Create();
  ASSERT_FALSE(allocator.ok());
  EXPECT_TRUE(IsUnimplemented(allocator.status()))
      << allocator.status().ToString();
  EXPECT_NE(allocator.status().message().find("without CUDA"),
            std::string_view::npos)
      << allocator.status().ToString();
}

#else  // ENGINE_ENABLE_CUDA

using engine::core::IsResourceExhausted;
using engine::core::IsUnavailable;
using engine::memory::Buffer;
using engine::tensor::DataType;
using engine::tensor::Device;
using engine::tensor::Shape;
using engine::tensor::Tensor;

[[nodiscard]] bool IsAligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// No GPU: pinning requires a CUDA driver context, reported eagerly (design
// §7.2). Only meaningful on a GPU-less machine — the inverse skip.
TEST(PinnedCpuAllocatorTest, CreateWithoutDeviceIsUnavailable) {
  if (engine::cuda::device_count() > 0) {
    GTEST_SKIP() << "CUDA devices present; the no-device taxonomy does not "
                    "apply";
  }
  const StatusOr<PinnedCpuAllocator> allocator = PinnedCpuAllocator::Create();
  ASSERT_FALSE(allocator.ok());
  EXPECT_TRUE(IsUnavailable(allocator.status()))
      << allocator.status().ToString();
}

TEST(PinnedCpuAllocatorTest, AllocateReturnsAlignedHostWritableBuffer) {
  ENGINE_SKIP_WITHOUT_CUDA();
  PinnedCpuAllocator allocator = PinnedCpuAllocator::Create().value();
  EXPECT_TRUE(allocator.device() == Device::Cpu());
  const Buffer buffer = allocator.Allocate(1024, 256).value();
  ASSERT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 1024U);
  EXPECT_TRUE(buffer.device() == Device::Cpu());
  EXPECT_TRUE(
      IsAligned(buffer.data(), PinnedCpuAllocator::kGuaranteedAlignment));
  // Pinned memory is ordinary host memory: write and read it back.
  auto* bytes = static_cast<std::uint8_t*>(buffer.data());
  for (std::size_t i = 0; i < 1024; ++i) {
    bytes[i] = static_cast<std::uint8_t>(i * 131U + 7U);
  }
  for (std::size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(bytes[i], static_cast<std::uint8_t>(i * 131U + 7U)) << i;
  }
}

TEST(PinnedCpuAllocatorTest, ZeroByteAllocationIsEngagedAndNull) {
  ENGINE_SKIP_WITHOUT_CUDA();
  PinnedCpuAllocator allocator = PinnedCpuAllocator::Create().value();
  const Buffer buffer = allocator.Allocate(0, 64).value();
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 0U);
  EXPECT_TRUE(buffer.device() == Device::Cpu());
  // Destruction runs the deleter on a null pointer — must be a benign no-op.
}

TEST(PinnedCpuAllocatorTest, HugeAllocationIsResourceExhausted) {
  ENGINE_SKIP_WITHOUT_CUDA();
  PinnedCpuAllocator allocator = PinnedCpuAllocator::Create().value();
  const StatusOr<Buffer> huge = allocator.Allocate(std::size_t{1} << 62U, 256);
  ASSERT_FALSE(huge.ok());
  EXPECT_TRUE(IsResourceExhausted(huge.status())) << huge.status().ToString();
  const Buffer small = allocator.Allocate(256, 256).value();
  EXPECT_NE(small.data(), nullptr);
}

// Deleters capture nothing (cudaFreeHost needs no device context), so a
// Buffer may outlive the allocator that produced it — the M1 contract.
TEST(PinnedCpuAllocatorTest, BufferOutlivesAllocator) {
  ENGINE_SKIP_WITHOUT_CUDA();
  Buffer buffer;
  {
    PinnedCpuAllocator allocator = PinnedCpuAllocator::Create().value();
    buffer = allocator.Allocate(512, 256).value();
  }
  EXPECT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size_bytes(), 512U);
}

// The §7.2 claim: tensors over pinned memory are ordinary CPU tensors —
// the same factories, fills, and accessors apply.
TEST(PinnedCpuAllocatorTest, PinnedTensorIsOrdinaryCpuTensor) {
  ENGINE_SKIP_WITHOUT_CUDA();
  PinnedCpuAllocator allocator = PinnedCpuAllocator::Create().value();
  const Tensor tensor = Tensor::empty(Shape({2, 3}), DataType::kFloat32,
                                      Device::Cpu(), &allocator)
                            .value();
  EXPECT_TRUE(tensor.device().is_cpu());
  ASSERT_TRUE(engine::tensor::ops::fill_uniform(tensor, 0.0, 1.0, 42).ok());
  const float value = tensor.item<float>({1, 2});
  EXPECT_GE(value, 0.0F);
  EXPECT_LT(value, 1.0F);
}

TEST(PinnedCpuAllocatorDeathTest, AlignmentMisuseAborts) {
  ENGINE_SKIP_WITHOUT_CUDA();
  PinnedCpuAllocator allocator = PinnedCpuAllocator::Create().value();
  EXPECT_DEATH((void)allocator.Allocate(16, 3), "power of two");
  EXPECT_DEATH((void)allocator.Allocate(16, 512), "exceeds");
}

#endif  // ENGINE_ENABLE_CUDA

}  // namespace
