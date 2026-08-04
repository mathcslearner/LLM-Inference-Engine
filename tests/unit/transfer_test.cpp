#include "common/cuda.h"
#include "core/status.h"
#include "cuda/cuda_utils.h"
#include "cuda/stream.h"
#include "cuda/stream_handle.h"
#include "memory/pinned_allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

// Host↔device transfer tests (M2-T07): the stream overload of ops::copy and
// Tensor::to. This TU compiles on every configuration: validation, H2H
// delegation, and the same-device fast path are build-invariant (design
// §8.1) and run everywhere — including the "mismatched shapes/dtypes →
// InvalidArgument" acceptance criterion — while the GPU tests skip (never
// fail) without a device. Everything here goes through toolkit-free public
// headers; no CUDA-build-only sibling TU is needed.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::cuda::StreamHandle;
using engine::memory::PinnedCpuAllocator;
using engine::tensor::DataType;
using engine::tensor::Device;
using engine::tensor::Shape;
using engine::tensor::Tensor;
namespace ops = engine::tensor::ops;

[[nodiscard]] std::size_t NumBytes(const Tensor& tensor) {
  return static_cast<std::size_t>(tensor.numel()) *
         static_cast<std::size_t>(engine::tensor::itemsize(tensor.dtype()));
}

// Fresh contiguous CPU tensor whose raw bytes hold a deterministic pattern.
// Transfers move bytes without interpreting them, so byte patterns (rather
// than typed values) make the round-trip check exact for every dtype at
// once — including exceptional float bit patterns.
[[nodiscard]] Tensor MakePatternTensor(const Shape& shape, DataType dtype,
                                       std::uint8_t salt) {
  Tensor tensor = Tensor::empty(shape, dtype, Device::Cpu()).value();
  auto* bytes = static_cast<std::uint8_t*>(tensor.data());
  const std::size_t size = NumBytes(tensor);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::uint8_t>(salt + (i * 131U));
  }
  return tensor;
}

[[nodiscard]] bool SameBytes(const Tensor& a, const Tensor& b) {
  return NumBytes(a) == NumBytes(b) &&
         std::memcmp(a.data(), b.data(), NumBytes(a)) == 0;
}

// --- Validation & host paths: every build, no GPU required ---

TEST(TransferTest, StreamCopyRejectsDtypeMismatch) {
  const Tensor dst =
      Tensor::empty(Shape({2, 3}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor src =
      Tensor::empty(Shape({2, 3}), DataType::kInt32, Device::Cpu()).value();
  const Status status = ops::copy(dst, src, StreamHandle());
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

TEST(TransferTest, StreamCopyRejectsShapeMismatch) {
  const Tensor dst =
      Tensor::empty(Shape({2, 3}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor src =
      Tensor::empty(Shape({3, 2}), DataType::kFloat32, Device::Cpu()).value();
  const Status status = ops::copy(dst, src, StreamHandle());
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

// The stream overload requires contiguity (strided device copy needs a
// kernel); the two-argument CPU overload still handles the same views.
TEST(TransferTest, StreamCopyRejectsNonContiguousViews) {
  const Tensor dst = MakePatternTensor(Shape({4, 4}), DataType::kFloat32, 0x11);
  const Tensor src = MakePatternTensor(Shape({4, 4}), DataType::kFloat32, 0x22);
  const Tensor dst_view = dst.slice(1, 0, 2).value();  // inner-dim slice:
  const Tensor src_view = src.slice(1, 0, 2).value();  // non-contiguous
  ASSERT_FALSE(dst_view.is_contiguous());
  const Status status = ops::copy(dst_view, src_view, StreamHandle());
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
  EXPECT_TRUE(ops::copy(dst_view, src_view).ok());
}

// CPU↔CPU delegates to the host path (stream ignored), on every build.
TEST(TransferTest, StreamCopyHostToHostCopiesValues) {
  const Tensor src = ops::arange(0, 6).value();
  const Tensor dst =
      Tensor::empty(src.shape(), src.dtype(), Device::Cpu()).value();
  ASSERT_TRUE(ops::copy(dst, src, StreamHandle()).ok());
  for (std::int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(dst.item<std::int64_t>({i}), i);
  }
}

TEST(TransferTest, StreamCopyHostToHostZeroNumel) {
  const Tensor dst =
      Tensor::empty(Shape({0, 3}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor src =
      Tensor::empty(Shape({0, 3}), DataType::kFloat32, Device::Cpu()).value();
  EXPECT_TRUE(ops::copy(dst, src, StreamHandle()).ok());
}

TEST(TransferTest, StreamCopyIdenticalViewIsNoOp) {
  const Tensor tensor =
      MakePatternTensor(Shape({2, 3}), DataType::kFloat32, 0x33);
  EXPECT_TRUE(ops::copy(tensor, tensor, StreamHandle()).ok());
}

// Same device: the same shared handle, no copy (design §8.3).
TEST(TransferTest, ToSameDeviceReturnsSameHandle) {
  const Tensor tensor =
      MakePatternTensor(Shape({2, 3}), DataType::kFloat32, 0x44);
  const Tensor result = tensor.to(Device::Cpu()).value();
  EXPECT_EQ(result.data(), tensor.data());
}

// Contiguity is checked before the same-device fast path (design §8.3), so
// even a would-be no-op rejects a strided view.
TEST(TransferTest, ToRejectsNonContiguousEvenSameDevice) {
  const Tensor tensor =
      MakePatternTensor(Shape({4, 4}), DataType::kFloat32, 0x55);
  const Tensor view = tensor.slice(1, 0, 2).value();
  const auto result = view.to(Device::Cpu());
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(IsInvalidArgument(result.status())) << result.status().ToString();
}

TEST(TransferDeathTest, UndefinedTensorAborts) {
  const Tensor defined =
      Tensor::empty(Shape({2}), DataType::kFloat32, Device::Cpu()).value();
  EXPECT_DEATH((void)ops::copy(Tensor(), defined, StreamHandle()),
               "undefined Tensor");
  EXPECT_DEATH((void)Tensor().to(Device::Cpu()), "undefined Tensor");
}

#ifndef ENGINE_ENABLE_CUDA

// CPU-only builds (design §2.2 taxonomy): asking for a CUDA destination is
// requesting a CUDA resource.
TEST(TransferTest, ToCudaIsUnimplementedOnCpuOnlyBuild) {
  const Tensor tensor =
      MakePatternTensor(Shape({2, 3}), DataType::kFloat32, 0x66);
  const auto result = tensor.to(Device::Cuda(0));
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(engine::core::IsUnimplemented(result.status()))
      << result.status().ToString();
}

#else  // ENGINE_ENABLE_CUDA

// No GPU required: an out-of-range CUDA index is InvalidArgument (design
// §5.2) — on a GPU-less machine device_count() is 0, so index 0 itself is
// out of range.
TEST(TransferTest, ToCudaOutOfRangeIndexIsInvalidArgument) {
  const Tensor tensor =
      MakePatternTensor(Shape({2, 3}), DataType::kFloat32, 0x77);
  const auto result = tensor.to(Device::Cuda(engine::cuda::device_count()));
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(IsInvalidArgument(result.status())) << result.status().ToString();
}

#endif  // ENGINE_ENABLE_CUDA

// --- GPU tests: on CudaTestFixture (M2-T09), skip without a device ---

// Tests whose subject is a specific stream/event pattern build their own
// objects; the rest use the fixture stream.
class TransferGpuTest : public engine::testing::CudaTestFixture {};

// The M2-T07 acceptance criterion: H2D→D2H round-trips preserve bytes for
// every (non-reserved) dtype. Odd extents exercise non-power-of-two sizes.
TEST_F(TransferGpuTest, RoundTripPreservesBytesForEveryDtype) {
  std::uint8_t salt = 1;
  for (const DataType dtype : engine::tensor::kAllDataTypes) {
    if (dtype == DataType::kInt4 || dtype == DataType::kFP8E4M3) {
      continue;  // reserved dtypes: no tensor can carry them yet
    }
    SCOPED_TRACE(engine::tensor::to_string(dtype));
    const Tensor src = MakePatternTensor(Shape({3, 5, 7}), dtype, salt++);
    const Tensor gpu = src.to(Device::Cuda(0)).value();
    EXPECT_TRUE(gpu.device() == Device::Cuda(0));
    EXPECT_EQ(gpu.dtype(), dtype);
    EXPECT_TRUE(gpu.shape() == src.shape());
    const Tensor back = gpu.to(Device::Cpu()).value();
    ASSERT_TRUE(back.device().is_cpu());
    EXPECT_TRUE(SameBytes(src, back));
  }
}

TEST_F(TransferGpuTest, DeviceToDeviceCopy) {
  const Tensor src = MakePatternTensor(Shape({4, 8}), DataType::kFloat32, 0x5A);
  const Tensor gpu_a = src.to(Device::Cuda(0)).value();
  const Tensor gpu_b =
      Tensor::empty(src.shape(), src.dtype(), Device::Cuda(0)).value();
  ASSERT_TRUE(ops::copy(gpu_b, gpu_a, StreamHandle()).ok());
  const Tensor back = gpu_b.to(Device::Cpu()).value();
  EXPECT_TRUE(SameBytes(src, back));
}

TEST_F(TransferGpuTest, ToSameCudaDeviceReturnsSameHandle) {
  const Tensor src = MakePatternTensor(Shape({2, 2}), DataType::kFloat32, 0x6B);
  const Tensor gpu = src.to(Device::Cuda(0)).value();
  const Tensor same = gpu.to(Device::Cuda(0)).value();
  EXPECT_EQ(same.data(), gpu.data());
}

// The M2-T07 acceptance criterion: async copies on an explicit stream with
// cross-stream event ordering (the design §6.4 pattern) — H2D on the
// fixture stream, D2H on a second stream gated by an event recorded after
// the upload.
TEST_F(TransferGpuTest, AsyncCopyOnStreamWithEventOrdering) {
  engine::cuda::CudaStream stream_b =
      engine::cuda::CudaStream::Create().value();
  const Tensor src =
      MakePatternTensor(Shape({64, 64}), DataType::kFloat32, 0x7C);
  const Tensor gpu =
      Tensor::empty(src.shape(), src.dtype(), Device::Cuda(0), allocator())
          .value();
  ASSERT_TRUE(ops::copy(gpu, src, stream_handle()).ok());
  engine::cuda::CudaEvent uploaded = engine::cuda::CudaEvent::Sync().value();
  ASSERT_TRUE(uploaded.Record(stream()).ok());
  ASSERT_TRUE(stream_b.WaitEvent(uploaded).ok());
  const Tensor out =
      Tensor::empty(src.shape(), src.dtype(), Device::Cpu()).value();
  ASSERT_TRUE(ops::copy(out, gpu, stream_b.handle()).ok());
  ASSERT_TRUE(stream_b.Synchronize().ok());
  EXPECT_TRUE(SameBytes(src, out));
}

// The M2-T07 acceptance criterion: pinned round-trip. Both host tensors live
// in page-locked memory, and same-stream FIFO orders the D2H after the H2D.
TEST_F(TransferGpuTest, PinnedRoundTripOnStream) {
  PinnedCpuAllocator pinned = PinnedCpuAllocator::Create().value();
  const Shape shape({16, 32});
  const Tensor src =
      Tensor::empty(shape, DataType::kFloat32, Device::Cpu(), &pinned).value();
  ASSERT_TRUE(ops::fill_uniform(src, -1.0, 1.0, 7).ok());
  const Tensor dst =
      Tensor::empty(shape, DataType::kFloat32, Device::Cpu(), &pinned).value();
  const Tensor gpu =
      Tensor::empty(shape, DataType::kFloat32, Device::Cuda(0), allocator())
          .value();
  ASSERT_TRUE(ops::copy(gpu, src, stream_handle()).ok());
  ASSERT_TRUE(ops::copy(dst, gpu, stream_handle()).ok());
  ASSERT_TRUE(stream().Synchronize().ok());
  EXPECT_TRUE(SameBytes(src, dst));
}

// to(device, stream) enqueues and returns; the result is consumable after a
// sync point (here an event synchronize — the §6.4 host-consumption rule).
TEST_F(TransferGpuTest, ToWithStreamThenEventSync) {
  const Tensor src = MakePatternTensor(Shape({8, 8}), DataType::kInt32, 0x1F);
  const Tensor gpu = src.to(Device::Cuda(0), stream_handle()).value();
  engine::cuda::CudaEvent done = engine::cuda::CudaEvent::Sync().value();
  ASSERT_TRUE(done.Record(stream()).ok());
  ASSERT_TRUE(done.Synchronize().ok());
  const Tensor back = gpu.to(Device::Cpu()).value();
  EXPECT_TRUE(SameBytes(src, back));
}

TEST_F(TransferGpuTest, ZeroNumelTransferRoundTrip) {
  const Tensor src =
      Tensor::empty(Shape({0, 3}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor gpu = src.to(Device::Cuda(0)).value();
  EXPECT_TRUE(gpu.device() == Device::Cuda(0));
  EXPECT_EQ(gpu.numel(), 0);
  const Tensor back = gpu.to(Device::Cpu()).value();
  EXPECT_EQ(back.numel(), 0);
}

TEST_F(TransferGpuTest, CrossDeviceCopyIsUnimplemented) {
  if (engine::cuda::device_count() < 2) {
    GTEST_SKIP() << "needs two CUDA devices";
  }
  const Tensor src = MakePatternTensor(Shape({2, 2}), DataType::kFloat32, 0x2E);
  const Tensor gpu0 = src.to(Device::Cuda(0)).value();
  const Tensor gpu1 =
      Tensor::empty(src.shape(), src.dtype(), Device::Cuda(1)).value();
  const Status status = ops::copy(gpu1, gpu0, StreamHandle());
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(engine::core::IsUnimplemented(status)) << status.ToString();
  const auto moved = gpu0.to(Device::Cuda(1));
  ASSERT_FALSE(moved.ok());
  EXPECT_TRUE(engine::core::IsUnimplemented(moved.status()))
      << moved.status().ToString();
}

}  // namespace
