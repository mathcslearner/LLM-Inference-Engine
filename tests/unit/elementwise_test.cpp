#include "kernels/elementwise.h"

#include "common/cuda.h"
#include "core/status.h"
#include "cuda/stream.h"
#include "cuda/stream_handle.h"
#include "kernels/launch.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>

// Elementwise kernel tests (M2-T08). This TU compiles on every
// configuration: LaunchConfig1D is plain host arithmetic, and the §9.2
// validation contract lives in the always-compiled elementwise.cpp — so the
// "kernels validate inputs and return Status" acceptance criterion runs on
// CPU-only CI (CPU tensors → InvalidArgument on every build; the design
// §2.2 Unimplemented taxonomy is unreachable through supported factories).
// The GPU acceptance tests — each kernel vs the CPU reference via
// ExpectTensorsClose across shapes and dtypes — run on CudaTestFixture
// (M2-T09) and skip (never fail) without a device.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsUnimplemented;
using engine::core::Status;
using engine::cuda::DefaultStream;
using engine::cuda::StreamHandle;
using engine::kernels::kDefaultBlockSize;
using engine::kernels::kMaxGridBlocks;
using engine::kernels::LaunchConfig1D;
using engine::tensor::bfloat16;
using engine::tensor::DataType;
using engine::tensor::Device;
using engine::tensor::float16;
using engine::tensor::Shape;
using engine::tensor::Tensor;
using engine::testing::ExpectTensorsClose;
namespace ops = engine::tensor::ops;

// --- LaunchConfig1D (host arithmetic; every configuration) ---

// constexpr all the way down: config computation is compile-time checkable.
static_assert(LaunchConfig1D(1).grid == 1);
static_assert(LaunchConfig1D(1).block == kDefaultBlockSize);
static_assert(LaunchConfig1D(kDefaultBlockSize).grid == 1);
static_assert(LaunchConfig1D(kDefaultBlockSize + 1).grid == 2);
static_assert(LaunchConfig1D(std::int64_t{2} * kDefaultBlockSize).grid == 2);

TEST(LaunchConfig1DTest, GoldenTable) {
  const std::int64_t full_grid =
      std::int64_t{kMaxGridBlocks} * kDefaultBlockSize;
  EXPECT_EQ(LaunchConfig1D(255).grid, 1U);
  EXPECT_EQ(LaunchConfig1D(257).grid, 2U);
  EXPECT_EQ(LaunchConfig1D(full_grid).grid, kMaxGridBlocks);
  // Beyond one grid's worth the cap holds and the grid-stride loop wraps.
  EXPECT_EQ(LaunchConfig1D(full_grid + 1).grid, kMaxGridBlocks);
  EXPECT_EQ(LaunchConfig1D(INT64_MAX).grid, kMaxGridBlocks);
}

TEST(LaunchConfig1DDeathTest, RequiresPositiveCount) {
  EXPECT_DEATH((void)LaunchConfig1D(0), "requires n > 0");
  EXPECT_DEATH((void)LaunchConfig1D(-1), "requires n > 0");
}

// --- Test helpers ---

[[nodiscard]] Tensor MakeCpu(const Shape& shape, DataType dtype,
                             std::uint64_t seed) {
  Tensor tensor = Tensor::empty(shape, dtype, Device::Cpu()).value();
  if (tensor.numel() > 0) {
    EXPECT_TRUE(ops::fill_uniform(tensor, -4.0, 4.0, seed).ok());
  }
  return tensor;
}

// CPU reference for the same-dtype kernels, mirroring the kernel semantics
// exactly (§9.3): widen each element to float, apply `op` in float, narrow
// back round-to-nearest-even (half.h). Contiguous tensors only.
template <typename T, typename F>
[[nodiscard]] Tensor ElementwiseReference(const Tensor& a, const Tensor& b,
                                          F&& op) {
  Tensor out = Tensor::empty(a.shape(), a.dtype(), Device::Cpu()).value();
  const T* a_data = a.data_ptr<T>();
  const T* b_data = b.data_ptr<T>();
  T* out_data = out.data_ptr<T>();
  for (std::int64_t i = 0; i < a.numel(); ++i) {
    out_data[i] = static_cast<T>(
        op(static_cast<float>(a_data[i]), static_cast<float>(b_data[i])));
  }
  return out;
}

template <typename F>
[[nodiscard]] Tensor ReferenceForDtype(DataType dtype, const Tensor& a,
                                       const Tensor& b, F&& op) {
  switch (dtype) {
    case DataType::kFloat32:
      return ElementwiseReference<float>(a, b, op);
    case DataType::kFloat16:
      return ElementwiseReference<float16>(a, b, op);
    case DataType::kBFloat16:
      return ElementwiseReference<bfloat16>(a, b, op);
    default:
      ADD_FAILURE() << "no reference for dtype "
                    << engine::tensor::to_string(dtype);
      return {};
  }
}

constexpr DataType kKernelDtypes[] = {DataType::kFloat32, DataType::kFloat16,
                                      DataType::kBFloat16};

// The §10.2 shape sweep: sizes below/at/above one block and beyond one
// grid's worth (kMaxGridBlocks * kDefaultBlockSize), so the grid-stride
// wrap path runs. Zero-numel is tested separately (it never launches).
constexpr std::int64_t kSizeSweep[] = {
    1, 255, 256, 257, (std::int64_t{kMaxGridBlocks} * kDefaultBlockSize) + 7};

// GPU acceptance tests run on CudaTestFixture (M2-T09): the fixture skips
// without a device, selects device 0, and provides the per-test stream and
// caching allocator every device tensor here comes from.
class ElementwiseGpuTest : public engine::testing::CudaTestFixture {
 protected:
  // Runs `launch` into a fresh device tensor on the fixture stream and
  // compares against `expected` via ExpectTensorsClose with the per-dtype
  // allclose defaults (fp32 {1e-5, 1e-8}, fp16 {1e-3, 1e-5}, bf16
  // {1.6e-2, 1e-5}) — comfortably loose: kernel and reference share the
  // widen-to-float arithmetic, so the results should in fact be
  // bit-identical.
  template <typename LaunchFn>
  void RunAndCompare(const Tensor& expected, LaunchFn&& launch) {
    const Tensor dst = Tensor::empty(expected.shape(), expected.dtype(),
                                     Device::Cuda(0), allocator())
                           .value();
    const Status status = launch(dst, stream_handle());
    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectTensorsClose(dst, expected, stream());
  }
};

// --- Validation (every configuration; the M2-T08 acceptance criterion) ---
//
// Check order in elementwise.cpp is shape → dtype → contiguity → device, so
// everything below is exercisable with CPU tensors on CPU-only CI. The
// device check itself is the last one: fully-valid-but-CPU operands are
// InvalidArgument on every build.

TEST(ElementwiseValidationTest, AddShapeMismatch) {
  const Tensor dst =
      Tensor::empty(Shape({2, 3}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor a =
      Tensor::empty(Shape({2, 3}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor b =
      Tensor::empty(Shape({3, 2}), DataType::kFloat32, Device::Cpu()).value();
  const Status status = engine::kernels::add(dst, a, b, StreamHandle());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

TEST(ElementwiseValidationTest, AddDtypeMismatch) {
  const Tensor dst =
      Tensor::empty(Shape({4}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor a =
      Tensor::empty(Shape({4}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor b =
      Tensor::empty(Shape({4}), DataType::kFloat16, Device::Cpu()).value();
  const Status status = engine::kernels::add(dst, a, b, StreamHandle());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

TEST(ElementwiseValidationTest, AddIntegerDtypeUnimplemented) {
  const Tensor dst =
      Tensor::empty(Shape({4}), DataType::kInt32, Device::Cpu()).value();
  const Tensor a =
      Tensor::empty(Shape({4}), DataType::kInt32, Device::Cpu()).value();
  const Tensor b =
      Tensor::empty(Shape({4}), DataType::kInt32, Device::Cpu()).value();
  const Status status = engine::kernels::add(dst, a, b, StreamHandle());
  EXPECT_TRUE(IsUnimplemented(status)) << status.ToString();
}

TEST(ElementwiseValidationTest, AddNonContiguous) {
  // Inner-dim slices: same shape and dtype, legitimately non-contiguous.
  const Tensor base =
      Tensor::empty(Shape({4, 4}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor view = base.slice(1, 0, 2).value();
  const Status status = engine::kernels::add(view, view, view, StreamHandle());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

TEST(ElementwiseValidationTest, AddRequiresCudaTensors) {
  // Valid in every respect except placement — rejected with InvalidArgument
  // on every build (not Unimplemented; the CUDA-resource taxonomy is for
  // CUDA-tagged requests, which supported factories cannot produce here).
  const Tensor dst =
      Tensor::empty(Shape({4}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor a = MakeCpu(Shape({4}), DataType::kFloat32, 1);
  const Tensor b = MakeCpu(Shape({4}), DataType::kFloat32, 2);
  const Status status = engine::kernels::add(dst, a, b, StreamHandle());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

TEST(ElementwiseValidationTest, MulValidatesLikeAdd) {
  const Tensor dst =
      Tensor::empty(Shape({2}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor other =
      Tensor::empty(Shape({3}), DataType::kFloat32, Device::Cpu()).value();
  EXPECT_TRUE(IsInvalidArgument(
      engine::kernels::mul(dst, other, other, StreamHandle())));
  EXPECT_TRUE(
      IsInvalidArgument(engine::kernels::mul(dst, dst, dst, StreamHandle())));
}

TEST(ElementwiseValidationTest, ScaleValidates) {
  const Tensor f32 =
      Tensor::empty(Shape({2}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor f16 =
      Tensor::empty(Shape({2}), DataType::kFloat16, Device::Cpu()).value();
  const Tensor i32 =
      Tensor::empty(Shape({2}), DataType::kInt32, Device::Cpu()).value();
  EXPECT_TRUE(
      IsInvalidArgument(engine::kernels::scale(f32, f16, 2.0, StreamHandle())));
  EXPECT_TRUE(
      IsUnimplemented(engine::kernels::scale(i32, i32, 2.0, StreamHandle())));
  EXPECT_TRUE(
      IsInvalidArgument(engine::kernels::scale(f32, f32, 2.0, StreamHandle())));
}

TEST(ElementwiseValidationTest, CastShapeMismatch) {
  const Tensor dst =
      Tensor::empty(Shape({2}), DataType::kFloat16, Device::Cpu()).value();
  const Tensor src =
      Tensor::empty(Shape({3}), DataType::kFloat32, Device::Cpu()).value();
  EXPECT_TRUE(
      IsInvalidArgument(engine::kernels::cast(dst, src, StreamHandle())));
}

TEST(ElementwiseValidationTest, CastUnsupportedDtypePair) {
  const Tensor f32 =
      Tensor::empty(Shape({2}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor i32 =
      Tensor::empty(Shape({2}), DataType::kInt32, Device::Cpu()).value();
  const Tensor boolean =
      Tensor::empty(Shape({2}), DataType::kBool, Device::Cpu()).value();
  // Either side outside {fp32, fp16, bf16} → Unimplemented (integer casts
  // stay CPU-only until a consumer exists, §9.1).
  EXPECT_TRUE(IsUnimplemented(engine::kernels::cast(i32, f32, StreamHandle())));
  EXPECT_TRUE(IsUnimplemented(engine::kernels::cast(f32, i32, StreamHandle())));
  EXPECT_TRUE(
      IsUnimplemented(engine::kernels::cast(f32, boolean, StreamHandle())));
}

TEST(ElementwiseValidationTest, CastRequiresCudaTensors) {
  const Tensor dst =
      Tensor::empty(Shape({2}), DataType::kFloat16, Device::Cpu()).value();
  const Tensor src = MakeCpu(Shape({2}), DataType::kFloat32, 3);
  EXPECT_TRUE(
      IsInvalidArgument(engine::kernels::cast(dst, src, StreamHandle())));
}

TEST(ElementwiseDeathTest, UndefinedHandleChecks) {
  const Tensor defined =
      Tensor::empty(Shape({2}), DataType::kFloat32, Device::Cpu()).value();
  EXPECT_DEATH(
      (void)engine::kernels::add(Tensor(), defined, defined, StreamHandle()),
      "undefined Tensor");
  EXPECT_DEATH(
      (void)engine::kernels::scale(defined, Tensor(), 2.0, StreamHandle()),
      "undefined Tensor");
  EXPECT_DEATH((void)engine::kernels::cast(defined, Tensor(), StreamHandle()),
               "undefined Tensor");
}

// --- GPU acceptance (skip without a device; the §10.2 pattern) ---

TEST_F(ElementwiseGpuTest, AddMatchesCpuReferenceAcrossShapesAndDtypes) {
  for (const DataType dtype : kKernelDtypes) {
    for (const std::int64_t n : kSizeSweep) {
      SCOPED_TRACE(testing::Message()
                   << "dtype " << to_string(dtype) << ", n " << n);
      const Tensor a =
          MakeCpu(Shape({n}), dtype, static_cast<std::uint64_t>(100 + n));
      const Tensor b =
          MakeCpu(Shape({n}), dtype, static_cast<std::uint64_t>(200 + n));
      const Tensor expected = ReferenceForDtype(
          dtype, a, b, [](float x, float y) { return x + y; });
      const Tensor a_gpu = a.to(Device::Cuda(0)).value();
      const Tensor b_gpu = b.to(Device::Cuda(0)).value();
      RunAndCompare(expected, [&](const Tensor& dst, StreamHandle stream) {
        return engine::kernels::add(dst, a_gpu, b_gpu, stream);
      });
    }
  }
}

TEST_F(ElementwiseGpuTest, MulMatchesCpuReferenceAcrossShapesAndDtypes) {
  for (const DataType dtype : kKernelDtypes) {
    for (const std::int64_t n : kSizeSweep) {
      SCOPED_TRACE(testing::Message()
                   << "dtype " << to_string(dtype) << ", n " << n);
      const Tensor a =
          MakeCpu(Shape({n}), dtype, static_cast<std::uint64_t>(300 + n));
      const Tensor b =
          MakeCpu(Shape({n}), dtype, static_cast<std::uint64_t>(400 + n));
      const Tensor expected = ReferenceForDtype(
          dtype, a, b, [](float x, float y) { return x * y; });
      const Tensor a_gpu = a.to(Device::Cuda(0)).value();
      const Tensor b_gpu = b.to(Device::Cuda(0)).value();
      RunAndCompare(expected, [&](const Tensor& dst, StreamHandle stream) {
        return engine::kernels::mul(dst, a_gpu, b_gpu, stream);
      });
    }
  }
}

TEST_F(ElementwiseGpuTest, ScaleMatchesCpuReferenceAcrossShapesAndDtypes) {
  constexpr double kScalar = -2.5;
  for (const DataType dtype : kKernelDtypes) {
    for (const std::int64_t n : kSizeSweep) {
      SCOPED_TRACE(testing::Message()
                   << "dtype " << to_string(dtype) << ", n " << n);
      const Tensor src =
          MakeCpu(Shape({n}), dtype, static_cast<std::uint64_t>(500 + n));
      // The kernel narrows the scalar to float once, host-side — mirror it.
      const Tensor expected = ReferenceForDtype(
          dtype, src, src,
          [](float x, float) { return x * static_cast<float>(kScalar); });
      const Tensor src_gpu = src.to(Device::Cuda(0)).value();
      RunAndCompare(expected, [&](const Tensor& dst, StreamHandle stream) {
        return engine::kernels::scale(dst, src_gpu, kScalar, stream);
      });
    }
  }
}

TEST_F(ElementwiseGpuTest, CastMatchesCpuCastForAllFloatingPairs) {
  const Shape shape({4099});  // non-multiple of the block size
  for (const DataType src_dtype : kKernelDtypes) {
    for (const DataType dst_dtype : kKernelDtypes) {
      SCOPED_TRACE(testing::Message()
                   << to_string(src_dtype) << " -> " << to_string(dst_dtype));
      const Tensor src = MakeCpu(shape, src_dtype, 600);
      const Tensor expected = ops::cast(src, dst_dtype).value();
      const Tensor src_gpu = src.to(Device::Cuda(0)).value();
      const Tensor dst =
          Tensor::empty(shape, dst_dtype, Device::Cuda(0), allocator()).value();
      const Status status =
          engine::kernels::cast(dst, src_gpu, stream_handle());
      ASSERT_TRUE(status.ok()) << status.ToString();
      // Exact tolerances: kernel and ops::cast both widen to float and
      // narrow round-to-nearest-even (§9.3 / half.h), so every pair —
      // including the same-dtype deep copy — must agree bit-for-bit on
      // these fill_uniform values (finite, no signaling-NaN quieting
      // involved).
      ExpectTensorsClose(dst, expected, stream(), 0.0, 0.0);
    }
  }
}

TEST_F(ElementwiseGpuTest, NullStreamHandleUsesDefaultStream) {
  const Tensor a = MakeCpu(Shape({1000}), DataType::kFloat32, 700);
  const Tensor b = MakeCpu(Shape({1000}), DataType::kFloat32, 701);
  const Tensor expected = ReferenceForDtype(
      DataType::kFloat32, a, b, [](float x, float y) { return x + y; });
  const Tensor a_gpu = a.to(Device::Cuda(0)).value();
  const Tensor b_gpu = b.to(Device::Cuda(0)).value();
  const Tensor dst =
      Tensor::empty(a.shape(), a.dtype(), Device::Cuda(0), allocator()).value();
  // Enqueue on the engine default stream (null handle, §6.3); comparing via
  // that same stream synchronizes it before the download (§8.3).
  const Status status = engine::kernels::add(dst, a_gpu, b_gpu, StreamHandle());
  ASSERT_TRUE(status.ok()) << status.ToString();
  ExpectTensorsClose(dst, expected, DefaultStream(0));
}

TEST_F(ElementwiseGpuTest, DstMayAliasInput) {
  const Tensor a = MakeCpu(Shape({512}), DataType::kFloat32, 800);
  const Tensor b = MakeCpu(Shape({512}), DataType::kFloat32, 801);
  const Tensor expected = ReferenceForDtype(
      DataType::kFloat32, a, b, [](float x, float y) { return x + y; });
  const Tensor a_gpu = a.to(Device::Cuda(0)).value();
  const Tensor b_gpu = b.to(Device::Cuda(0)).value();
  // In-place: dst IS a_gpu (identical handle → identical data pointer),
  // well-defined for the same-dtype kernels (elementwise.h aliasing rule).
  const Status status =
      engine::kernels::add(a_gpu, a_gpu, b_gpu, stream_handle());
  ASSERT_TRUE(status.ok()) << status.ToString();
  ExpectTensorsClose(a_gpu, expected, stream());
}

TEST_F(ElementwiseGpuTest, MixedDevicePlacementIsInvalid) {
  const Tensor cpu = MakeCpu(Shape({4}), DataType::kFloat32, 900);
  const Tensor gpu = cpu.to(Device::Cuda(0)).value();
  const Tensor dst = Tensor::empty(Shape({4}), DataType::kFloat32,
                                   Device::Cuda(0), allocator())
                         .value();
  const Status status = engine::kernels::add(dst, cpu, gpu, StreamHandle());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
}

TEST_F(ElementwiseGpuTest, ZeroNumelReturnsOkWithoutLaunching) {
  const Shape shape({0});
  const Tensor dst =
      Tensor::empty(shape, DataType::kFloat32, Device::Cuda(0)).value();
  const Tensor a =
      Tensor::empty(shape, DataType::kFloat32, Device::Cuda(0)).value();
  const Tensor b =
      Tensor::empty(shape, DataType::kFloat32, Device::Cuda(0)).value();
  EXPECT_TRUE(engine::kernels::add(dst, a, b, StreamHandle()).ok());
  EXPECT_TRUE(engine::kernels::mul(dst, a, b, StreamHandle()).ok());
  EXPECT_TRUE(engine::kernels::scale(dst, a, 2.0, StreamHandle()).ok());
  EXPECT_TRUE(engine::kernels::cast(dst, a, StreamHandle()).ok());
}

}  // namespace
