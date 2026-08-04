#include "common/cuda.h"
#include "cuda/cuda_utils.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

// Self-tests for the shared GPU test infrastructure (M2-T09; design:
// docs/design/cuda-backend.md §10): CudaTestFixture's provided members and
// ExpectTensorsClose's pass/fail reporting. This TU compiles on every
// configuration; without a device every test here skips via the fixture —
// itself a demonstration of the "reports skips, not failures" acceptance
// criterion (the predicate/device_count agreement is covered in
// cuda_utils_test.cpp).

namespace {

using engine::tensor::DataType;
using engine::tensor::Device;
using engine::tensor::Shape;
using engine::tensor::Tensor;
using engine::testing::ExpectTensorsClose;
namespace ops = engine::tensor::ops;

class CudaFixtureGpuTest : public engine::testing::CudaTestFixture {};

TEST_F(CudaFixtureGpuTest, ProvidesStreamAndAllocatorOnDevice0) {
  EXPECT_NE(stream().get(), nullptr);
  EXPECT_EQ(stream().device_index(), 0);
  EXPECT_EQ(stream_handle().get(), stream().get());
  ASSERT_NE(allocator(), nullptr);
  EXPECT_TRUE(allocator()->device() == Device::Cuda(0));
}

TEST_F(CudaFixtureGpuTest, ExpectTensorsCloseAcceptsARoundTrip) {
  const Tensor host = ops::arange(0, 64).value();
  // to(device, stream) enqueues without synchronizing (§6.4) —
  // ExpectTensorsClose's stream sync is what makes the comparison safe.
  const Tensor gpu = host.to(Device::Cuda(0), stream_handle()).value();
  ExpectTensorsClose(gpu, host, stream(), 0.0, 0.0);
}

TEST_F(CudaFixtureGpuTest, ExpectTensorsCloseReportsAMismatch) {
  const Tensor zeros =
      ops::zeros(Shape({8}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor ones =
      ops::ones(Shape({8}), DataType::kFloat32, Device::Cpu()).value();
  const Tensor zeros_gpu = zeros.to(Device::Cuda(0)).value();
  // Exactly one non-fatal failure, carrying allclose's mismatch report.
  EXPECT_NONFATAL_FAILURE(ExpectTensorsClose(zeros_gpu, ones, stream()),
                          "allclose: FAILED");
}

TEST_F(CudaFixtureGpuTest, AllocatorServesDeviceTensors) {
  const Tensor host = ops::arange(0, 32).value();
  const Tensor gpu =
      Tensor::empty(host.shape(), host.dtype(), Device::Cuda(0), allocator())
          .value();
  ASSERT_TRUE(ops::copy(gpu, host, stream_handle()).ok());
  ExpectTensorsClose(gpu, host, stream(), 0.0, 0.0);
}

}  // namespace
