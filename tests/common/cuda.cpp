#include "common/cuda.h"

#include "core/status.h"
#include "cuda/cuda_utils.h"
#include "cuda/stream.h"
#include "memory/allocator.h"
#include "memory/cuda_allocator.h"
#include "tensor/device.h"
#include "tensor/ops.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <utility>

namespace engine::testing {

bool HasCudaDevice() { return cuda::device_count() > 0; }

void CudaTestFixture::SetUp() {
  ENGINE_SKIP_WITHOUT_CUDA();
  device_guard_.emplace(0);
  core::StatusOr<cuda::CudaStream> stream = cuda::CudaStream::Create();
  ASSERT_TRUE(stream.ok()) << stream.status();
  stream_.emplace(*std::move(stream));
  const core::StatusOr<memory::Allocator*> upstream =
      memory::DefaultCudaAllocator(0);
  ASSERT_TRUE(upstream.ok()) << upstream.status();
  allocator_.emplace(*upstream);
}

void CudaTestFixture::TearDown() {
  // Skipped tests engaged nothing; a fatal SetUp failure may have engaged a
  // prefix, which the optionals unwind in reverse construction order.
  if (stream_.has_value()) {
    const core::Status status = stream_->Synchronize();
    EXPECT_TRUE(status.ok()) << status.ToString();
  }
  allocator_.reset();  // CHECKs no live Buffers, returns cached blocks
  stream_.reset();
  device_guard_.reset();
}

void ExpectTensorsClose(const tensor::Tensor& actual,
                        const tensor::Tensor& expected,
                        cuda::CudaStream& stream, std::optional<double> rtol,
                        std::optional<double> atol) {
  const core::Status sync = stream.Synchronize();
  ASSERT_TRUE(sync.ok()) << sync.ToString();
  // Stream-less to(): enqueues on the engine default stream and
  // synchronizes (§8.3) — safe here because `stream` has already drained,
  // so the download observes the completed writes. Same handle for a CPU
  // `actual`.
  const core::StatusOr<tensor::Tensor> host = actual.to(tensor::Device::Cpu());
  ASSERT_TRUE(host.ok()) << host.status();
  const core::StatusOr<tensor::ops::AllCloseResult> close =
      tensor::ops::allclose(*host, expected, rtol, atol);
  ASSERT_TRUE(close.ok()) << close.status();
  EXPECT_TRUE(close->allclose) << close->Summary();
}

}  // namespace engine::testing
