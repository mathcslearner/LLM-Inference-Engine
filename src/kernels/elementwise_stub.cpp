// CPU-only-build stub for kernels/elementwise_detail.h (M2-T08; design:
// docs/design/cuda-backend.md §2.2): same symbols, no CUDA toolkit. CUDA
// builds compile elementwise.cu instead (the source-list seam).
//
// Unreachable through supported factories: validation in elementwise.cpp
// rejects non-CUDA tensors with InvalidArgument before the seam, and CUDA
// tensors cannot be created on a CPU-only build (Tensor::empty returns
// Unimplemented). A Status rather than a CHECK, per the design §2.2
// taxonomy — a hand-built CUDA-tagged tensor asking for a kernel launch is
// requesting a CUDA resource.

#include "core/status.h"
#include "cuda/stream_handle.h"
#include "kernels/elementwise_detail.h"
#include "tensor/tensor.h"

namespace engine::kernels::detail {
namespace {

[[nodiscard]] core::Status BuiltWithoutCuda(const char* op) {
  return core::UnimplementedError("{}: engine built without CUDA support", op);
}

}  // namespace

core::Status LaunchBinary(BinaryOp op, const tensor::Tensor& /*dst*/,
                          const tensor::Tensor& /*a*/,
                          const tensor::Tensor& /*b*/,
                          cuda::StreamHandle /*stream*/) {
  return BuiltWithoutCuda(op == BinaryOp::kAdd ? "add" : "mul");
}

core::Status LaunchScale(const tensor::Tensor& /*dst*/,
                         const tensor::Tensor& /*src*/, double /*scalar*/,
                         cuda::StreamHandle /*stream*/) {
  return BuiltWithoutCuda("scale");
}

core::Status LaunchCast(const tensor::Tensor& /*dst*/,
                        const tensor::Tensor& /*src*/,
                        cuda::StreamHandle /*stream*/) {
  return BuiltWithoutCuda("cast");
}

}  // namespace engine::kernels::detail
