// Always-compiled half of the elementwise launchers (M2-T08; design:
// docs/design/cuda-backend.md §9.2): the validation contract and the
// numel() == 0 guard, on every configuration — so the "kernels validate
// inputs and return Status" acceptance criterion runs on CPU-only CI (the
// M2-T07 transfer precedent). The toolkit-touching launch sits behind
// elementwise_detail.h.

#include "kernels/elementwise.h"

#include "core/check.h"
#include "core/status.h"
#include "cuda/stream_handle.h"
#include "kernels/elementwise_detail.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <initializer_list>

namespace engine::kernels {
namespace {

using core::InvalidArgumentError;
using core::OkStatus;
using core::Status;
using core::UnimplementedError;
using tensor::DataType;
using tensor::Tensor;

// The dtypes M2 kernels dispatch over (design §9.1): the non-reserved
// floating kinds. Integer variants are added when a consumer exists.
[[nodiscard]] bool HasGpuKernelDtype(DataType dtype) {
  return dtype == DataType::kFloat32 || dtype == DataType::kFloat16 ||
         dtype == DataType::kBFloat16;
}

// The §9.2 validation contract, shared by all four launchers. `operands`
// pairs each tensor with its role name for error messages; the first entry
// is dst, whose shape/device anchor the comparisons. Check order — shape,
// dtype (per-launcher, before this is called for the parts that differ),
// contiguity, then device — puts the data-driven checks first so they are
// exercisable with CPU tensors on CPU-only CI (CUDA tensors cannot be
// constructed there).
struct Operand {
  const char* role;
  const Tensor* tensor;
};

[[nodiscard]] Status ValidateShapes(const char* op,
                                    std::initializer_list<Operand> operands) {
  const Tensor& dst = *operands.begin()->tensor;
  for (const Operand& operand : operands) {
    CHECK(operand.tensor->defined(), "{}: undefined Tensor for {}", op,
          operand.role);
    if (operand.tensor->shape() != dst.shape()) {
      return InvalidArgumentError("{}: shape mismatch: {} is {}, dst is {}", op,
                                  operand.role, operand.tensor->shape(),
                                  dst.shape());
    }
  }
  return OkStatus();
}

[[nodiscard]] Status ValidatePlacement(
    const char* op, std::initializer_list<Operand> operands) {
  const Tensor& dst = *operands.begin()->tensor;
  for (const Operand& operand : operands) {
    if (!operand.tensor->is_contiguous()) {
      return InvalidArgumentError(
          "{}: {} must be contiguous (strides {}); strided elementwise "
          "kernels are not implemented",
          op, operand.role, operand.tensor->strides());
    }
    if (operand.tensor->device() != dst.device()) {
      return InvalidArgumentError(
          "{}: all tensors must be on one device; {} is on {}, dst is on {}",
          op, operand.role, operand.tensor->device().ToString(),
          dst.device().ToString());
    }
  }
  if (!dst.device().is_cuda()) {
    return InvalidArgumentError("{}: requires CUDA tensors; device is {}", op,
                                dst.device().ToString());
  }
  return OkStatus();
}

// add/mul/scale: one floating dtype across all operands.
[[nodiscard]] Status ValidateUniformDtype(
    const char* op, std::initializer_list<Operand> operands) {
  const Tensor& dst = *operands.begin()->tensor;
  for (const Operand& operand : operands) {
    if (operand.tensor->dtype() != dst.dtype()) {
      return InvalidArgumentError(
          "{}: dtype mismatch: {} is {}, dst is {} (dtype conversion is "
          "cast)",
          op, operand.role, tensor::to_string(operand.tensor->dtype()),
          tensor::to_string(dst.dtype()));
    }
  }
  if (!HasGpuKernelDtype(dst.dtype())) {
    return UnimplementedError(
        "{}: no GPU kernel for dtype {} (floating dtypes only: "
        "fp32/fp16/bf16)",
        op, tensor::to_string(dst.dtype()));
  }
  return OkStatus();
}

[[nodiscard]] Status ValidateBinary(const char* op, const Tensor& dst,
                                    const Tensor& a, const Tensor& b) {
  const std::initializer_list<Operand> operands = {
      {.role = "dst", .tensor = &dst},
      {.role = "a", .tensor = &a},
      {.role = "b", .tensor = &b}};
  RETURN_IF_ERROR(ValidateShapes(op, operands));
  RETURN_IF_ERROR(ValidateUniformDtype(op, operands));
  return ValidatePlacement(op, operands);
}

}  // namespace

Status add(const Tensor& dst, const Tensor& a, const Tensor& b,
           cuda::StreamHandle stream) {
  RETURN_IF_ERROR(ValidateBinary("add", dst, a, b));
  if (dst.numel() == 0) {
    return OkStatus();
  }
  return detail::LaunchBinary(detail::BinaryOp::kAdd, dst, a, b, stream);
}

Status mul(const Tensor& dst, const Tensor& a, const Tensor& b,
           cuda::StreamHandle stream) {
  RETURN_IF_ERROR(ValidateBinary("mul", dst, a, b));
  if (dst.numel() == 0) {
    return OkStatus();
  }
  return detail::LaunchBinary(detail::BinaryOp::kMul, dst, a, b, stream);
}

Status scale(const Tensor& dst, const Tensor& src, double scalar,
             cuda::StreamHandle stream) {
  const std::initializer_list<Operand> operands = {
      {.role = "dst", .tensor = &dst}, {.role = "src", .tensor = &src}};
  RETURN_IF_ERROR(ValidateShapes("scale", operands));
  RETURN_IF_ERROR(ValidateUniformDtype("scale", operands));
  RETURN_IF_ERROR(ValidatePlacement("scale", operands));
  if (dst.numel() == 0) {
    return OkStatus();
  }
  return detail::LaunchScale(dst, src, scalar, stream);
}

Status cast(const Tensor& dst, const Tensor& src, cuda::StreamHandle stream) {
  const std::initializer_list<Operand> operands = {
      {.role = "dst", .tensor = &dst}, {.role = "src", .tensor = &src}};
  RETURN_IF_ERROR(ValidateShapes("cast", operands));
  // Unlike the uniform-dtype launchers, any {fp32, fp16, bf16} pair —
  // including equal dtypes, a device deep copy. Outside that set the cast
  // is not implemented on GPU (integer casts stay CPU-only, §9.1).
  if (!HasGpuKernelDtype(src.dtype()) || !HasGpuKernelDtype(dst.dtype())) {
    return UnimplementedError(
        "cast: no GPU kernel for dtype pair {} -> {} (floating dtypes only: "
        "fp32/fp16/bf16)",
        tensor::to_string(src.dtype()), tensor::to_string(dst.dtype()));
  }
  RETURN_IF_ERROR(ValidatePlacement("cast", operands));
  if (dst.numel() == 0) {
    return OkStatus();
  }
  return detail::LaunchCast(dst, src, stream);
}

}  // namespace engine::kernels
