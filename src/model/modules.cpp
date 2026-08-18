#include "model/modules.h"

#include "core/status.h"
#include "cpu/ops.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace engine::model {

namespace {

// The float storage dtypes a weight/bias may use (model-execution.md §3.3):
// checkpoints are f32/f16/bf16; the reference widens f16/bf16 per element.
[[nodiscard]] bool IsSupportedFloat(tensor::DataType dtype) {
  return dtype == tensor::DataType::kFloat32 ||
         dtype == tensor::DataType::kFloat16 ||
         dtype == tensor::DataType::kBFloat16;
}

}  // namespace

core::StatusOr<ReferenceLinear> ReferenceLinear::Create(
    tensor::Tensor weight, std::optional<tensor::Tensor> bias) {
  if (!weight.defined()) {
    return core::InvalidArgumentError("ReferenceLinear: weight is undefined");
  }
  if (weight.shape().rank() != 2) {
    return core::InvalidArgumentError(
        "ReferenceLinear: weight must be rank-2 [out, in], got rank-{}",
        weight.shape().rank());
  }
  if (!weight.is_contiguous()) {
    return core::InvalidArgumentError(
        "ReferenceLinear: weight must be contiguous");
  }
  if (!IsSupportedFloat(weight.dtype())) {
    return core::InvalidArgumentError(
        "ReferenceLinear: weight dtype must be f32/f16/bf16, got {}",
        tensor::to_string(weight.dtype()));
  }
  const std::int64_t out_features = weight.shape().dim(0);
  const std::int64_t in_features = weight.shape().dim(1);

  if (bias.has_value()) {
    if (!bias->defined()) {
      return core::InvalidArgumentError("ReferenceLinear: bias is undefined");
    }
    if (bias->shape().rank() != 1) {
      return core::InvalidArgumentError(
          "ReferenceLinear: bias must be rank-1 [out], got rank-{}",
          bias->shape().rank());
    }
    if (!bias->is_contiguous()) {
      return core::InvalidArgumentError(
          "ReferenceLinear: bias must be contiguous");
    }
    if (!IsSupportedFloat(bias->dtype())) {
      return core::InvalidArgumentError(
          "ReferenceLinear: bias dtype must be f32/f16/bf16, got {}",
          tensor::to_string(bias->dtype()));
    }
    if (bias->shape().dim(0) != out_features) {
      return core::InvalidArgumentError(
          "ReferenceLinear: bias length {} must equal out_features {}",
          bias->shape().dim(0), out_features);
    }
  }

  return ReferenceLinear(std::move(weight), std::move(bias), in_features,
                         out_features);
}

core::Status ReferenceLinear::forward(const tensor::Tensor& x,
                                      tensor::Tensor& y) const {
  // Validate the activation tensors here so the messages name `x`/`y` (the
  // Linear contract's terms) rather than leaking cpu::gemm's a/b/c. `T` is not
  // constrained (0 rows would be caught by gemm's M >= 1, but Linear callers
  // always pass T >= 1). The actual compute — including weight widening — is
  // gemm, which re-validates cheaply.
  if (!x.defined()) {
    return core::InvalidArgumentError(
        "ReferenceLinear::forward: x is undefined");
  }
  if (x.shape().rank() != 2 || x.shape().dim(1) != in_features_) {
    return core::InvalidArgumentError(
        "ReferenceLinear::forward: x must be [T, {}], got {}", in_features_,
        x.shape());
  }
  if (!y.defined()) {
    return core::InvalidArgumentError(
        "ReferenceLinear::forward: y is undefined");
  }
  if (y.shape().rank() != 2 || y.shape().dim(0) != x.shape().dim(0) ||
      y.shape().dim(1) != out_features_) {
    return core::InvalidArgumentError(
        "ReferenceLinear::forward: y must be caller-allocated [{}, {}], got {}",
        x.shape().dim(0), out_features_, y.shape());
  }
  const tensor::Tensor* bias = bias_.has_value() ? &*bias_ : nullptr;
  return cpu::gemm(x, weight_, bias, y);
}

}  // namespace engine::model
