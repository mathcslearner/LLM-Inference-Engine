#include "model/packed_linear.h"

#include "core/status.h"
#include "kernels/gemm.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace engine::model {

namespace {

[[nodiscard]] bool IsSupportedFloat(tensor::DataType dtype) {
  return dtype == tensor::DataType::kFloat32 ||
         dtype == tensor::DataType::kFloat16 ||
         dtype == tensor::DataType::kBFloat16;
}

// float16/bfloat16 are standard-layout wrappers around one std::uint16_t
// (static_asserted in tensor/half.h), so their object representations coincide
// — the same reinterpretation the convert kernels use (kernels/convert.cpp).
template <typename Half>
[[nodiscard]] const std::uint16_t* AsBits(const Half* p) {
  return reinterpret_cast<const std::uint16_t*>(p);
}
template <typename Half>
[[nodiscard]] std::uint16_t* AsBits(Half* p) {
  return reinterpret_cast<std::uint16_t*>(p);
}

// Repack the contiguous checkpoint weight `[out, in]` into the panel layout
// `[P, K, kNr]` (§3.2), dispatching on storage dtype. Both tensors are
// contiguous and validated by the caller.
void PackInto(const tensor::Tensor& weight, std::int64_t out_features,
              std::int64_t in_features, tensor::Tensor& packed) {
  switch (weight.dtype()) {
    case tensor::DataType::kBFloat16:
      kernels::PackWeightPanels(AsBits(weight.data_ptr<tensor::bfloat16>()),
                                out_features, in_features,
                                AsBits(packed.data_ptr<tensor::bfloat16>()));
      return;
    case tensor::DataType::kFloat16:
      kernels::PackWeightPanels(AsBits(weight.data_ptr<tensor::float16>()),
                                out_features, in_features,
                                AsBits(packed.data_ptr<tensor::float16>()));
      return;
    case tensor::DataType::kFloat32:
      kernels::PackWeightPanels(weight.data_ptr<float>(), out_features,
                                in_features, packed.data_ptr<float>());
      return;
    default:
      // Unreachable: Create validated the dtype (IsSupportedFloat) first.
      CHECK(false, "PackedLinear: unsupported weight dtype reached PackInto");
  }
}

}  // namespace

core::StatusOr<PackedLinear> PackedLinear::Create(
    const tensor::Tensor& weight, const std::optional<tensor::Tensor>& bias) {
  if (!weight.defined()) {
    return core::InvalidArgumentError("PackedLinear: weight is undefined");
  }
  if (weight.shape().rank() != 2) {
    return core::InvalidArgumentError(
        "PackedLinear: weight must be rank-2 [out, in], got rank-{}",
        weight.shape().rank());
  }
  if (!weight.is_contiguous()) {
    return core::InvalidArgumentError(
        "PackedLinear: weight must be contiguous");
  }
  if (!IsSupportedFloat(weight.dtype())) {
    return core::InvalidArgumentError(
        "PackedLinear: weight dtype must be f32/f16/bf16, got {}",
        tensor::to_string(weight.dtype()));
  }
  const std::int64_t out_features = weight.shape().dim(0);
  const std::int64_t in_features = weight.shape().dim(1);

  std::optional<tensor::Tensor> bias_f32;
  if (bias.has_value()) {
    if (!bias->defined()) {
      return core::InvalidArgumentError("PackedLinear: bias is undefined");
    }
    if (bias->shape().rank() != 1) {
      return core::InvalidArgumentError(
          "PackedLinear: bias must be rank-1 [out], got rank-{}",
          bias->shape().rank());
    }
    if (!bias->is_contiguous()) {
      return core::InvalidArgumentError(
          "PackedLinear: bias must be contiguous");
    }
    if (!IsSupportedFloat(bias->dtype())) {
      return core::InvalidArgumentError(
          "PackedLinear: bias dtype must be f32/f16/bf16, got {}",
          tensor::to_string(bias->dtype()));
    }
    if (bias->shape().dim(0) != out_features) {
      return core::InvalidArgumentError(
          "PackedLinear: bias length {} must equal out_features {}",
          bias->shape().dim(0), out_features);
    }
    // Convert the bias to fp32 once (§3.5): it is read every token, so a
    // per-element widen in the hot path is wasteful. A f32 bias is cast to a
    // fresh f32 tensor too (a copy) so `bias_` never aliases the checkpoint,
    // matching the "checkpoint handle dropped" contract for the whole module.
    core::StatusOr<tensor::Tensor> converted =
        tensor::ops::cast(*bias, tensor::DataType::kFloat32);
    if (!converted.ok()) {
      return converted.status();
    }
    bias_f32 = *std::move(converted);
  }

  // Allocate the packed buffer [P, K, kNr] in the weight's storage dtype and
  // fill it. `Tensor::empty` propagates allocation failure.
  const std::int64_t panels = kernels::PackedPanels(out_features);
  core::StatusOr<tensor::Tensor> packed =
      tensor::Tensor::empty(tensor::Shape{panels, in_features, kernels::kNr},
                            weight.dtype(), weight.device());
  if (!packed.ok()) {
    return packed.status();
  }
  PackInto(weight, out_features, in_features, *packed);

  return PackedLinear(*std::move(packed), std::move(bias_f32), weight.dtype(),
                      in_features, out_features);
  // `weight` (the checkpoint handle) goes out of scope here — not retained.
}

core::Status PackedLinear::forward(const tensor::Tensor& x,
                                   tensor::Tensor& y) const {
  // Front-load every recoverable check so the parallel region does only
  // arithmetic (§5): naming `x`/`y` (the Linear contract's terms).
  if (!x.defined()) {
    return core::InvalidArgumentError("PackedLinear::forward: x is undefined");
  }
  if (x.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "PackedLinear::forward: x must be fp32, got {}",
        tensor::to_string(x.dtype()));
  }
  if (x.shape().rank() != 2 || x.shape().dim(1) != in_features_) {
    return core::InvalidArgumentError(
        "PackedLinear::forward: x must be [T, {}], got {}", in_features_,
        x.shape());
  }
  if (!x.is_contiguous()) {
    return core::InvalidArgumentError(
        "PackedLinear::forward: x must be contiguous");
  }
  if (!y.defined()) {
    return core::InvalidArgumentError("PackedLinear::forward: y is undefined");
  }
  if (y.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "PackedLinear::forward: y must be fp32, got {}",
        tensor::to_string(y.dtype()));
  }
  if (y.shape().rank() != 2 || y.shape().dim(0) != x.shape().dim(0) ||
      y.shape().dim(1) != out_features_) {
    return core::InvalidArgumentError(
        "PackedLinear::forward: y must be caller-allocated [{}, {}], got {}",
        x.shape().dim(0), out_features_, y.shape());
  }
  if (!y.is_contiguous()) {
    return core::InvalidArgumentError(
        "PackedLinear::forward: y must be contiguous");
  }

  const std::int64_t t = x.shape().dim(0);
  const float* bias_ptr =
      bias_.has_value() ? bias_->data_ptr<float>() : nullptr;
  kernels::PackedGemm(x.data_ptr<float>(), t, in_features_, packed_.data(),
                      weight_dtype_, out_features_, bias_ptr,
                      y.data_ptr<float>());
  return core::OkStatus();
}

}  // namespace engine::model
