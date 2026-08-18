#include "model/modules.h"

#include "core/status.h"
#include "cpu/ops.h"
#include "model/config.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace engine::model {

namespace {

// The float storage dtypes a weight/bias may use (model-execution.md §3.3):
// checkpoints are f32/f16/bf16; the reference widens f16/bf16 per element.
[[nodiscard]] bool IsSupportedFloat(tensor::DataType dtype) {
  return dtype == tensor::DataType::kFloat32 ||
         dtype == tensor::DataType::kFloat16 ||
         dtype == tensor::DataType::kBFloat16;
}

// Base inverse frequencies, HF form: inv_freq[i] = 1 / theta^(2i/d) for
// i ∈ [0, d/2) (model-execution.md §7). Computed in double and stored fp32 —
// the correctly-rounded fp32 value, within ~1 ulp of HF's fp32 pow.
[[nodiscard]] std::vector<float> BaseInvFreq(int head_dim, float theta) {
  const int half = head_dim / 2;
  std::vector<float> inv(static_cast<std::size_t>(half));
  const auto d = static_cast<double>(head_dim);
  const auto base = static_cast<double>(theta);
  for (int i = 0; i < half; ++i) {
    const double exponent = static_cast<double>(2 * i) / d;
    inv[static_cast<std::size_t>(i)] =
        static_cast<float>(1.0 / std::pow(base, exponent));
  }
  return inv;
}

// Applies `rope_scaling` to the base inverse frequencies (model-execution.md
// §7). nullopt / "default" → identity; "linear" → inv_freq /= factor; "llama3"
// → HF's `_compute_llama3_parameters` piecewise wavelength scaling. Any other
// rope_type → Unimplemented. Intermediate math is double for accuracy; the
// result is fp32. Malformed llama3 parameters → InvalidArgument.
[[nodiscard]] core::StatusOr<std::vector<float>> ApplyRopeScaling(
    std::vector<float> inv_freq, const std::optional<RopeScaling>& scaling) {
  if (!scaling.has_value() || scaling->rope_type.empty() ||
      scaling->rope_type == "default") {
    return inv_freq;
  }
  const RopeScaling& s = *scaling;
  if (s.rope_type == "linear") {
    if (!(s.factor > 0.0F)) {
      return core::InvalidArgumentError(
          "Rope: linear rope_scaling requires factor > 0, got {}", s.factor);
    }
    const auto factor = static_cast<double>(s.factor);
    for (float& v : inv_freq) {
      v = static_cast<float>(static_cast<double>(v) / factor);
    }
    return inv_freq;
  }
  if (s.rope_type == "llama3") {
    if (!(s.factor > 0.0F)) {
      return core::InvalidArgumentError(
          "Rope: llama3 rope_scaling requires factor > 0, got {}", s.factor);
    }
    if (s.high_freq_factor == s.low_freq_factor) {
      return core::InvalidArgumentError(
          "Rope: llama3 rope_scaling requires high_freq_factor != "
          "low_freq_factor, both are {}",
          s.low_freq_factor);
    }
    if (s.original_max_position_embeddings <= 0) {
      return core::InvalidArgumentError(
          "Rope: llama3 rope_scaling requires original_max_position_embeddings "
          "> 0, got {}",
          s.original_max_position_embeddings);
    }
    const auto factor = static_cast<double>(s.factor);
    const auto low_freq_factor = static_cast<double>(s.low_freq_factor);
    const auto high_freq_factor = static_cast<double>(s.high_freq_factor);
    const auto old_context_len =
        static_cast<double>(s.original_max_position_embeddings);
    const double low_freq_wavelen = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;
    constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;
    for (float& v : inv_freq) {
      const auto f = static_cast<double>(v);
      const double wavelen = kTwoPi / f;
      // First branch: below high frequency untouched, above low frequency
      // divided by factor (matches HF's `torch.where(wavelen > ...)`).
      double inv_llama = wavelen > low_freq_wavelen ? f / factor : f;
      // Smooth ramp between the two wavelengths.
      const double smooth = ((old_context_len / wavelen) - low_freq_factor) /
                            (high_freq_factor - low_freq_factor);
      const double smoothed =
          ((1.0 - smooth) * inv_llama / factor) + (smooth * inv_llama);
      const bool is_medium =
          !(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen);
      if (is_medium) {
        inv_llama = smoothed;
      }
      v = static_cast<float>(inv_llama);
    }
    return inv_freq;
  }
  return core::UnimplementedError(
      "Rope: unsupported rope_type '{}' — supported: default, linear, llama3",
      s.rope_type);
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

core::StatusOr<RmsNorm> RmsNorm::Create(tensor::Tensor weight, float eps) {
  if (!weight.defined()) {
    return core::InvalidArgumentError("RmsNorm: weight is undefined");
  }
  if (weight.shape().rank() != 1) {
    return core::InvalidArgumentError(
        "RmsNorm: weight must be rank-1 [E], got rank-{}",
        weight.shape().rank());
  }
  if (!weight.is_contiguous()) {
    return core::InvalidArgumentError("RmsNorm: weight must be contiguous");
  }
  if (!IsSupportedFloat(weight.dtype())) {
    return core::InvalidArgumentError(
        "RmsNorm: weight dtype must be f32/f16/bf16, got {}",
        tensor::to_string(weight.dtype()));
  }
  const std::int64_t hidden_size = weight.shape().dim(0);
  return RmsNorm(std::move(weight), eps, hidden_size);
}

core::Status RmsNorm::forward(const tensor::Tensor& x,
                              tensor::Tensor& y) const {
  // Validate the activation tensors here so the messages name `x`/`y`; the
  // per-element widening and reduction happen in cpu::rmsnorm, which
  // re-validates cheaply. The M5 graph passes fp32 activations.
  if (!x.defined()) {
    return core::InvalidArgumentError("RmsNorm::forward: x is undefined");
  }
  if (x.shape().rank() != 2 || x.shape().dim(1) != hidden_size_) {
    return core::InvalidArgumentError(
        "RmsNorm::forward: x must be [T, {}], got {}", hidden_size_, x.shape());
  }
  if (x.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError("RmsNorm::forward: x must be f32, got {}",
                                      tensor::to_string(x.dtype()));
  }
  if (!y.defined()) {
    return core::InvalidArgumentError("RmsNorm::forward: y is undefined");
  }
  if (y.shape().rank() != 2 || y.shape().dim(0) != x.shape().dim(0) ||
      y.shape().dim(1) != hidden_size_) {
    return core::InvalidArgumentError(
        "RmsNorm::forward: y must be caller-allocated [{}, {}], got {}",
        x.shape().dim(0), hidden_size_, y.shape());
  }
  return cpu::rmsnorm(x, weight_, eps_, y);
}

core::StatusOr<Embedding> Embedding::Create(tensor::Tensor weight) {
  if (!weight.defined()) {
    return core::InvalidArgumentError("Embedding: weight is undefined");
  }
  if (weight.shape().rank() != 2) {
    return core::InvalidArgumentError(
        "Embedding: weight must be rank-2 [V, E], got rank-{}",
        weight.shape().rank());
  }
  if (!weight.is_contiguous()) {
    return core::InvalidArgumentError("Embedding: weight must be contiguous");
  }
  if (!IsSupportedFloat(weight.dtype())) {
    return core::InvalidArgumentError(
        "Embedding: weight dtype must be f32/f16/bf16, got {}",
        tensor::to_string(weight.dtype()));
  }
  const std::int64_t vocab_size = weight.shape().dim(0);
  const std::int64_t hidden_size = weight.shape().dim(1);
  return Embedding(std::move(weight), vocab_size, hidden_size);
}

core::Status Embedding::forward(std::span<const std::int32_t> ids,
                                tensor::Tensor& y) const {
  // Validate `y` here so the message names `y` (the module's term); the gather
  // and widening happen in cpu::embedding_lookup, which re-validates cheaply
  // and reports an out-of-range id.
  if (!y.defined()) {
    return core::InvalidArgumentError("Embedding::forward: y is undefined");
  }
  const auto rows = static_cast<std::int64_t>(ids.size());
  if (y.shape().rank() != 2 || y.shape().dim(0) != rows ||
      y.shape().dim(1) != hidden_size_) {
    return core::InvalidArgumentError(
        "Embedding::forward: y must be caller-allocated [{}, {}], got {}", rows,
        hidden_size_, y.shape());
  }
  return cpu::embedding_lookup(weight_, ids, y);
}

core::StatusOr<Rope> Rope::Create(int head_dim, float theta,
                                  const std::optional<RopeScaling>& scaling,
                                  std::int64_t num_positions) {
  if (head_dim < 2 || head_dim % 2 != 0) {
    return core::InvalidArgumentError(
        "Rope: head_dim must be even and >= 2, got {}", head_dim);
  }
  if (!(theta > 0.0F) || !std::isfinite(theta)) {
    return core::InvalidArgumentError(
        "Rope: theta must be finite and > 0, "
        "got {}",
        theta);
  }
  if (num_positions < 1) {
    return core::InvalidArgumentError(
        "Rope: num_positions must be >= 1, got {}", num_positions);
  }

  core::StatusOr<std::vector<float>> inv_freq =
      ApplyRopeScaling(BaseInvFreq(head_dim, theta), scaling);
  RETURN_IF_ERROR(inv_freq.status());

  const std::int64_t half = head_dim / 2;
  core::StatusOr<tensor::Tensor> cos = tensor::ops::zeros(
      tensor::Shape{num_positions, half}, tensor::DataType::kFloat32);
  RETURN_IF_ERROR(cos.status());
  core::StatusOr<tensor::Tensor> sin = tensor::ops::zeros(
      tensor::Shape{num_positions, half}, tensor::DataType::kFloat32);
  RETURN_IF_ERROR(sin.status());

  // Tables: cos[p, j] = cos(p · inv_freq[j]), sin likewise (model-execution.md
  // §7). The angle is formed in double for accuracy; only the first half (d/2)
  // is stored — the half-rotation reuses cos[p,j] for both elements of a pair.
  auto* cos_data = cos->data_ptr<float>();
  auto* sin_data = sin->data_ptr<float>();
  for (std::int64_t p = 0; p < num_positions; ++p) {
    float* cos_row = cos_data + (p * half);
    float* sin_row = sin_data + (p * half);
    for (std::int64_t j = 0; j < half; ++j) {
      const double angle =
          static_cast<double>(p) *
          static_cast<double>((*inv_freq)[static_cast<std::size_t>(j)]);
      cos_row[j] = static_cast<float>(std::cos(angle));
      sin_row[j] = static_cast<float>(std::sin(angle));
    }
  }

  return Rope(*std::move(cos), *std::move(sin), *std::move(inv_freq), head_dim,
              num_positions);
}

core::Status Rope::apply(tensor::Tensor& q, tensor::Tensor& k,
                         std::span<const std::int32_t> positions) const {
  // Guard q/k head dims here so a mismatch names q/k rather than leaking
  // cpu::rope_apply's cos/sin shape message; the rotation and position-bounds
  // check happen in cpu::rope_apply.
  if (!q.defined() || !k.defined()) {
    return core::InvalidArgumentError("Rope::apply: q/k must be defined");
  }
  if (q.shape().rank() != 3 || q.shape().dim(2) != head_dim_) {
    return core::InvalidArgumentError(
        "Rope::apply: q must be [T, H, {}], got {}", head_dim_, q.shape());
  }
  if (k.shape().rank() != 3 || k.shape().dim(2) != head_dim_) {
    return core::InvalidArgumentError(
        "Rope::apply: k must be [T, Hkv, {}], got {}", head_dim_, k.shape());
  }
  RETURN_IF_ERROR(cpu::rope_apply(q, positions, cos_, sin_));
  return cpu::rope_apply(k, positions, cos_, sin_);
}

}  // namespace engine::model
