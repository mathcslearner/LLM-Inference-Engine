#include "model/modules.h"

#include "core/status.h"
#include "cpu/ops.h"
#include "kvcache/kv_cache.h"
#include "model/config.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <memory>
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

core::StatusOr<Attention> Attention::Create(std::unique_ptr<Linear> q_proj,
                                            std::unique_ptr<Linear> k_proj,
                                            std::unique_ptr<Linear> v_proj,
                                            std::unique_ptr<Linear> o_proj,
                                            Rope rope, int num_heads,
                                            int num_kv_heads, int head_dim) {
  if (q_proj == nullptr || k_proj == nullptr || v_proj == nullptr ||
      o_proj == nullptr) {
    return core::InvalidArgumentError(
        "Attention: q/k/v/o projections must all be non-null");
  }
  if (num_heads < 1 || num_kv_heads < 1 || head_dim < 1) {
    return core::InvalidArgumentError(
        "Attention: num_heads/num_kv_heads/head_dim must be >= 1, got {}/{}/{}",
        num_heads, num_kv_heads, head_dim);
  }
  if (num_heads % num_kv_heads != 0) {
    return core::InvalidArgumentError(
        "Attention: num_heads ({}) must be a positive multiple of num_kv_heads "
        "({})",
        num_heads, num_kv_heads);
  }
  if (rope.head_dim() != head_dim) {
    return core::InvalidArgumentError(
        "Attention: rope head_dim ({}) must equal head_dim ({})",
        rope.head_dim(), head_dim);
  }

  const std::int64_t q_rows = static_cast<std::int64_t>(num_heads) * head_dim;
  const std::int64_t kv_rows =
      static_cast<std::int64_t>(num_kv_heads) * head_dim;
  // Hidden size E is the shared projection input width; it is NOT assumed equal
  // to H·d (a decoupled head_dim is honored — model-execution.md §3.1).
  const std::int64_t hidden = q_proj->in_features();

  if (q_proj->out_features() != q_rows) {
    return core::InvalidArgumentError(
        "Attention: q_proj out_features {} must equal num_heads·head_dim {}",
        q_proj->out_features(), q_rows);
  }
  if (k_proj->out_features() != kv_rows) {
    return core::InvalidArgumentError(
        "Attention: k_proj out_features {} must equal num_kv_heads·head_dim {}",
        k_proj->out_features(), kv_rows);
  }
  if (v_proj->out_features() != kv_rows) {
    return core::InvalidArgumentError(
        "Attention: v_proj out_features {} must equal num_kv_heads·head_dim {}",
        v_proj->out_features(), kv_rows);
  }
  if (k_proj->in_features() != hidden || v_proj->in_features() != hidden) {
    return core::InvalidArgumentError(
        "Attention: q/k/v_proj must share in_features (hidden size); got q={} "
        "k={} v={}",
        hidden, k_proj->in_features(), v_proj->in_features());
  }
  if (o_proj->in_features() != q_rows) {
    return core::InvalidArgumentError(
        "Attention: o_proj in_features {} must equal num_heads·head_dim {}",
        o_proj->in_features(), q_rows);
  }
  if (o_proj->out_features() != hidden) {
    return core::InvalidArgumentError(
        "Attention: o_proj out_features {} must equal the hidden size {}",
        o_proj->out_features(), hidden);
  }

  return Attention(std::move(q_proj), std::move(k_proj), std::move(v_proj),
                   std::move(o_proj), std::move(rope), num_heads, num_kv_heads,
                   head_dim, hidden);
}

core::Status Attention::forward(const tensor::Tensor& x,
                                std::span<const std::int32_t> positions,
                                int layer, kvcache::KvCache& cache,
                                tensor::Tensor& y) const {
  // Front-load all input validation (ADR-003) so a failure never leaves a
  // half-appended layer: nothing touches the cache until every check passes.
  if (!x.defined()) {
    return core::InvalidArgumentError("Attention::forward: x is undefined");
  }
  if (x.shape().rank() != 2 || x.shape().dim(1) != hidden_size_) {
    return core::InvalidArgumentError(
        "Attention::forward: x must be [T, {}], got {}", hidden_size_,
        x.shape());
  }
  if (x.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "Attention::forward: x must be f32, got {}",
        tensor::to_string(x.dtype()));
  }
  const std::int64_t t_dim = x.shape().dim(0);
  const auto num_positions = static_cast<std::int64_t>(positions.size());
  if (num_positions != t_dim) {
    return core::InvalidArgumentError(
        "Attention::forward: positions length {} must equal T {}",
        num_positions, t_dim);
  }
  if (!y.defined()) {
    return core::InvalidArgumentError("Attention::forward: y is undefined");
  }
  if (y.shape().rank() != 2 || y.shape().dim(0) != t_dim ||
      y.shape().dim(1) != hidden_size_) {
    return core::InvalidArgumentError(
        "Attention::forward: y must be caller-allocated [{}, {}], got {}",
        t_dim, hidden_size_, y.shape());
  }

  const std::int64_t q_rows = static_cast<std::int64_t>(num_heads_) * head_dim_;
  const std::int64_t kv_rows =
      static_cast<std::int64_t>(num_kv_heads_) * head_dim_;

  // Project QKV into freshly allocated fp32 work tensors (the reference
  // allocates freely; M6/M12 provide workspaces). q [T, H·d], k/v [T, Hkv·d].
  ASSIGN_OR_RETURN(tensor::Tensor q_flat,
                   tensor::ops::zeros(tensor::Shape{t_dim, q_rows},
                                      tensor::DataType::kFloat32));
  ASSIGN_OR_RETURN(tensor::Tensor k_flat,
                   tensor::ops::zeros(tensor::Shape{t_dim, kv_rows},
                                      tensor::DataType::kFloat32));
  ASSIGN_OR_RETURN(tensor::Tensor v_flat,
                   tensor::ops::zeros(tensor::Shape{t_dim, kv_rows},
                                      tensor::DataType::kFloat32));
  RETURN_IF_ERROR(q_proj_->forward(x, q_flat));
  RETURN_IF_ERROR(k_proj_->forward(x, k_flat));
  RETURN_IF_ERROR(v_proj_->forward(x, v_flat));

  // Reshape to head-split [T, heads, d] views (contiguous → zero-copy). RoPE
  // rotates Q and K in place (V is not rotated).
  ASSIGN_OR_RETURN(tensor::Tensor q3,
                   q_flat.reshape(tensor::Shape{t_dim, num_heads_, head_dim_}));
  ASSIGN_OR_RETURN(tensor::Tensor k3, k_flat.reshape(tensor::Shape{
                                          t_dim, num_kv_heads_, head_dim_}));
  ASSIGN_OR_RETURN(
      const tensor::Tensor v3,
      v_flat.reshape(tensor::Shape{t_dim, num_kv_heads_, head_dim_}));
  RETURN_IF_ERROR(rope_.apply(q3, k3, positions));

  // Append this call's K/V (token-major [T, Hkv, d]) and read the accumulated
  // [Hkv, P+T, d] head-major view back — RoPE and the cache write stay separate
  // observable steps for the reference (§4.2).
  RETURN_IF_ERROR(cache.append(layer, k3, v3));
  ASSIGN_OR_RETURN(const kvcache::KvView kv, cache.view(layer));

  // Causal-masked GQA attention → context [T, H, d], then o_proj. Scale is
  // 1/sqrt(d) formed in double and cast to fp32 (HF's head_dim**-0.5).
  ASSIGN_OR_RETURN(
      tensor::Tensor ctx3,
      tensor::ops::zeros(tensor::Shape{t_dim, num_heads_, head_dim_},
                         tensor::DataType::kFloat32));
  const auto scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
  RETURN_IF_ERROR(cpu::attention(q3, kv.k, kv.v, scale, ctx3));

  ASSIGN_OR_RETURN(const tensor::Tensor ctx2,
                   ctx3.reshape(tensor::Shape{t_dim, q_rows}));
  return o_proj_->forward(ctx2, y);
}

}  // namespace engine::model
