#pragma once

#include "core/status.h"
#include "model/config.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// Model graph modules (M5; design: docs/design/model-execution.md §4). Each
// module owns its weight handles (borrowed, zero-copy, into the mapped
// checkpoint — model-loading.md §3.5, §4.3) and exposes a single forward-style
// entry. The reference implementations call `cpu::` ops.
//
// This file grows across M5: M5-T02 adds `Linear` + `ReferenceLinear`; T03–T07
// add `RmsNorm`, `Rope`, `Attention`, `Mlp`, `DecoderLayer`.

namespace engine::model {

// `Linear` is the one module that is an *interface*, not a concrete struct,
// because M13 slots `QuantizedLinear` in behind it without touching any layer
// code (model-execution.md §4.1). The weight's storage dtype and layout are the
// implementation's private business — nothing outside a `Linear` implementation
// may read the weight as an fp32 matrix, so no call site assumes a layout.
class Linear {
 public:
  virtual ~Linear() = default;

  // y[T, out] = x[T, in] * W^T  (+ bias[out] if present). Row-major, fp32
  // activations in and out. `y` is caller-allocated and must already be a
  // contiguous fp32 [T, out] tensor (the allocation-free-decode property M12
  // relies on); a wrong-shape/dtype `y`, or an `x` that is not contiguous fp32
  // [T, in], is InvalidArgument. The same call serves prefill (T large, GEMM)
  // and decode (T == 1, GEMV) with no branch at this level (§4.1).
  [[nodiscard]] virtual core::Status forward(const tensor::Tensor& x,
                                             tensor::Tensor& y) const = 0;

  [[nodiscard]] virtual std::int64_t in_features() const = 0;
  [[nodiscard]] virtual std::int64_t out_features() const = 0;
  [[nodiscard]] virtual bool has_bias() const = 0;

 protected:
  // Special members are protected (not deleted): a `Linear&` cannot be sliced,
  // but concrete implementations stay copyable/movable so they can be returned
  // by value (e.g. StatusOr<ReferenceLinear>) and stored in containers.
  Linear() = default;
  Linear(const Linear&) = default;
  Linear& operator=(const Linear&) = default;
  Linear(Linear&&) = default;
  Linear& operator=(Linear&&) = default;
};

// The reference `Linear`: holds the canonical checkpoint weight `[out, in]`
// (model-loading.md §4) and optional bias `[out]` as zero-copy handles (shared
// Buffer — the mapped checkpoint stays alive as long as this module lives), and
// computes `cpu::gemm` with on-the-fly weight/bias widening. No repacking, no
// copy.
class ReferenceLinear final : public Linear {
 public:
  // Validates the weight `[out, in]` (rank-2, contiguous, f32/f16/bf16) and,
  // if given, the bias `[out]` (rank-1, contiguous, f32/f16/bf16, length ==
  // out). Bias is per-projection and per-config: Llama has none; Qwen2 biases
  // q/k/v (model-execution.md §4.1) — the builder decides, so Qwen2 support is
  // wiring, not new layer code. Malformed weights → InvalidArgument naming the
  // problem.
  [[nodiscard]] static core::StatusOr<ReferenceLinear> Create(
      tensor::Tensor weight, std::optional<tensor::Tensor> bias = std::nullopt);

  [[nodiscard]] core::Status forward(const tensor::Tensor& x,
                                     tensor::Tensor& y) const override;

  [[nodiscard]] std::int64_t in_features() const override {
    return in_features_;
  }
  [[nodiscard]] std::int64_t out_features() const override {
    return out_features_;
  }
  [[nodiscard]] bool has_bias() const override { return bias_.has_value(); }

 private:
  ReferenceLinear(tensor::Tensor weight, std::optional<tensor::Tensor> bias,
                  std::int64_t in_features, std::int64_t out_features)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        in_features_(in_features),
        out_features_(out_features) {}

  tensor::Tensor weight_;               // [out, in], borrowed checkpoint handle
  std::optional<tensor::Tensor> bias_;  // [out], borrowed; nullopt when absent
  std::int64_t in_features_ = 0;
  std::int64_t out_features_ = 0;
};

// RMSNorm (M5-T03; design: docs/design/model-execution.md §4.2). Holds the
// `[E]` scale weight as a zero-copy checkpoint handle and the epsilon from
// `config.rms_norm_eps`; `forward` computes `y = x * rsqrt(mean(x²) + eps) *
// weight` per token in fp32 via `cpu::rmsnorm`. A plain concrete class (no
// interface needed in M5 — unlike `Linear`, which M13 specializes).
class RmsNorm {
 public:
  // Validates the weight `[E]` (rank-1, contiguous, f32/f16/bf16). `eps` is the
  // model's `rms_norm_eps`. Malformed weight → InvalidArgument naming the
  // problem.
  [[nodiscard]] static core::StatusOr<RmsNorm> Create(tensor::Tensor weight,
                                                      float eps);

  // y[T, E] = rmsnorm(x[T, E]) * weight. `x` and `y` are caller-allocated
  // contiguous fp32 [T, E] (activations are fp32 in the M5 graph); a
  // wrong-shape/dtype `x`/`y`, or a hidden dim disagreeing with the weight, is
  // InvalidArgument.
  [[nodiscard]] core::Status forward(const tensor::Tensor& x,
                                     tensor::Tensor& y) const;

  [[nodiscard]] std::int64_t hidden_size() const { return hidden_size_; }
  [[nodiscard]] float eps() const { return eps_; }

 private:
  RmsNorm(tensor::Tensor weight, float eps, std::int64_t hidden_size)
      : weight_(std::move(weight)), eps_(eps), hidden_size_(hidden_size) {}

  tensor::Tensor weight_;  // [E], borrowed checkpoint handle
  float eps_ = 0.0F;
  std::int64_t hidden_size_ = 0;
};

// Token-embedding table (M5-T04; design: docs/design/model-execution.md §4,
// §7). Holds the `[V, E]` embedding weight as a zero-copy checkpoint handle
// (bf16/f16 storage preserved — M4 never up-converts at load) and gathers rows
// widened to fp32. A plain concrete class like `RmsNorm`; the natural seam for
// M6-T06's tied-weight sharing (lookup vs projection may hold *different*
// physical layouts of the same logical weight), so the lookup goes through a
// module rather than a bare op in `ReferenceModel`.
class Embedding {
 public:
  // Validates the weight `[V, E]` (rank-2, contiguous, f32/f16/bf16). Malformed
  // weight → InvalidArgument naming the problem.
  [[nodiscard]] static core::StatusOr<Embedding> Create(tensor::Tensor weight);

  // y[T, E] = table[ids[t], :] widened to fp32. `ids` are absolute token ids in
  // [0, V) (the `ForwardRequest.token_ids` span, passed straight through); `y`
  // is caller-allocated contiguous fp32 [T, E]. An out-of-range id, or a
  // wrong-shape/dtype `y`, is InvalidArgument. Calls `cpu::embedding_lookup`.
  [[nodiscard]] core::Status forward(std::span<const std::int32_t> ids,
                                     tensor::Tensor& y) const;

  [[nodiscard]] std::int64_t vocab_size() const { return vocab_size_; }
  [[nodiscard]] std::int64_t hidden_size() const { return hidden_size_; }

 private:
  Embedding(tensor::Tensor weight, std::int64_t vocab_size,
            std::int64_t hidden_size)
      : weight_(std::move(weight)),
        vocab_size_(vocab_size),
        hidden_size_(hidden_size) {}

  tensor::Tensor weight_;  // [V, E], borrowed checkpoint handle
  std::int64_t vocab_size_ = 0;
  std::int64_t hidden_size_ = 0;
};

// Rotary position embeddings (M5-T04; design: docs/design/model-execution.md
// §7). Precomputes the `cos`/`sin` tables `[num_positions, head_dim/2]` at
// construction from `config.rope_theta` and any `rope_scaling`, then `apply`
// rotates Q and K in place using each token's absolute position — the
// HF-Llama half-rotation layout. A plain concrete class like `RmsNorm`.
//
// The frequencies obey `inv_freq[i] = 1 / theta^(2i/d)` (fp32), with
// `rope_scaling` interpreted per §7: absent/`"default"` → identity, `"linear"`
// → `inv_freq /= factor`, `"llama3"` → HF's `_compute_llama3_parameters`
// piecewise wavelength scaling. Any other `rope_type` is `Unimplemented` (the
// executor rejects what it can't honor).
class Rope {
 public:
  // `head_dim` must be even and >= 2; `theta` finite and > 0; `num_positions`
  // >= 1 (the builder passes `config.max_position_embeddings`; tests pass
  // explicit small counts). `scaling` is `config.rope_scaling` (nullopt →
  // default). A malformed argument, or an unsupported/ill-formed `rope_type`,
  // → InvalidArgument / Unimplemented naming the problem.
  [[nodiscard]] static core::StatusOr<Rope> Create(
      int head_dim, float theta, const std::optional<RopeScaling>& scaling,
      std::int64_t num_positions);

  // Rotates `q [T, H, d]` and `k [T, Hkv, d]` in place by their tokens'
  // absolute `positions` (each in [0, num_positions)). q/k must be contiguous
  // fp32 rank-3 with `dim(2) == head_dim`; `positions.size() == T`. Malformed
  // q/k or an out-of-range position → InvalidArgument. Calls `cpu::rope_apply`
  // once per tensor (RoPE and the KV-cache write stay separate observable steps
  // for the reference — §4.2).
  [[nodiscard]] core::Status apply(
      tensor::Tensor& q, tensor::Tensor& k,
      std::span<const std::int32_t> positions) const;

  [[nodiscard]] const tensor::Tensor& cos() const { return cos_; }
  [[nodiscard]] const tensor::Tensor& sin() const { return sin_; }
  // inv_freq `[head_dim/2]`, fp32 — exposed so the llama3/linear scaling
  // goldens can validate the frequency computation directly (§12).
  [[nodiscard]] std::span<const float> inv_freq() const { return inv_freq_; }
  [[nodiscard]] int head_dim() const { return head_dim_; }
  [[nodiscard]] std::int64_t num_positions() const { return num_positions_; }

 private:
  Rope(tensor::Tensor cos, tensor::Tensor sin, std::vector<float> inv_freq,
       int head_dim, std::int64_t num_positions)
      : cos_(std::move(cos)),
        sin_(std::move(sin)),
        inv_freq_(std::move(inv_freq)),
        head_dim_(head_dim),
        num_positions_(num_positions) {}

  tensor::Tensor cos_;           // [num_positions, head_dim/2], fp32
  tensor::Tensor sin_;           // [num_positions, head_dim/2], fp32
  std::vector<float> inv_freq_;  // [head_dim/2], fp32
  int head_dim_ = 0;
  std::int64_t num_positions_ = 0;
};

}  // namespace engine::model
