#pragma once

#include "core/status.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <optional>

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

}  // namespace engine::model
