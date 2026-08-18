#pragma once

#include "core/status.h"
#include "model/modules.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <optional>
#include <utility>

// PackedLinear (M6-T02; design: docs/design/optimized-cpu-execution.md §3, §4).
// The optimized `Linear`: at construction it repacks the canonical checkpoint
// weight `[out, in]` into the K-major panel layout (§3.2) and drops the
// checkpoint handle — the packed copy is authoritative for the optimized path
// (§3.1). `forward` runs the dispatched packed GEMM/GEMV (kernels::PackedGemm).
// It slots into `Attention`/`Mlp`'s `unique_ptr<Linear>` unchanged (§2.2), and
// M13's `QuantizedLinear` will be a third implementation of the same interface.
//
// This header stays free of any `kernels` type so the `model → kernels` edge
// is a PRIVATE link (§2.1): the packed bytes are exposed only as a plain
// `tensor::Tensor` (its dtype and shape `[P, K, panel_width]` are an
// implementation detail M6-T06's tied-embedding gather reads through
// `packed_weight()`), never as a kernels layout type.

namespace engine::model {

class PackedLinear final : public Linear {
 public:
  // Validates the weight `[out, in]` (rank-2, contiguous, f32/f16/bf16) and,
  // if given, the bias `[out]` (rank-1, contiguous, f32/f16/bf16, length ==
  // out) exactly as `ReferenceLinear` does — same messages — then repacks the
  // weight and converts the bias to fp32 once (§3.5). The checkpoint weight
  // handle is not retained. Malformed inputs → InvalidArgument naming the
  // problem; an allocation failure while packing propagates.
  [[nodiscard]] static core::StatusOr<PackedLinear> Create(
      const tensor::Tensor& weight,
      const std::optional<tensor::Tensor>& bias = std::nullopt);

  // y[T, out] = x[T, in] * W^T (+ bias). `x`/`y` caller-allocated contiguous
  // fp32 [T, in]/[T, out]; a wrong-shape/dtype/non-contiguous `x`/`y` is
  // InvalidArgument. Serves prefill (T large, tiled GEMM) and decode (T == 1,
  // GEMV) with no branch here (§3.4).
  [[nodiscard]] core::Status forward(const tensor::Tensor& x,
                                     tensor::Tensor& y) const override;

  [[nodiscard]] std::int64_t in_features() const override {
    return in_features_;
  }
  [[nodiscard]] std::int64_t out_features() const override {
    return out_features_;
  }
  [[nodiscard]] bool has_bias() const override { return bias_.has_value(); }

  // The packed weight `[P, K, panel_width]` in its storage dtype — the seam
  // M6-T06's tied-embedding lookup gathers a logical row out of (§7). Its
  // panel layout is defined by kernels::kNr.
  [[nodiscard]] const tensor::Tensor& packed_weight() const { return packed_; }

 private:
  PackedLinear(tensor::Tensor packed, std::optional<tensor::Tensor> bias,
               tensor::DataType weight_dtype, std::int64_t in_features,
               std::int64_t out_features)
      : packed_(std::move(packed)),
        bias_(std::move(bias)),
        weight_dtype_(weight_dtype),
        in_features_(in_features),
        out_features_(out_features) {}

  tensor::Tensor packed_;               // [P, K, kNr], weight storage dtype
  std::optional<tensor::Tensor> bias_;  // [out], fp32 (converted at pack time)
  tensor::DataType weight_dtype_;
  std::int64_t in_features_ = 0;
  std::int64_t out_features_ = 0;
};

}  // namespace engine::model
