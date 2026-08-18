#pragma once

#include "core/status.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <span>

// Reference CPU ops — the correctness oracle (docs/design/cpu-backend.md §2.1,
// docs/design/model-execution.md §2). These are deliberately unoptimized,
// clarity-first, fp32-accumulation implementations: every optimized/vectorized
// kernel from M6 on is validated against them, and they in turn are validated
// against HuggingFace fixtures. An oracle you optimize is an oracle you no
// longer trust, so nothing here is tuned.
//
// Layering (ADR-002, model-execution.md §2.1): `cpu` links `tensor` and
// `parallel` only — never `kernels`. On-the-fly fp16/bf16 -> fp32 widening
// therefore goes through tensor/half.h's value-type conversions
// (`operator float()`), not kernels/convert.
//
// Numerics (cpu-backend.md §5, §6.3): all arithmetic accumulates in fp32
// regardless of weight storage dtype. GEMM is Class T against HF fixtures (HF
// reduces in its own order); but internally each output element sums its K
// terms in a single fp32 accumulator in ascending-k order, so a `cpu::gemm`
// result is *bit-identical* across thread counts and tile sizes — a property
// the tests exploit (against a serial triple loop) independent of any fixture.

namespace engine::cpu {

// C[M, N] = A[M, K] * B[N, K]^T  (+ bias[N] if `bias != nullptr`).
//
// This is the "NT" (non-transposed A, transposed B) form and the only GEMM
// shape the reference provides, because every `Linear` in the model is exactly
// it: with the weight stored row-major `[out, in]` (checkpoint order,
// model-loading.md §4) the inner K loop is a contiguous dot product over both
// operands, which is what keeps the reference obvious. Attention's
// score/context matmuls (M5-T05) are written as their own loop nests in
// `cpu::attention`, so no transpose flag is needed here.
//
//   a     : [M, K], contiguous, dtype kFloat32.
//   b     : [N, K], contiguous, dtype kFloat32 / kFloat16 / kBFloat16 (weight
//           storage dtype; widened per element as it is read).
//   bias  : nullptr, or [N] contiguous, dtype kFloat32 / kFloat16 / kBFloat16.
//           Added once after the K accumulation (not seeded into the
//           accumulator), matching torch addmm's effective rounding.
//   c     : [M, N], contiguous, dtype kFloat32, caller-allocated — fully
//           overwritten (this call does not read `c`).
//
// All of M, N, K must be >= 1 (K == 1 is allowed and exercised). Malformed
// inputs — wrong rank/dtype/shape, non-contiguous operand, K mismatch, wrong
// bias length — are recoverable InvalidArgument naming the offending input
// (ADR-003); the caller-allocated `c` having the wrong shape/dtype is likewise
// InvalidArgument. Undefined tensors are a programmer error (CHECK).
[[nodiscard]] core::Status gemm(const tensor::Tensor& a,
                                const tensor::Tensor& b,
                                const tensor::Tensor* bias, tensor::Tensor& c);

// y[T, E] = rmsnorm(x)[T, E] * weight[E], per HF LlamaRMSNorm computed in fp32
// (model-execution.md §4.2): for each row, `y = x * rsqrt(mean(x²) + eps) *
// weight`, the mean-of-squares taken over the hidden dimension E in a single
// ascending fp32 accumulator. This is the *pure fp32 forward* — it deliberately
// omits HF's intermediate `.to(input_dtype)` round-trip, matching the fixture
// goldens, which are the fp32 forward of the checkpoint (fixtures README,
// §3.3).
//
//   x      : [T, E], contiguous, dtype kFloat32 / kFloat16 / kBFloat16. The op
//            accepts half storage (the "RMSNorm on bf16 input" criterion, §12)
//            and widens per element; the M5 model graph only ever passes f32.
//   weight : [E], contiguous, dtype kFloat32 / kFloat16 / kBFloat16 (checkpoint
//            storage; widened per element).
//   eps    : added to the mean of squares before rsqrt (config.rms_norm_eps).
//   y      : [T, E], contiguous, dtype kFloat32, caller-allocated — fully
//            overwritten. Aliasing y with x is allowed only when x is f32.
//
// T >= 1, E >= 1. Rows are independent and each reduces in a single fp32
// accumulator, so the result is bit-identical across thread counts. Malformed
// inputs — wrong rank/dtype/shape, non-contiguous operand, E mismatch between x
// and weight, wrong y — are recoverable InvalidArgument naming the offending
// input (ADR-003). Undefined handles are a programmer error (CHECK).
[[nodiscard]] core::Status rmsnorm(const tensor::Tensor& x,
                                   const tensor::Tensor& weight, float eps,
                                   tensor::Tensor& y);

// y[T, I] = silu(gate)[T, I] (elementwise *) up[T, I] — the SwiGLU activation
// (model-execution.md §4.2). `silu(v) = v / (1 + exp(-v))` (HF `F.silu`),
// computed in fp32; large-magnitude gates are handled without overflow to NaN
// (a saturating exp is finite). All of gate/up/y are [T, I] contiguous
// kFloat32 (activations are fp32, §3.3); y is caller-allocated and may alias
// gate or up. Malformed inputs are recoverable InvalidArgument (ADR-003);
// undefined handles are CHECK.
[[nodiscard]] core::Status silu_mul(const tensor::Tensor& gate,
                                    const tensor::Tensor& up,
                                    tensor::Tensor& y);

// y[T, E] = a[T, E] + b[T, E] — the residual add (model-execution.md §4.2).
// All of a/b/y are [T, E] contiguous kFloat32; y is caller-allocated and may
// alias a or b. Malformed inputs are recoverable InvalidArgument (ADR-003);
// undefined handles are CHECK.
[[nodiscard]] core::Status add(const tensor::Tensor& a, const tensor::Tensor& b,
                               tensor::Tensor& y);

// y[R, N] = softmax(x)[R, N] over the last dimension, numerically stable
// (model-execution.md §4.2): per row, subtract the row max, exponentiate, and
// divide by the fp32 sum of exponentials. A `-inf` entry maps to exactly 0
// (what M5-T05's causal mask relies on); a row that is entirely `-inf` yields
// NaN and is a caller error (torch does likewise). x and y are [R, N]
// contiguous kFloat32 (attention scores are fp32, §3.1); y is caller-allocated
// and may alias x. Rows are independent — bit-identical across thread counts.
// Malformed inputs are recoverable InvalidArgument (ADR-003); undefined handles
// are CHECK.
[[nodiscard]] core::Status softmax(const tensor::Tensor& x, tensor::Tensor& y);

// y[T, E] = table[ids[t], :] widened to fp32 — the token-embedding lookup
// (model-execution.md §4, §7). Each output row is the gathered table row
// converted per element via tensor/half.h, so a bf16/f16 embedding table is
// read zero-copy (M4 preserves its storage dtype) and never up-converted at
// load. Duplicate ids gather the same row independently.
//
//   table : [V, E], contiguous, dtype kFloat32 / kFloat16 / kBFloat16 (weight
//           storage dtype; widened per element as it is read).
//   ids   : [T], each in [0, V). The `ForwardRequest.token_ids` span is passed
//           straight through — no allocation, no i64 copy.
//   y     : [T, E], contiguous, dtype kFloat32, caller-allocated — fully
//           overwritten.
//
// T (== ids.size()) >= 1, E >= 1. Rows are independent (a pure gather + widen),
// so the result is bit-identical across thread counts. An id outside [0, V) is
// a recoverable InvalidArgument naming the offending index *and* value; wrong
// rank/dtype/shape, a non-contiguous operand, or a T/E disagreement between
// `ids`/`table`/`y` are likewise InvalidArgument (ADR-003). Undefined handles
// are a programmer error (CHECK).
[[nodiscard]] core::Status embedding_lookup(const tensor::Tensor& table,
                                            std::span<const std::int32_t> ids,
                                            tensor::Tensor& y);

// In-place rotary position embedding on x[T, Hx, d], HF half-rotation layout
// (model-execution.md §7). For each token t (absolute position p =
// positions[t]) and head h, the pair (x[j], x[j+d/2]) is rotated by angle
// `p · inv_freq[j]`:
//
//   out[j]     = x[j]     · cos[p,j] − x[j+d/2] · sin[p,j]
//   out[j+d/2] = x[j+d/2] · cos[p,j] + x[j]     · sin[p,j]     for j ∈ [0, d/2)
//
// This is the GPT-NeoX / HF-Llama "half" layout, **not** interleaved pairs. The
// precomputed `cos`/`sin` tables (built once by the `Rope` module from
// `config.rope_theta` and any `rope_scaling`, §7) supply `cos[p,j]`; the op is
// layout-agnostic about how they were produced. Both operands of each pair are
// read before either is written, so the rotation is correct in place.
//
//   x         : [T, Hx, d], contiguous, dtype kFloat32, rotated in place. Hx is
//               H (queries) or Hkv (keys) — the op is called once per tensor.
//               d must be even.
//   positions : [T], each in [0, rows(cos)). This is where §5's
//               `max(positions) < max_position_embeddings` requirement
//               materializes for the reference — an out-of-range position names
//               the offending index/value.
//   cos, sin  : [P, d/2], contiguous, dtype kFloat32, identical shape, P >= 1
//               (the table's position count).
//
// T (== positions.size()) >= 1, Hx >= 1. The rotation is a pure per-element
// map, so the result is bit-identical across thread counts. Malformed inputs —
// wrong rank/dtype/shape, non-contiguous operand, odd d, cos/sin shape
// mismatch, a positions/T disagreement, an out-of-range position — are
// recoverable InvalidArgument naming the offending input (ADR-003). Undefined
// handles are a programmer error (CHECK).
[[nodiscard]] core::Status rope_apply(tensor::Tensor& x,
                                      std::span<const std::int32_t> positions,
                                      const tensor::Tensor& cos,
                                      const tensor::Tensor& sin);

}  // namespace engine::cpu
