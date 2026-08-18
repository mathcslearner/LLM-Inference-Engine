#pragma once

#include "core/status.h"
#include "tensor/tensor.h"

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

}  // namespace engine::cpu
