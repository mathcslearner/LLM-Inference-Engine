#pragma once

#include <cstdint>

// RMSNorm (M6-T03; design: docs/design/optimized-cpu-execution.md §4, §5, §10).
// The optimized analog of `cpu::rmsnorm` (the oracle): each result is validated
// against it within the §10 RMSNorm tolerance.
//
// y[t, :] = x[t, :] * rsqrt(mean(x[t, :]²) + eps) * weight  (HF order), per
// row. The mean of squares and the scaling run in fp32. Unlike the oracle, the
// weight is **fp32** here: the optimized backend converts norm scales to fp32
// once at load (§4 — they are read every token), so no per-element widen is on
// the hot path; the activation `x` is fp32 (the residual stream, §4).
//
// Numerics (§10): per-row sum-of-squares in a single fp32 accumulator, rows
// independent — **bit-identical across thread counts** (asserted). The
// reciprocal-sqrt is an **exact** `1/sqrtf` (or a Newton-refined `rsqrt` within
// ~1 ulp of it), never a raw hardware `rsqrte` estimate (§10 forbids it — it is
// far outside tolerance). Class T across ISAs / vs the oracle (sum order +
// FMA).
//
// Raw-pointer entry: preconditions (non-null, contiguous, x/y [rows, e] fp32,
// weight [e] fp32, rows>=1, e>=1) are the CALLER's — `OptimizedModel`
// front-loads validation (§5). `y` may alias `x`.

namespace engine::kernels {

// y[rows, e] = rmsnorm(x[rows, e], weight[e], eps). x/weight/y all fp32; y may
// alias x.
void RmsNormF32(const float* x, const float* weight, float eps,
                std::int64_t rows, std::int64_t e, float* y);

}  // namespace engine::kernels
