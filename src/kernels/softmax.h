#pragma once

#include <cstdint>

// Numerically-stable row-wise softmax (M6-T03; design:
// docs/design/optimized-cpu-execution.md §5, §10). The optimized analog of
// `cpu::softmax` (the oracle): each result is validated against it within the
// §10 softmax tolerance.
//
// y[r, :] = softmax(x[r, :]) over the last dim, per row: subtract the row max,
// exponentiate (the vector `expf` polynomial, internal/exp_common.h), divide by
// the fp32 sum. Matches the oracle's contract exactly, so tests are 1:1:
//  - a `-inf` entry maps to exactly 0 (the causal-mask contract — the exp
//    polynomial flushes `x - max < kExpLo` to 0);
//  - a row that is entirely `-inf` yields NaN (a caller error, as in torch).
//
// Numerics (§10): per-row max and sum reduce in a single fp32 accumulator, rows
// independent — so the result is **bit-identical across thread counts**
// (asserted), Class T across ISAs and vs the oracle (vector `exp` + horizontal
// reductions differ in rounding).
//
// Raw-pointer entry: preconditions (non-null, contiguous [rows, n] fp32,
// rows>=1, n>=1) are the CALLER's — the reference-vs-optimized test and the
// `OptimizedModel` caller front-load validation, so nothing inside a parallel
// region does anything but arithmetic (§5). `y` may alias `x`.

namespace engine::kernels {

// y[rows, n] = row-wise softmax(x[rows, n]). y may alias x.
void SoftmaxF32(const float* x, float* y, std::int64_t rows, std::int64_t n);

}  // namespace engine::kernels
