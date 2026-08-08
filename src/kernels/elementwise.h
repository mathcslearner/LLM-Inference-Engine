#pragma once

#include <cstdint>

// Elementwise fp32 kernels (M3-T06; design: docs/design/cpu-backend.md
// §5–§6). Numerics class E (§6.3): per-element pure fp ops with one rounding
// each, so results are bit-identical across ISAs and thread counts — tests
// compare bit patterns, not tolerances.
//
// Shape rules, shared by every entry point:
//  - n >= 0 elements; n == 0 is a no-op.
//  - Any pointer alignment (design §7 — alignment is a performance property,
//    never a correctness precondition).
//  - `out` may exactly alias an input (out == a and/or out == b); partial
//    overlap is undefined behavior.
//
// Each entry dispatches to the SelectedIsa() variant (kernels/dispatch.h)
// and, above the per-family grain, runs it as chunk bodies inside
// parallel_for — threading is orthogonal to dispatch (design §4.2).

namespace engine::kernels {

// out[i] = a[i] + b[i].
void AddF32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] * b[i].
void MulF32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = x[i] * s.
void ScaleF32(const float* x, float s, float* out, std::int64_t n);

}  // namespace engine::kernels
