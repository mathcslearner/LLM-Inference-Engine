#pragma once

#include <cstdint>

// SwiGLU activation combine (M6-T03; design:
// docs/design/optimized-cpu-execution.md §5, §10). The optimized analog of
// `cpu::silu_mul` (the oracle): each result is validated against it within the
// §10 SiLU tolerance.
//
// y[i] = silu(gate[i]) * up[i], with silu(v) = v / (1 + exp(-v)) (HF `F.silu`).
// The `exp` is the vector polynomial (internal/exp_common.h). Flat over the
// full [T, I] activation (n = T*I elements) — a pure elementwise map, so
// bit-identical across thread counts and chunkings; Class T across ISAs / vs
// the oracle (vector exp differs from std::exp in rounding). Large-magnitude
// gates stay finite: exp(-v) saturates for very negative v (silu → 0), and the
// exp poly clamps the very-positive tail (silu → v).
//
// Raw-pointer entry: preconditions (non-null, n>=0 fp32, gate/up/y all length
// n) are the CALLER's — `OptimizedModel` front-loads validation (§5). `y` may
// alias gate or up.

namespace engine::kernels {

// y[i] = silu(gate[i]) * up[i] for i in [0, n). y may alias gate or up.
void SiluMulF32(const float* gate, const float* up, float* y, std::int64_t n);

}  // namespace engine::kernels
