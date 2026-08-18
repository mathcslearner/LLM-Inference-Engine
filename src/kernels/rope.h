#pragma once

#include <cstdint>

// Rotary position embedding, in place (M6-T03; design:
// docs/design/optimized-cpu-execution.md §5, §10, §8). The optimized analog of
// `cpu::rope_apply` (the oracle): each result is validated against it within
// the §10 RoPE tolerance. HF half-rotation layout — the same tables the `Rope`
// module builds feed both backends (§2.2).
//
// For each token t (absolute position positions[t]) and head h, the pair
// (x[j], x[j+d/2]) is rotated by angle positions[t]·inv_freq[j]:
//
//   out[j]     = x[j]·cos[p,j] − x[j+d/2]·sin[p,j]
//   out[j+d/2] = x[j+d/2]·cos[p,j] + x[j]·sin[p,j]     for j ∈ [0, d/2)
//
// Both halves of each pair are read before either is written, so the rotation
// is correct in place. `positions` are arbitrary (unsorted / repeated allowed —
// this is what makes the kernel ready for later batched/paged use, §8); each
// token indexes its own cos/sin row, so tokens are independent.
//
// Numerics (§10): a pure per-element map, so bit-identical across thread
// counts; Class T across ISAs / vs the oracle (FMA in the rotate — the cos/sin
// tables are shared fp32).
//
// Raw-pointer entry: preconditions (non-null, contiguous x [t, hx, d] fp32,
// cos/sin [P, d/2] fp32, d even, positions length t, each position in [0, P))
// are the CALLER's — `OptimizedModel` front-loads validation (§5).

namespace engine::kernels {

// In-place rotary embedding on x[t, hx, d] (hx = H for queries or Hkv for
// keys). cos/sin are [P, d/2]; positions[i] selects token i's table row.
void RopeApplyF32(float* x, std::int64_t t, std::int64_t hx, std::int64_t d,
                  const std::int32_t* positions, const float* cos,
                  const float* sin);

}  // namespace engine::kernels
