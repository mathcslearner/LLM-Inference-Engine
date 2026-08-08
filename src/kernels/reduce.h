#pragma once

#include <cstdint>

// fp32 reduction kernels (M3-T06; design: docs/design/cpu-backend.md §3.3,
// §6.3). Numerics class R: bit-identical across thread counts (fixed
// pairwise chunk tree, parallel_reduce) and across ISAs (the fixed 16-lane
// accumulation convention below).
//
// Shared preconditions: any pointer alignment (design §7); inputs must be
// NaN-free — for MaxF32 because NaN is unordered, for SumF32 because NaN
// payload propagation through fp adds is ISA-specific, so a NaN input would
// break the cross-platform bit-identity contract (±inf is fine as long as
// the sum never forms inf + -inf).

namespace engine::kernels {

// The reduction grain (design §3.2: a named per-call-site constant). Part of
// SumF32's bit-exact specification, not a tuning knob: it fixes the chunk
// decomposition, hence the per-chunk partials and the fold-tree shape —
// changing it changes results at the last-ulp level. (MaxF32 shares it for
// uniformity but is order-insensitive, so its value never affects results.)
inline constexpr std::int64_t kReduceGrain = 32768;

// Sum of x[0..n). n >= 0; returns +0.0F when n == 0.
//
// Bit-exact specification (design §6.3, the fixed 16-lane convention):
// the range is chunked by (n, kReduceGrain); within a chunk, 16 fp32
// accumulators — initialized to +0.0F — accumulate with plain adds (no FMA),
// the element at chunk-relative offset k landing in lane k mod 16; lanes
// fold by fixed halving (lane[j] += lane[j+8], then +4, +2, +1); chunk
// partials fold in parallel_reduce's fixed pairwise tree. Every ISA variant
// implements exactly this, so the result is one bit pattern on every
// machine, backend, and thread count. (+0.0F lane init implies e.g.
// SumF32 of {-0.0F} is +0.0F.)
float SumF32(const float* x, std::int64_t n);

// Maximum of x[0..n). n >= 1 (CHECK — an empty max has no identity in fp32).
//
// The order is IEEE totalOrder-like on the NaN-free domain: ordinary
// comparison, sharpened with -0.0F < +0.0F so ±0 ties resolve identically
// on every ISA (NEON FMAX has this semantic natively; scalar and AVX2
// implement it explicitly). Total-order max is associative and commutative,
// so bit-identity needs no lane convention.
float MaxF32(const float* x, std::int64_t n);

}  // namespace engine::kernels
