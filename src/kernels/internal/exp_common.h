#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// Vector-exp specification (M6-T03; design:
// docs/design/optimized-cpu-execution.md §10). The one new numerical algorithm
// M6 introduces: a polynomial `expf` approximation replacing `std::expf`,
// shared verbatim (same constants, same reduction) by the scalar reference and
// the NEON/AVX2 lane helpers, so all three ISAs run one algorithm and the only
// cross-ISA difference is FMA contraction / lane-fold order (Class T). The
// dedicated ulp sweep (`vector_exp_test`) bounds this vs `std::expf`; the
// softmax/SiLU kernels that call it derive their oracle tolerances from that
// bound.
//
// Algorithm (Cephes/`avx_mathfun` lineage): range-reduce `x = n·ln2 + r` with
// two-part (Cody-Waite) ln2 so `|r| ≤ ln2/2`, evaluate a degree-5 minimax
// polynomial for `eʳ`, then scale by `2ⁿ` by writing `n` into the fp32 exponent
// field. `n` stays in `[-126, 127]` across the guaranteed domain, so `2ⁿ` is
// always a finite normal — no denormal-trick breakage.
//
// Domain contract (the sweep asserts each clause):
//  - `x ∈ [kExpLo, kExpHi]`  → within ~2 ulp of `std::expf`.
//  - `x < kExpLo` (incl. −inf) → exactly +0.0. The true value there is
//    ≤ FLT_MIN (`expf(kExpLo) ≈ 1.18e-38`), so flushing costs ≤ FLT_MIN
//    absolute — and it makes softmax's `−inf → 0` causal-mask contract exact
//    (the property `cpu::softmax`'s goldens pin).
//  - `x > kExpHi` → saturates at `expf(kExpHi) ≈ 2.87e38` (input clamped). This
//    band is immaterial to both callers: softmax inputs are `≤ 0` after the
//    row-max subtraction, and SiLU only ever forms `exp(−v)` for very negative
//    `v`, where the result feeds `1/(1+e)` and underflows to 0 regardless.
//  - NaN in → NaN out (weights/activations are finite by precondition).

namespace engine::kernels::internal {

// Guaranteed-accuracy domain bounds (see the domain contract above).
inline constexpr float kExpHi = 88.3762626647949F;   // n_max = 127
inline constexpr float kExpLo = -87.3365478515625F;  // flush below → +0.0

// Range-reduction and minimax constants (single-precision Cephes `expf`). These
// are a matched set tuned together; `kExpC1`/`kExpC2` are the deliberate
// hi/lo split of ln2 (Cody-Waite) — NOT ln2 itself — and `kExpLog2ef` is kept
// as the Cephes literal so the polynomial's measured ≤1-ulp accuracy is not
// perturbed. Hence NOLINT on the std::numbers substitution the linter proposes.
// NOLINTNEXTLINE(modernize-use-std-numbers)
inline constexpr float kExpLog2ef = 1.44269504088896341F;  // 1/ln2
// NOLINTNEXTLINE(modernize-use-std-numbers)
inline constexpr float kExpC1 = 0.693359375F;     // ln2, hi part
inline constexpr float kExpC2 = -2.12194440e-4F;  // ln2, lo part
inline constexpr float kExpP0 = 1.9875691500e-4F;
inline constexpr float kExpP1 = 1.3981999507e-3F;
inline constexpr float kExpP2 = 8.3334519073e-3F;
inline constexpr float kExpP3 = 4.1665795894e-2F;
inline constexpr float kExpP4 = 1.6666665459e-1F;
inline constexpr float kExpP5 = 5.0000001201e-1F;

// Scalar realization of the spec above — one element. The NEON/AVX2 helpers
// (`internal/neon_exp.h`, `internal/avx2_exp.h`) mirror this exactly with
// intrinsics; results agree within the ulp bound (Class T, not bitwise, since
// the vector paths contract with FMA).
[[nodiscard]] inline float ExpF32Scalar(float x) {
  if (x < kExpLo) {
    return 0.0F;  // includes −inf → 0 (softmax causal-mask contract).
  }
  x = std::min(x, kExpHi);  // saturate; immaterial to callers (see contract).
  const float fx = std::floor((kExpLog2ef * x) + 0.5F);
  x -= fx * kExpC1;
  x -= fx * kExpC2;
  const float z = x * x;
  float y = kExpP0;
  y = (y * x) + kExpP1;
  y = (y * x) + kExpP2;
  y = (y * x) + kExpP3;
  y = (y * x) + kExpP4;
  y = (y * x) + kExpP5;
  y = (y * z) + x + 1.0F;
  // 2^n by exponent-field write: n ∈ [-126, 127] here, so (n + 127) ∈ [1, 254]
  // is always a valid finite-normal biased exponent.
  const auto n = static_cast<std::int32_t>(fx);
  const auto bits = static_cast<std::uint32_t>((n + 127) << 23);
  float pow2n = 0.0F;
  std::memcpy(&pow2n, &bits, sizeof(pow2n));
  return y * pow2n;
}

}  // namespace engine::kernels::internal
