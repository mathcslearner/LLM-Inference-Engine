#pragma once

#include "kernels/internal/exp_common.h"

#include <immintrin.h>

// AVX2 lane-parallel `expf` (x86-64 builds only). Included ONLY by AVX2 per-ISA
// TUs (compiled with -mavx2 -mfma), never by a portable header
// (cpu-backend.md §4.4 rule 2). Implements the exp_common.h spec across eight
// fp32 lanes; agreement with ExpF32Scalar is within the ulp bound (§10), not
// bitwise (FMA contraction differs).

namespace engine::kernels::avx2 {

[[nodiscard]] inline __m256 Exp(__m256 x) {
  const __m256 flush =
      _mm256_cmp_ps(x, _mm256_set1_ps(internal::kExpLo), _CMP_LT_OQ);
  x = _mm256_min_ps(x, _mm256_set1_ps(internal::kExpHi));

  // n = floor(x/ln2 + 0.5); r = x - n*ln2 (two-part).
  __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(internal::kExpLog2ef),
                              _mm256_set1_ps(0.5F));
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(internal::kExpC1), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(internal::kExpC2), x);

  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(internal::kExpP0);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(internal::kExpP1));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(internal::kExpP2));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(internal::kExpP3));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(internal::kExpP4));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(internal::kExpP5));
  y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0F)));

  // 2^n via exponent field: (n + 127) << 23, n ∈ [-126, 127] → valid normal.
  __m256i n = _mm256_cvttps_epi32(fx);  // fx integral; truncation == value.
  n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
  const __m256 pow2n = _mm256_castsi256_ps(n);

  const __m256 res = _mm256_mul_ps(y, pow2n);
  return _mm256_blendv_ps(res, _mm256_setzero_ps(), flush);  // flushed → +0.0.
}

}  // namespace engine::kernels::avx2
