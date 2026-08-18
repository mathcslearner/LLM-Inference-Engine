#include "kernels/internal/rope_impl.h"

#include <immintrin.h>

#include <cstdint>

// AVX2 RoPE-apply variant (x86-64 builds only; -mavx2 -mfma per-source).
// Vectorizes the half-rotation across 8 pairs (j lanes); the (lo, hi) halves
// are read before either is written, so the in-place rotation stays correct.
// Class T vs the oracle (FMA in the rotate); bit-identical across thread
// counts.

namespace engine::kernels::avx2 {

void RopeRows(float* x, std::int64_t t_count, std::int64_t hx, std::int64_t d,
              const std::int32_t* positions, const float* cos,
              const float* sin) {
  const std::int64_t half = d / 2;
  for (std::int64_t t = 0; t < t_count; ++t) {
    const auto p = static_cast<std::int64_t>(positions[t]);
    const float* cos_row = cos + (p * half);
    const float* sin_row = sin + (p * half);
    float* x_tok = x + (t * hx * d);
    for (std::int64_t h = 0; h < hx; ++h) {
      float* v = x_tok + (h * d);
      std::int64_t j = 0;
      for (; j + 8 <= half; j += 8) {
        const __m256 c = _mm256_loadu_ps(cos_row + j);
        const __m256 s = _mm256_loadu_ps(sin_row + j);
        const __m256 lo = _mm256_loadu_ps(v + j);
        const __m256 hi = _mm256_loadu_ps(v + j + half);
        // new_lo = lo*c − hi*s ; new_hi = hi*c + lo*s.
        _mm256_storeu_ps(v + j, _mm256_fnmadd_ps(hi, s, _mm256_mul_ps(lo, c)));
        _mm256_storeu_ps(v + j + half,
                         _mm256_fmadd_ps(lo, s, _mm256_mul_ps(hi, c)));
      }
      for (; j < half; ++j) {
        const float c = cos_row[j];
        const float s = sin_row[j];
        const float lo = v[j];
        const float hi = v[j + half];
        v[j] = (lo * c) - (hi * s);
        v[j + half] = (hi * c) + (lo * s);
      }
    }
  }
}

}  // namespace engine::kernels::avx2
