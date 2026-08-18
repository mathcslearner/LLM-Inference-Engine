#include "kernels/internal/norm_impl.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>

// AVX2 RMSNorm variant (x86-64 builds only; -mavx2 -mfma per-source). Sum of
// squares via an 8-lane FMA accumulator folded once, an **exact** scalar
// 1/sqrtf (never rsqrtps, §10), then a vector scale by inv_rms and the fp32
// weight. Class T vs the oracle; bit-identical across thread counts.

namespace engine::kernels::avx2 {

namespace {

[[nodiscard]] inline float ReduceSum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
  return _mm_cvtss_f32(s);
}

}  // namespace

void RmsNormRows(const float* x, const float* weight, float eps,
                 std::int64_t rows, std::int64_t e, float* y) {
  const auto e_f = static_cast<float>(e);
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x_row = x + (r * e);
    float* y_row = y + (r * e);

    __m256 vacc = _mm256_setzero_ps();
    std::int64_t j = 0;
    for (; j + 8 <= e; j += 8) {
      const __m256 v = _mm256_loadu_ps(x_row + j);
      vacc = _mm256_fmadd_ps(v, v, vacc);
    }
    float sum_sq = ReduceSum8(vacc);
    for (; j < e; ++j) {
      sum_sq += x_row[j] * x_row[j];
    }

    const float inv_rms = 1.0F / std::sqrt((sum_sq / e_f) + eps);
    const __m256 vinv = _mm256_set1_ps(inv_rms);
    j = 0;
    for (; j + 8 <= e; j += 8) {
      const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(x_row + j), vinv);
      _mm256_storeu_ps(y_row + j,
                       _mm256_mul_ps(scaled, _mm256_loadu_ps(weight + j)));
    }
    for (; j < e; ++j) {
      y_row[j] = x_row[j] * inv_rms * weight[j];
    }
  }
}

}  // namespace engine::kernels::avx2
