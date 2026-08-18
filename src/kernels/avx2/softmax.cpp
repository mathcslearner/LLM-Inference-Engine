#include "kernels/internal/avx2_exp.h"
#include "kernels/internal/softmax_impl.h"

#include <immintrin.h>

#include <cstdint>

// AVX2 softmax variant (x86-64 builds only; -mavx2 -mfma per-source). Row max
// via a vector max reduction, exp via internal::Exp, row sum via an 8-lane
// accumulator folded once, then a vector normalize. Class T vs the scalar
// variant and the oracle; bit-identical across thread counts (each row wholly
// in one call).

namespace engine::kernels::avx2 {

namespace {

// Horizontal max of eight lanes.
[[nodiscard]] inline float ReduceMax8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_max_ps(lo, hi);
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x1));
  return _mm_cvtss_f32(m);
}

// Horizontal sum of eight lanes.
[[nodiscard]] inline float ReduceSum8(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
  return _mm_cvtss_f32(s);
}

// Row max seeded at x[0]; a `-inf` lane never wins unless the row is all
// `-inf`.
[[nodiscard]] float RowMax(const float* x, std::int64_t n) {
  std::int64_t i = 0;
  float m = x[0];
  if (n >= 8) {
    __m256 acc = _mm256_loadu_ps(x);
    for (i = 8; i + 8 <= n; i += 8) {
      acc = _mm256_max_ps(acc, _mm256_loadu_ps(x + i));
    }
    m = ReduceMax8(acc);
  } else {
    i = 1;
  }
  for (; i < n; ++i) {
    m = x[i] > m ? x[i] : m;
  }
  return m;
}

}  // namespace

void SoftmaxRows(const float* x, float* y, std::int64_t rows, std::int64_t n) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x_row = x + (r * n);
    float* y_row = y + (r * n);

    const float row_max = RowMax(x_row, n);
    const __m256 vmax = _mm256_set1_ps(row_max);

    __m256 vsum = _mm256_setzero_ps();
    float sum = 0.0F;
    std::int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
      const __m256 e = Exp(_mm256_sub_ps(_mm256_loadu_ps(x_row + j), vmax));
      _mm256_storeu_ps(y_row + j, e);
      vsum = _mm256_add_ps(vsum, e);
    }
    sum = ReduceSum8(vsum);
    for (; j < n; ++j) {
      const float e = internal::ExpF32Scalar(x_row[j] - row_max);
      y_row[j] = e;
      sum += e;
    }

    const __m256 vinv = _mm256_set1_ps(1.0F / sum);
    j = 0;
    for (; j + 8 <= n; j += 8) {
      _mm256_storeu_ps(y_row + j,
                       _mm256_mul_ps(_mm256_loadu_ps(y_row + j), vinv));
    }
    const float inv = 1.0F / sum;
    for (; j < n; ++j) {
      y_row[j] *= inv;
    }
  }
}

}  // namespace engine::kernels::avx2
