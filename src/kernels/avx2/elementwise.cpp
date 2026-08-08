#include "kernels/internal/elementwise_impl.h"

#include <immintrin.h>

// AVX2 elementwise variants (x86-64 builds only; compiled with per-source
// -mavx2 -mfma -mf16c, design §4.4 rule 1). Unaligned loads throughout
// (loadu/storeu, design §7); scalar tail epilogues are exact per element,
// so Class E bit-identity holds for any split between body and tail. No
// FMA: these ops are single-rounding by definition.

namespace engine::kernels::avx2 {

void AddF32(const float* a, const float* b, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(
        out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  for (; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void MulF32(const float* a, const float* b, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(
        out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
  for (; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
}

void ScaleF32(const float* x, float s, float* out, std::int64_t n) {
  const __m256 vs = _mm256_set1_ps(s);
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
  }
  for (; i < n; ++i) {
    out[i] = x[i] * s;
  }
}

}  // namespace engine::kernels::avx2
