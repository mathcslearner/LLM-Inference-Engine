#include "kernels/internal/activation_impl.h"
#include "kernels/internal/avx2_exp.h"

#include <immintrin.h>

#include <cstdint>

// AVX2 SwiGLU-combine variant (x86-64 builds only; -mavx2 -mfma per-source).
// silu(g) = g / (1 + exp(-g)) with exact division (_mm256_div_ps — never a
// reciprocal estimate, §10), exp via internal::Exp. Class T vs the oracle;
// bit-identical across thread counts.

namespace engine::kernels::avx2 {

void SiluMul(const float* gate, const float* up, float* y, std::int64_t n) {
  const __m256 one = _mm256_set1_ps(1.0F);
  const __m256 zero = _mm256_setzero_ps();
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 g = _mm256_loadu_ps(gate + i);
    const __m256 denom = _mm256_add_ps(one, Exp(_mm256_sub_ps(zero, g)));
    const __m256 silu = _mm256_div_ps(g, denom);
    _mm256_storeu_ps(y + i, _mm256_mul_ps(silu, _mm256_loadu_ps(up + i)));
  }
  for (; i < n; ++i) {
    const float g = gate[i];
    y[i] = (g / (1.0F + internal::ExpF32Scalar(-g))) * up[i];
  }
}

}  // namespace engine::kernels::avx2
