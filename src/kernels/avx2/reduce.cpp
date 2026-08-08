#include "kernels/internal/reduce_common.h"
#include "kernels/internal/reduce_impl.h"

#include <immintrin.h>

// AVX2 reduction variants (x86-64 builds only). SumF32Chunk realizes the
// 16-lane convention (reduce.h) as 2×__m256: accumulator k's lane l is spec
// lane 8k + l, so a full 16-element step routes element 8k + l into exactly
// the lane the scalar variant uses. The tail adds per element into the
// spilled lane array — same lanes, same order — and the 16→1 fold is the
// shared internal::FoldLanes16, literally the scalar code. Plain adds only;
// no FMA in reductions (design §6.3).

namespace engine::kernels::avx2 {

float SumF32Chunk(const float* x, std::int64_t n) {
  __m256 acc0 = _mm256_setzero_ps();  // +0.0F init is part of the spec.
  __m256 acc1 = _mm256_setzero_ps();
  std::int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(x + i));
    acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(x + i + 8));
  }
  float lanes[16];
  _mm256_storeu_ps(lanes, acc0);
  _mm256_storeu_ps(lanes + 8, acc1);
  for (; i < n; ++i) {
    lanes[i % 16] += x[i];
  }
  return internal::FoldLanes16(lanes);
}

float MaxF32Chunk(const float* x, std::int64_t n) {
  std::int64_t i = 0;
  float m = x[0];
  if (n >= 8) {
    __m256 acc = _mm256_loadu_ps(x);
    for (i = 8; i + 8 <= n; i += 8) {
      const __m256 v = _mm256_loadu_ps(x + i);
      // maxps alone returns its second operand on equality, which gets the
      // -0/+0 tie wrong (reduce.h specifies -0.0F < +0.0F). On equal lanes
      // — bit-identical unless ±0 — the bitwise AND of the operands is the
      // total-order answer (+0 wins unless both are -0), so blend it in.
      const __m256 mx = _mm256_max_ps(acc, v);
      const __m256 eq = _mm256_cmp_ps(acc, v, _CMP_EQ_OQ);
      acc = _mm256_blendv_ps(mx, _mm256_and_ps(acc, v), eq);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    m = lanes[0];
    for (int j = 1; j < 8; ++j) {
      m = internal::TotalOrderMax(m, lanes[j]);
    }
  } else {
    i = 1;
  }
  for (; i < n; ++i) {
    m = internal::TotalOrderMax(m, x[i]);
  }
  return m;
}

}  // namespace engine::kernels::avx2
