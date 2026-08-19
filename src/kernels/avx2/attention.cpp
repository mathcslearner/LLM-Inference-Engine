#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "kernels/internal/avx2_exp.h"
#include "kernels/internal/exp_common.h"

#include <immintrin.h>

#include <cstdint>
#include <limits>

// AVX2 prefill-attention variant (x86-64 builds only; compiled -mavx2 -mfma).
// The 8-lane mirror of the NEON variant: same four Ops primitives over `d` and
// the key row, same shared online-softmax control flow
// (internal::PrefillUnitsImpl), the shared exp polynomial (avx2::Exp). Class T
// vs the scalar variant and the oracle; bit-identical across thread counts.
// Written blind on the arm64 dev machine and proven by CI's x86-64 build
// (design §9, CLAUDE.md tidy note); reviewed by hand against .clang-tidy.

namespace engine::kernels::avx2 {

namespace {

// Horizontal sum of the 8 lanes of an __m256.
[[nodiscard]] inline float HSum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

struct Avx2Ops {
  static float DotScoreRow(const float* q, const float* k_block, std::int64_t n,
                           std::int64_t d, float scale, float* scores) {
    float row_max = -std::numeric_limits<float>::infinity();
    for (std::int64_t j = 0; j < n; ++j) {
      const float* k_vec = k_block + (j * d);
      __m256 acc = _mm256_setzero_ps();
      std::int64_t e = 0;
      for (; e + 8 <= d; e += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + e),
                              _mm256_loadu_ps(k_vec + e), acc);
      }
      float dot = HSum256(acc);
      for (; e < d; ++e) {
        dot += q[e] * k_vec[e];
      }
      const float s = dot * scale;  // HF order: scale the completed dot.
      scores[j] = s;
      row_max = s > row_max ? s : row_max;
    }
    return row_max;
  }

  static float ExpRowSum(float* scores, std::int64_t n, float m_new) {
    const __m256 vm = _mm256_set1_ps(m_new);
    __m256 vsum = _mm256_setzero_ps();
    std::int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
      const __m256 e = Exp(_mm256_sub_ps(_mm256_loadu_ps(scores + j), vm));
      _mm256_storeu_ps(scores + j, e);
      vsum = _mm256_add_ps(vsum, e);
    }
    float sum = HSum256(vsum);
    for (; j < n; ++j) {
      const float e = internal::ExpF32Scalar(scores[j] - m_new);
      scores[j] = e;
      sum += e;
    }
    return sum;
  }

  static void ScaleRow(float* out, float s, std::int64_t d) {
    const __m256 vs = _mm256_set1_ps(s);
    std::int64_t e = 0;
    for (; e + 8 <= d; e += 8) {
      _mm256_storeu_ps(out + e, _mm256_mul_ps(_mm256_loadu_ps(out + e), vs));
    }
    for (; e < d; ++e) {
      out[e] *= s;
    }
  }

  static void AxpyRow(float* out, float s, const float* x, std::int64_t d) {
    const __m256 vs = _mm256_set1_ps(s);
    std::int64_t e = 0;
    for (; e + 8 <= d; e += 8) {
      _mm256_storeu_ps(out + e, _mm256_fmadd_ps(vs, _mm256_loadu_ps(x + e),
                                                _mm256_loadu_ps(out + e)));
    }
    for (; e < d; ++e) {
      out[e] += s * x[e];
    }
  }
};

}  // namespace

void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end) {
  internal::PrefillUnitsImpl<Avx2Ops>(a, unit_begin, unit_end);
}

}  // namespace engine::kernels::avx2
