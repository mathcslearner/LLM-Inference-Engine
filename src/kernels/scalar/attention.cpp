#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "kernels/internal/exp_common.h"

#include <cstdint>
#include <limits>

// Scalar prefill-attention variant (compiled on every platform — the fallback
// and forced-scalar target). It embeds the shared exp polynomial
// (ExpF32Scalar), not std::expf, so the forced-scalar pass exercises the same
// numeric code the vector paths run (design §10). Class T vs the oracle (the
// online-softmax rescale + the polynomial exp), bit-identical across thread
// counts. All four primitives are plain ascending loops — the correctness
// reference the NEON/AVX2 Ops mirror.

namespace engine::kernels::scalar {

namespace {

struct ScalarOps {
  static float DotScoreRow(const float* q, const float* k_block, std::int64_t n,
                           std::int64_t d, float scale, float* scores) {
    float row_max = -std::numeric_limits<float>::infinity();
    for (std::int64_t j = 0; j < n; ++j) {
      const float* k_vec = k_block + (j * d);
      float dot = 0.0F;
      for (std::int64_t e = 0; e < d; ++e) {
        dot += q[e] * k_vec[e];
      }
      const float s = dot * scale;  // HF order: scale the completed dot.
      scores[j] = s;
      row_max = s > row_max ? s : row_max;
    }
    return row_max;
  }

  static float ExpRowSum(float* scores, std::int64_t n, float m_new) {
    float sum = 0.0F;
    for (std::int64_t j = 0; j < n; ++j) {
      const float e = internal::ExpF32Scalar(scores[j] - m_new);
      scores[j] = e;
      sum += e;
    }
    return sum;
  }

  static void ScaleRow(float* out, float s, std::int64_t d) {
    for (std::int64_t e = 0; e < d; ++e) {
      out[e] *= s;
    }
  }

  static void AxpyRow(float* out, float s, const float* x, std::int64_t d) {
    for (std::int64_t e = 0; e < d; ++e) {
      out[e] += s * x[e];
    }
  }
};

}  // namespace

void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end) {
  internal::PrefillUnitsImpl<ScalarOps>(a, unit_begin, unit_end);
}

}  // namespace engine::kernels::scalar
