#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "kernels/internal/exp_common.h"
#include "kernels/internal/neon_exp.h"

#include <arm_neon.h>

#include <cstdint>
#include <limits>

// NEON prefill (M6-T04) + decode (M6-T05) attention variants (arm64 builds
// only). The four Ops primitives vectorize over `d` (the dot's contraction, the
// V axpy, the row scale) and over the key row (the block exp), with scalar
// tails; the online-softmax control flow is the shared
// internal::PrefillUnitsImpl / DecodeUnitsImpl (both drive the same Ops). Class
// T vs the scalar variant and the oracle (FMA contraction + horizontal
// reductions + the vector exp differ in rounding); bit-identical across thread
// counts (each unit's recurrence wholly in one call). exp uses the shared
// polynomial (neon::Exp).

namespace engine::kernels::neon {

namespace {

struct NeonOps {
  static float DotScoreRow(const float* q, const float* k_block, std::int64_t n,
                           std::int64_t d, float scale, float* scores) {
    float row_max = -std::numeric_limits<float>::infinity();
    for (std::int64_t j = 0; j < n; ++j) {
      const float* k_vec = k_block + (j * d);
      float32x4_t acc = vdupq_n_f32(0.0F);
      std::int64_t e = 0;
      for (; e + 4 <= d; e += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(q + e), vld1q_f32(k_vec + e));
      }
      float dot = vaddvq_f32(acc);
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
    const float32x4_t vm = vdupq_n_f32(m_new);
    float32x4_t vsum = vdupq_n_f32(0.0F);
    std::int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      const float32x4_t e = Exp(vsubq_f32(vld1q_f32(scores + j), vm));
      vst1q_f32(scores + j, e);
      vsum = vaddq_f32(vsum, e);
    }
    float sum = vaddvq_f32(vsum);
    for (; j < n; ++j) {
      const float e = internal::ExpF32Scalar(scores[j] - m_new);
      scores[j] = e;
      sum += e;
    }
    return sum;
  }

  static void ScaleRow(float* out, float s, std::int64_t d) {
    const float32x4_t vs = vdupq_n_f32(s);
    std::int64_t e = 0;
    for (; e + 4 <= d; e += 4) {
      vst1q_f32(out + e, vmulq_f32(vld1q_f32(out + e), vs));
    }
    for (; e < d; ++e) {
      out[e] *= s;
    }
  }

  static void AxpyRow(float* out, float s, const float* x, std::int64_t d) {
    const float32x4_t vs = vdupq_n_f32(s);
    std::int64_t e = 0;
    for (; e + 4 <= d; e += 4) {
      vst1q_f32(out + e, vfmaq_f32(vld1q_f32(out + e), vs, vld1q_f32(x + e)));
    }
    for (; e < d; ++e) {
      out[e] += s * x[e];
    }
  }
};

}  // namespace

void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end) {
  internal::PrefillUnitsImpl<NeonOps>(a, unit_begin, unit_end);
}

void DecodeUnits(const internal::DecodeArgs& a, std::int64_t unit_begin,
                 std::int64_t unit_end) {
  internal::DecodeUnitsImpl<NeonOps>(a, unit_begin, unit_end);
}

void PagedDecodeUnits(const internal::PagedDecodeArgs& a,
                      std::int64_t unit_begin, std::int64_t unit_end) {
  internal::PagedDecodeUnitsImpl<NeonOps>(a, unit_begin, unit_end);
}

}  // namespace engine::kernels::neon
