#include "kernels/internal/norm_impl.h"

#include <arm_neon.h>

#include <cmath>
#include <cstdint>

// NEON RMSNorm variant (arm64 builds only). Sum of squares via a 4-lane FMA
// accumulator folded once, an **exact** scalar 1/sqrtf (never vrsqrte, §10),
// then a vector scale by inv_rms and the fp32 weight. Class T vs the oracle;
// bit-identical across thread counts (each row wholly in one call).

namespace engine::kernels::neon {

void RmsNormRows(const float* x, const float* weight, float eps,
                 std::int64_t rows, std::int64_t e, float* y) {
  const auto e_f = static_cast<float>(e);
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x_row = x + (r * e);
    float* y_row = y + (r * e);

    float32x4_t vacc = vdupq_n_f32(0.0F);
    std::int64_t j = 0;
    for (; j + 4 <= e; j += 4) {
      const float32x4_t v = vld1q_f32(x_row + j);
      vacc = vfmaq_f32(vacc, v, v);
    }
    float sum_sq = vaddvq_f32(vacc);
    for (; j < e; ++j) {
      sum_sq += x_row[j] * x_row[j];
    }

    const float inv_rms = 1.0F / std::sqrt((sum_sq / e_f) + eps);
    const float32x4_t vinv = vdupq_n_f32(inv_rms);
    j = 0;
    for (; j + 4 <= e; j += 4) {
      const float32x4_t scaled = vmulq_f32(vld1q_f32(x_row + j), vinv);
      vst1q_f32(y_row + j, vmulq_f32(scaled, vld1q_f32(weight + j)));
    }
    for (; j < e; ++j) {
      y_row[j] = x_row[j] * inv_rms * weight[j];
    }
  }
}

}  // namespace engine::kernels::neon
