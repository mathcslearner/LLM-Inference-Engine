#include "kernels/internal/rope_impl.h"

#include <arm_neon.h>

#include <cstdint>

// NEON RoPE-apply variant (arm64 builds only). Vectorizes the half-rotation
// across 4 pairs (j lanes): the (lo, hi) halves are read before either is
// written, so the in-place rotation stays correct. Class T vs the oracle (FMA
// in the rotate); bit-identical across thread counts.

namespace engine::kernels::neon {

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
      for (; j + 4 <= half; j += 4) {
        const float32x4_t c = vld1q_f32(cos_row + j);
        const float32x4_t s = vld1q_f32(sin_row + j);
        const float32x4_t lo = vld1q_f32(v + j);
        const float32x4_t hi = vld1q_f32(v + j + half);
        // new_lo = lo*c − hi*s ; new_hi = hi*c + lo*s.
        vst1q_f32(v + j, vfmsq_f32(vmulq_f32(lo, c), hi, s));
        vst1q_f32(v + j + half, vfmaq_f32(vmulq_f32(hi, c), lo, s));
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

}  // namespace engine::kernels::neon
