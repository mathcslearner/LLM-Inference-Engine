#include "kernels/internal/activation_impl.h"
#include "kernels/internal/neon_exp.h"

#include <arm_neon.h>

#include <cstdint>

// NEON SwiGLU-combine variant (arm64 builds only). silu(g) = g / (1 + exp(-g))
// with exact division (vdivq_f32 — never a reciprocal estimate, §10), exp via
// internal::Exp. Class T vs the oracle; bit-identical across thread counts.

namespace engine::kernels::neon {

void SiluMul(const float* gate, const float* up, float* y, std::int64_t n) {
  const float32x4_t one = vdupq_n_f32(1.0F);
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t g = vld1q_f32(gate + i);
    const float32x4_t denom = vaddq_f32(one, Exp(vnegq_f32(g)));
    const float32x4_t silu = vdivq_f32(g, denom);
    vst1q_f32(y + i, vmulq_f32(silu, vld1q_f32(up + i)));
  }
  for (; i < n; ++i) {
    const float g = gate[i];
    y[i] = (g / (1.0F + internal::ExpF32Scalar(-g))) * up[i];
  }
}

}  // namespace engine::kernels::neon
