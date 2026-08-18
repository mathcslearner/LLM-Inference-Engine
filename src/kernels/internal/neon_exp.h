#pragma once

#include "kernels/internal/exp_common.h"

#include <arm_neon.h>

// NEON lane-parallel `expf` (arm64 builds only). Included ONLY by NEON per-ISA
// TUs (softmax/activation/exp) — it pulls in <arm_neon.h>, so it must never
// reach a portable header (cpu-backend.md §4.4 rule 2). Implements the
// exp_common.h spec across four fp32 lanes; agreement with ExpF32Scalar is
// within the ulp bound (§10), not bitwise (FMA contraction differs).

namespace engine::kernels::neon {

[[nodiscard]] inline float32x4_t Exp(float32x4_t x) {
  const uint32x4_t flush = vcltq_f32(x, vdupq_n_f32(internal::kExpLo));
  x = vminq_f32(x, vdupq_n_f32(internal::kExpHi));

  // n = floor(x/ln2 + 0.5); r = x - n*ln2 (two-part).
  float32x4_t fx =
      vmlaq_f32(vdupq_n_f32(0.5F), x, vdupq_n_f32(internal::kExpLog2ef));
  fx = vrndmq_f32(fx);  // round toward −inf == floor (ARMv8 baseline).
  x = vfmsq_f32(x, fx, vdupq_n_f32(internal::kExpC1));
  x = vfmsq_f32(x, fx, vdupq_n_f32(internal::kExpC2));

  const float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(internal::kExpP0);
  y = vfmaq_f32(vdupq_n_f32(internal::kExpP1), y, x);
  y = vfmaq_f32(vdupq_n_f32(internal::kExpP2), y, x);
  y = vfmaq_f32(vdupq_n_f32(internal::kExpP3), y, x);
  y = vfmaq_f32(vdupq_n_f32(internal::kExpP4), y, x);
  y = vfmaq_f32(vdupq_n_f32(internal::kExpP5), y, x);
  y = vfmaq_f32(vaddq_f32(x, vdupq_n_f32(1.0F)), y, z);  // y*z + (x + 1)

  // 2^n via exponent field: (n + 127) << 23, n ∈ [-126, 127] → valid normal.
  int32x4_t n = vcvtq_s32_f32(fx);  // fx integral; truncation == value.
  n = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  const float32x4_t pow2n = vreinterpretq_f32_s32(n);

  const float32x4_t res = vmulq_f32(y, pow2n);
  return vbslq_f32(flush, vdupq_n_f32(0.0F), res);  // flushed lanes → +0.0.
}

}  // namespace engine::kernels::neon
