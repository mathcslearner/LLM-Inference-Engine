#include "kernels/internal/elementwise_impl.h"

#include <arm_neon.h>

// NEON elementwise variants (arm64 builds only; design §4.4 — NEON is the
// AArch64 baseline, so this TU carries no extra flags). Unaligned loads
// throughout: vld1q/vst1q are alignment-agnostic (design §7). Scalar tail
// epilogues are exact per element, so Class E bit-identity holds for any
// split between body and tail.

namespace engine::kernels::neon {

void AddF32(const float* a, const float* b, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void MulF32(const float* a, const float* b, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
  for (; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
}

void ScaleF32(const float* x, float s, float* out, std::int64_t n) {
  const float32x4_t vs = vdupq_n_f32(s);
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), vs));
  }
  for (; i < n; ++i) {
    out[i] = x[i] * s;
  }
}

}  // namespace engine::kernels::neon
