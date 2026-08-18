#include "kernels/internal/neon_exp.h"
#include "kernels/internal/softmax_impl.h"

#include <arm_neon.h>

#include <cstdint>

// NEON softmax variant (arm64 builds only). Row max via a vector max reduction,
// exp via internal::Exp (the shared polynomial), row sum via a 4-lane
// accumulator folded once, then a vector normalize. Class T vs the scalar
// variant and the oracle (vector exp + horizontal reductions differ in
// rounding); bit-identical across thread counts (each row wholly in one call).

namespace engine::kernels::neon {

namespace {

// Horizontal max of a row, seeded at x[0]; a `-inf` lane never wins unless the
// whole row is `-inf`. Order-insensitive (max is associative on finite/±inf).
[[nodiscard]] float RowMax(const float* x, std::int64_t n) {
  std::int64_t i = 0;
  float m = x[0];
  if (n >= 4) {
    float32x4_t acc = vld1q_f32(x);
    for (i = 4; i + 4 <= n; i += 4) {
      acc = vmaxq_f32(acc, vld1q_f32(x + i));
    }
    m = vmaxvq_f32(acc);
  } else {
    i = 1;
  }
  for (; i < n; ++i) {
    m = x[i] > m ? x[i] : m;
  }
  return m;
}

}  // namespace

void SoftmaxRows(const float* x, float* y, std::int64_t rows, std::int64_t n) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x_row = x + (r * n);
    float* y_row = y + (r * n);

    const float row_max = RowMax(x_row, n);
    const float32x4_t vmax = vdupq_n_f32(row_max);

    // exp(x - max) → y, accumulating the sum in 4 lanes + a scalar tail.
    float32x4_t vsum = vdupq_n_f32(0.0F);
    float sum = 0.0F;
    std::int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      const float32x4_t e = Exp(vsubq_f32(vld1q_f32(x_row + j), vmax));
      vst1q_f32(y_row + j, e);
      vsum = vaddq_f32(vsum, e);
    }
    sum = vaddvq_f32(vsum);
    for (; j < n; ++j) {
      const float e = internal::ExpF32Scalar(x_row[j] - row_max);
      y_row[j] = e;
      sum += e;
    }

    const float32x4_t vinv = vdupq_n_f32(1.0F / sum);
    j = 0;
    for (; j + 4 <= n; j += 4) {
      vst1q_f32(y_row + j, vmulq_f32(vld1q_f32(y_row + j), vinv));
    }
    const float inv = 1.0F / sum;
    for (; j < n; ++j) {
      y_row[j] *= inv;
    }
  }
}

}  // namespace engine::kernels::neon
