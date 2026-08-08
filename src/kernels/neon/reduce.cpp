#include "kernels/internal/reduce_common.h"
#include "kernels/internal/reduce_impl.h"

#include <arm_neon.h>

// NEON reduction variants (arm64 builds only). SumF32Chunk realizes the
// 16-lane convention (reduce.h) as 4×float32x4_t: accumulator k's lane l is
// spec lane 4k + l, so a full 16-element step routes element 4k + l into
// exactly the lane the scalar variant uses. The tail adds per element into
// the spilled lane array — same lanes, same order — and the 16→1 fold is
// the shared internal::FoldLanes16, literally the scalar code. Plain adds
// only; no FMA in reductions (design §6.3).

namespace engine::kernels::neon {

float SumF32Chunk(const float* x, std::int64_t n) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);  // +0.0F init is part of the spec.
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);
  std::int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = vaddq_f32(acc0, vld1q_f32(x + i));
    acc1 = vaddq_f32(acc1, vld1q_f32(x + i + 4));
    acc2 = vaddq_f32(acc2, vld1q_f32(x + i + 8));
    acc3 = vaddq_f32(acc3, vld1q_f32(x + i + 12));
  }
  float lanes[16];
  vst1q_f32(lanes, acc0);
  vst1q_f32(lanes + 4, acc1);
  vst1q_f32(lanes + 8, acc2);
  vst1q_f32(lanes + 12, acc3);
  for (; i < n; ++i) {
    lanes[i % 16] += x[i];
  }
  return internal::FoldLanes16(lanes);
}

float MaxF32Chunk(const float* x, std::int64_t n) {
  // FMAX natively implements the total order this kernel is specified
  // against: -0.0 compares less than +0.0 (reduce.h). Total-order max is
  // associative, so lane structure and fold order are free.
  std::int64_t i = 0;
  float m = x[0];
  if (n >= 4) {
    float32x4_t acc = vld1q_f32(x);
    for (i = 4; i + 4 <= n; i += 4) {
      acc = vmaxq_f32(acc, vld1q_f32(x + i));
    }
    float lanes[4];
    vst1q_f32(lanes, acc);
    m = internal::TotalOrderMax(internal::TotalOrderMax(lanes[0], lanes[1]),
                                internal::TotalOrderMax(lanes[2], lanes[3]));
  } else {
    i = 1;
  }
  for (; i < n; ++i) {
    m = internal::TotalOrderMax(m, x[i]);
  }
  return m;
}

}  // namespace engine::kernels::neon
