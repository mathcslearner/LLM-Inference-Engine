#include "kernels/internal/norm_impl.h"

#include <cmath>
#include <cstdint>

// Scalar RMSNorm variant: the cpu::rmsnorm arithmetic with an fp32 weight (the
// optimized backend pre-converts norm scales, §4). Ascending single-accumulator
// sum of squares, an exact 1/sqrtf, then the HF-order scale. Compiled on every
// platform (forced-scalar target); Class T vs the oracle within the §10
// tolerance, exact 1/sqrtf (never rsqrte, §10).

namespace engine::kernels::scalar {

void RmsNormRows(const float* x, const float* weight, float eps,
                 std::int64_t rows, std::int64_t e, float* y) {
  const auto e_f = static_cast<float>(e);
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x_row = x + (r * e);
    float* y_row = y + (r * e);

    float sum_sq = 0.0F;
    for (std::int64_t j = 0; j < e; ++j) {
      const float v = x_row[j];
      sum_sq += v * v;
    }
    const float inv_rms = 1.0F / std::sqrt((sum_sq / e_f) + eps);
    for (std::int64_t j = 0; j < e; ++j) {
      y_row[j] = x_row[j] * inv_rms * weight[j];
    }
  }
}

}  // namespace engine::kernels::scalar
