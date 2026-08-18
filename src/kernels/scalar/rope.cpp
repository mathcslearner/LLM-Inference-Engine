#include "kernels/internal/rope_impl.h"

#include <cstddef>
#include <cstdint>

// Scalar RoPE-apply variant: the cpu::rope_apply half-rotation, one element at
// a time. Compiled on every platform (forced-scalar target); Class T vs the
// oracle within the §10 tolerance (the tables are shared fp32).

namespace engine::kernels::scalar {

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
      for (std::int64_t j = 0; j < half; ++j) {
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

}  // namespace engine::kernels::scalar
