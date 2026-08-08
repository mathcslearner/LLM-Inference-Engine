#include "kernels/internal/elementwise_impl.h"

// Scalar elementwise variants: compiled on every platform with no arch
// flags — the production fallback, the forced-scalar test target, and (until
// the M5 cpu backend) the reference the vector variants must match
// bit-exactly (design §2.1, §6.3 Class E).

namespace engine::kernels::scalar {

void AddF32(const float* a, const float* b, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void MulF32(const float* a, const float* b, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
}

void ScaleF32(const float* x, float s, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = x[i] * s;
  }
}

}  // namespace engine::kernels::scalar
