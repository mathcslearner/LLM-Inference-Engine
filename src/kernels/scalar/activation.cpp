#include "kernels/internal/activation_impl.h"
#include "kernels/internal/exp_common.h"

#include <cstdint>

// Scalar SwiGLU-combine variant: silu(gate) * up with the exp_common.h
// polynomial in place of std::exp. Compiled on every platform (forced-scalar
// target); Class T vs the oracle within the §10 tolerance.

namespace engine::kernels::scalar {

void SiluMul(const float* gate, const float* up, float* y, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    const float g = gate[i];
    // silu(g) = g / (1 + exp(-g)). exp(-g) saturates for very negative g
    // (silu → 0) and the exp poly clamps very positive g (silu → g).
    const float silu = g / (1.0F + internal::ExpF32Scalar(-g));
    y[i] = silu * up[i];
  }
}

}  // namespace engine::kernels::scalar
