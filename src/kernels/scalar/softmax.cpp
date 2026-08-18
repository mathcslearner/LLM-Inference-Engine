#include "kernels/internal/exp_common.h"
#include "kernels/internal/softmax_impl.h"

#include <algorithm>
#include <cstdint>

// Scalar softmax variant: the cpu::softmax arithmetic (row max, then exp and
// ascending fp32 sum, then normalize) with the exp_common.h polynomial in place
// of std::exp. Compiled on every platform (forced-scalar target); Class T vs
// the oracle within the §10 tolerance (the exp poly is ≤2 ulp of std::expf).

namespace engine::kernels::scalar {

void SoftmaxRows(const float* x, float* y, std::int64_t rows, std::int64_t n) {
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* x_row = x + (r * n);
    float* y_row = y + (r * n);

    // Row max in a single accumulator (a `-inf` entry never wins unless the
    // whole row is `-inf`, which yields NaN downstream — a caller error).
    float row_max = x_row[0];
    for (std::int64_t j = 1; j < n; ++j) {
      row_max = std::max(row_max, x_row[j]);
    }

    // exp(x - max) and its ascending fp32 sum. `x - max < kExpLo` (a masked
    // `-inf`) flushes to exactly 0 — the causal-mask contract.
    float sum = 0.0F;
    for (std::int64_t j = 0; j < n; ++j) {
      const float e = internal::ExpF32Scalar(x_row[j] - row_max);
      y_row[j] = e;
      sum += e;
    }
    const float inv_sum = 1.0F / sum;
    for (std::int64_t j = 0; j < n; ++j) {
      y_row[j] *= inv_sum;
    }
  }
}

}  // namespace engine::kernels::scalar
