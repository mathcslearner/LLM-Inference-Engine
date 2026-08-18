#include "kernels/internal/exp_common.h"
#include "kernels/internal/exp_impl.h"

// Scalar vector-exp variant: the exp_common.h spec, one element at a time.
// Compiled on every platform (the portability floor and forced-scalar target);
// the sweep test proves it ≤2 ulp vs std::expf on every host.

namespace engine::kernels::scalar {

void ExpF32(const float* in, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = internal::ExpF32Scalar(in[i]);
  }
}

}  // namespace engine::kernels::scalar
