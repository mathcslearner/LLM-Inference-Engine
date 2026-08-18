#include "kernels/internal/exp_common.h"
#include "kernels/internal/exp_impl.h"
#include "kernels/internal/neon_exp.h"

#include <arm_neon.h>

// NEON vector-exp variant (arm64 builds only). Four lanes via internal::Exp;
// scalar tail via the shared spec so body/tail agree.

namespace engine::kernels::neon {

void ExpF32(const float* in, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    vst1q_f32(out + i, Exp(vld1q_f32(in + i)));
  }
  for (; i < n; ++i) {
    out[i] = internal::ExpF32Scalar(in[i]);
  }
}

}  // namespace engine::kernels::neon
