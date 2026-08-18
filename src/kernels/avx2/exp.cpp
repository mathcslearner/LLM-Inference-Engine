#include "kernels/internal/avx2_exp.h"
#include "kernels/internal/exp_common.h"
#include "kernels/internal/exp_impl.h"

#include <immintrin.h>

// AVX2 vector-exp variant (x86-64 builds only; -mavx2 -mfma per-source). Eight
// lanes via internal::Exp; scalar tail via the shared spec so body/tail agree.

namespace engine::kernels::avx2 {

void ExpF32(const float* in, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, Exp(_mm256_loadu_ps(in + i)));
  }
  for (; i < n; ++i) {
    out[i] = internal::ExpF32Scalar(in[i]);
  }
}

}  // namespace engine::kernels::avx2
