#include "kernels/internal/probe_impl.h"

#include <immintrin.h>

#include <cstdint>

// AVX2 probe variant (x86-64 builds only; compiled with per-source
// -mavx2 -mfma -mf16c, design §4.4 rule 1 — those flags never appear
// target-wide). The answer is computed with a real AVX2 intrinsic so the
// per-TU flag machinery is genuinely exercised: without -mavx2 this TU does
// not compile.

namespace engine::kernels::avx2 {

Isa Probe() {
  const __m256i lanes =
      _mm256_set1_epi32(static_cast<std::int32_t>(Isa::kAvx2));
  return static_cast<Isa>(_mm256_extract_epi32(lanes, 0));
}

}  // namespace engine::kernels::avx2
