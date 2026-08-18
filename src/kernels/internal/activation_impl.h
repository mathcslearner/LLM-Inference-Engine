#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Per-ISA variants of the SwiGLU-combine kernel (M6-T03; design:
// optimized-cpu-execution.md §10). Internal: included only by activation.cpp,
// the per-ISA TUs, and tests. Variants are single-threaded chunk bodies over a
// contiguous run of n elements — threading (flat over elements) lives in the
// public entry (activation.cpp). `y` may alias gate or up.

namespace engine::kernels {

namespace scalar {
void SiluMul(const float* gate, const float* up, float* y, std::int64_t n);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void SiluMul(const float* gate, const float* up, float* y, std::int64_t n);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void SiluMul(const float* gate, const float* up, float* y, std::int64_t n);
}  // namespace avx2
#endif

namespace detail {

// Test seam (M3 audit): the SiluMul variant Select would return for `isa`.
using SiluMulFn = void (*)(const float*, const float*, float*, std::int64_t);
[[nodiscard]] SiluMulFn SiluMulVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
