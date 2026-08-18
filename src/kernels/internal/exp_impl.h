#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Array-form vector `expf` variants (M6-T03; design:
// optimized-cpu-execution.md §10). `exp` is not a public dispatched kernel —
// it is an implementation detail of the softmax/SiLU kernels, which call the
// header-only lane helpers (internal/{neon,avx2}_exp.h) directly. These array
// entries exist so the dedicated ulp sweep (`vector_exp_test`) can exercise the
// polynomial on every ISA the host builds, independent of the kernels that use
// it. Each variant maps out[i] = exp(in[i]); n >= 0, any alignment.

namespace engine::kernels {

namespace scalar {
void ExpF32(const float* in, float* out, std::int64_t n);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void ExpF32(const float* in, float* out, std::int64_t n);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void ExpF32(const float* in, float* out, std::int64_t n);
}  // namespace avx2
#endif

namespace detail {

// Test seam: the ExpF32 variant for `isa` (scalar fallback when the vector slot
// is absent), so the sweep can request each ISA on one host — mirrors
// internal/elementwise_impl.h's variant accessors.
using ExpF32Fn = void (*)(const float*, float*, std::int64_t);
[[nodiscard]] ExpF32Fn ExpF32Variant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
