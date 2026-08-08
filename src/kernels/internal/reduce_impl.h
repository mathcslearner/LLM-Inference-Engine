#pragma once

#include <cstdint>

// Per-ISA variants of the reduction kernels (M3-T06; design: cpu-backend.md
// §4.2, §6.3). Internal: included only by reduce.cpp, the per-ISA TUs, and
// tests. Variants compute one chunk's partial — the cross-chunk fold is
// parallel_reduce's fixed pairwise tree, owned by the public entry points.
//
// SumF32Chunk implements the fixed 16-lane convention exactly (reduce.h):
// +0.0f-initialized lanes, element at offset k into lane k mod 16, plain
// adds, fixed halving fold. MaxF32Chunk (n >= 1) uses the total-order max
// with -0.0f < +0.0f (internal/reduce_common.h).

namespace engine::kernels {

namespace scalar {
float SumF32Chunk(const float* x, std::int64_t n);
float MaxF32Chunk(const float* x, std::int64_t n);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
float SumF32Chunk(const float* x, std::int64_t n);
float MaxF32Chunk(const float* x, std::int64_t n);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma -mf16c; must only run when dispatch
// selected kAvx2.
float SumF32Chunk(const float* x, std::int64_t n);
float MaxF32Chunk(const float* x, std::int64_t n);
}  // namespace avx2
#endif

}  // namespace engine::kernels
