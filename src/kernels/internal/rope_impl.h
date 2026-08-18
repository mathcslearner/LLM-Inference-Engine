#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Per-ISA variants of the RoPE-apply kernel (M6-T03; design:
// optimized-cpu-execution.md §10). Internal: included only by rope.cpp, the
// per-ISA TUs, and tests. Each variant rotates a contiguous run of tokens
// [begin_t, begin_t+t_count) in place; the caller passes the base x pointer and
// the token offset so positions[]/x line up. Threading (over tokens) lives in
// the public entry (rope.cpp).

namespace engine::kernels {

namespace scalar {
void RopeRows(float* x, std::int64_t t_count, std::int64_t hx, std::int64_t d,
              const std::int32_t* positions, const float* cos,
              const float* sin);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void RopeRows(float* x, std::int64_t t_count, std::int64_t hx, std::int64_t d,
              const std::int32_t* positions, const float* cos,
              const float* sin);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void RopeRows(float* x, std::int64_t t_count, std::int64_t hx, std::int64_t d,
              const std::int32_t* positions, const float* cos,
              const float* sin);
}  // namespace avx2
#endif

namespace detail {

// Test seam (M3 audit): the RopeRows variant Select would return for `isa`.
using RopeRowsFn = void (*)(float*, std::int64_t, std::int64_t, std::int64_t,
                            const std::int32_t*, const float*, const float*);
[[nodiscard]] RopeRowsFn RopeRowsVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
