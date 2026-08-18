#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Per-ISA variants of the RMSNorm kernel (M6-T03; design:
// optimized-cpu-execution.md §10). Internal: included only by norm.cpp, the
// per-ISA TUs, and tests. Each variant is a single-threaded chunk body over a
// contiguous run of rows [rows, e] — threading (over rows) lives in the public
// entry (norm.cpp). `y` may alias `x`.

namespace engine::kernels {

namespace scalar {
void RmsNormRows(const float* x, const float* weight, float eps,
                 std::int64_t rows, std::int64_t e, float* y);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void RmsNormRows(const float* x, const float* weight, float eps,
                 std::int64_t rows, std::int64_t e, float* y);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void RmsNormRows(const float* x, const float* weight, float eps,
                 std::int64_t rows, std::int64_t e, float* y);
}  // namespace avx2
#endif

namespace detail {

// Test seam (M3 audit): the RmsNormRows variant Select would return for `isa`.
using RmsNormRowsFn = void (*)(const float*, const float*, float, std::int64_t,
                               std::int64_t, float*);
[[nodiscard]] RmsNormRowsFn RmsNormRowsVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
