#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Per-ISA variants of the softmax kernel (M6-T03; design:
// optimized-cpu-execution.md §10). Internal: included only by softmax.cpp, the
// per-ISA TUs, and tests. Each variant is a single-threaded chunk body over a
// contiguous run of rows [rows, n] — threading (over rows) lives in the public
// entry (softmax.cpp). `y` may alias `x`.

namespace engine::kernels {

namespace scalar {
void SoftmaxRows(const float* x, float* y, std::int64_t rows, std::int64_t n);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void SoftmaxRows(const float* x, float* y, std::int64_t rows, std::int64_t n);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void SoftmaxRows(const float* x, float* y, std::int64_t rows, std::int64_t n);
}  // namespace avx2
#endif

namespace detail {

// Test seam (M3 audit): the SoftmaxRows variant Select would return for `isa`.
using SoftmaxRowsFn = void (*)(const float*, float*, std::int64_t,
                               std::int64_t);
[[nodiscard]] SoftmaxRowsFn SoftmaxRowsVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
