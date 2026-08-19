#pragma once

#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"

#include <cstdint>

// Per-ISA variants of the prefill-attention kernel (M6-T04; design:
// optimized-cpu-execution.md §8, §10). Internal: included only by
// attention.cpp, the per-ISA TUs, and tests. Each `PrefillUnits` is a
// single-threaded chunk body over a contiguous run of (head, query-block)
// units; threading (over units) lives in the public entry (attention.cpp).
// Each just instantiates internal::PrefillUnitsImpl with its ISA `Ops`.

namespace engine::kernels {

namespace scalar {
void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end);
}  // namespace avx2
#endif

namespace detail {

// Test seam: the PrefillUnits variant Select would return for `isa`.
using PrefillUnitsFn = void (*)(const internal::PrefillArgs&, std::int64_t,
                                std::int64_t);
[[nodiscard]] PrefillUnitsFn PrefillAttentionVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
