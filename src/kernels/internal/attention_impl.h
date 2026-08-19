#pragma once

#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"
#include "kernels/internal/paged_attention_common.h"

#include <cstdint>

// Per-ISA variants of the prefill (M6-T04) and decode (M6-T05) attention
// kernels (design: optimized-cpu-execution.md §8, §10). Internal: included only
// by attention.cpp, the per-ISA TUs, and tests. Each `PrefillUnits` /
// `DecodeUnits` is a single-threaded chunk body over a contiguous run of units
// (prefill: (head, query-block) pairs; decode: kv heads); threading over units
// lives in the public entry (attention.cpp). Each just instantiates the
// matching internal::*UnitsImpl with its ISA `Ops` (the same four primitives
// serve both kernels).

namespace engine::kernels {

namespace scalar {
void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end);
void DecodeUnits(const internal::DecodeArgs& a, std::int64_t unit_begin,
                 std::int64_t unit_end);
void PagedDecodeUnits(const internal::PagedDecodeArgs& a,
                      std::int64_t unit_begin, std::int64_t unit_end);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end);
void DecodeUnits(const internal::DecodeArgs& a, std::int64_t unit_begin,
                 std::int64_t unit_end);
void PagedDecodeUnits(const internal::PagedDecodeArgs& a,
                      std::int64_t unit_begin, std::int64_t unit_end);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma; must only run when dispatch selected
// kAvx2.
void PrefillUnits(const internal::PrefillArgs& a, std::int64_t unit_begin,
                  std::int64_t unit_end);
void DecodeUnits(const internal::DecodeArgs& a, std::int64_t unit_begin,
                 std::int64_t unit_end);
void PagedDecodeUnits(const internal::PagedDecodeArgs& a,
                      std::int64_t unit_begin, std::int64_t unit_end);
}  // namespace avx2
#endif

namespace detail {

// Test seam: the PrefillUnits variant Select would return for `isa`.
using PrefillUnitsFn = void (*)(const internal::PrefillArgs&, std::int64_t,
                                std::int64_t);
[[nodiscard]] PrefillUnitsFn PrefillAttentionVariant(Isa isa);

// The whole varlen prefill kernel minus dispatch + threading (M9-T06 test
// seam): runs sequence-major units [unit_begin, unit_end) by walking the
// sequences of `a` and invoking the per-ISA PrefillUnits variant `fn` on a
// PrefillArgs synthesized per sequence. Sequence b owns H·ceil(T_b/kAttnQb)
// contiguous units; the per-sequence call is byte-for-byte what a standalone
// PrefillAttentionF32(seq_b) would run, so thread/chunk invariance and
// per-sequence bit-identity hold by construction. No ISA content here — the
// arithmetic lives entirely in `fn`.
void PrefillVarlenUnits(const internal::PrefillVarlenArgs& a, PrefillUnitsFn fn,
                        std::int64_t unit_begin, std::int64_t unit_end);

// Test seam: the DecodeUnits variant Select would return for `isa`.
using DecodeUnitsFn = void (*)(const internal::DecodeArgs&, std::int64_t,
                               std::int64_t);
[[nodiscard]] DecodeUnitsFn DecodeAttentionVariant(Isa isa);

// Test seam: the PagedDecodeUnits variant Select would return for `isa`.
using PagedDecodeUnitsFn = void (*)(const internal::PagedDecodeArgs&,
                                    std::int64_t, std::int64_t);
[[nodiscard]] PagedDecodeUnitsFn PagedDecodeAttentionVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
