#pragma once

#include "core/check.h"

#include <cstdint>
#include <string_view>

// SIMD dispatch infrastructure (M3-T05; design: docs/design/cpu-backend.md
// §4).
//
// Every production kernel ships per-ISA variants (scalar always; NEON on
// arm64, AVX2 on x86-64) in per-ISA translation units, bound behind one
// function pointer per kernel in a KernelTable. One process, one
// SelectedIsa(), resolved exactly once — there is no per-kernel ISA choice
// and no mixed-ISA execution (design §4.1). This header is intrinsic-free
// and includable everywhere; intrinsic headers appear only inside per-ISA
// TUs (design §4.4 rule 2).

namespace engine::kernels {

// The instruction sets kernels ship variants for. Scalar is compiled on
// every platform — it is the fallback, the forced-scalar test target, and
// the portability floor (design §4.4 rule 3).
enum class Isa : std::uint8_t { kScalar, kNeon, kAvx2 };

// Canonical lowercase name ("scalar", "neon", "avx2") — the exact spellings
// ENGINE_FORCE_ISA accepts (case-sensitive, design §4.3).
[[nodiscard]] constexpr std::string_view IsaName(Isa isa) {
  switch (isa) {
    case Isa::kScalar:
      return "scalar";
    case Isa::kNeon:
      return "neon";
    case Isa::kAvx2:
      return "avx2";
  }
  CHECK(false, "invalid Isa value {}", static_cast<int>(isa));
}

// The ISA every dispatched kernel uses, resolved exactly once per process
// and memoized: ENGINE_FORCE_ISA if set (unknown or host-unavailable values
// are fatal with an actionable message at the first dispatch — resolution
// is lazy, design §4.3), else the best ISA the host supports
// (detail::BestHostIsa).
[[nodiscard]] Isa SelectedIsa();

// One table per kernel: a function pointer per ISA variant. `scalar` is
// never null — every kernel ships a scalar variant. A vector slot is null
// when the variant's TU is absent from the build (wrong arch) or the
// variant simply hasn't been written yet (a kernel may gain vector variants
// in a later ticket than its scalar one).
template <typename Fn>
struct KernelTable {
  Fn scalar;
  Fn neon = nullptr;
  Fn avx2 = nullptr;
};

// The entry `isa` selects from `table`, falling back to `scalar` when that
// slot is null — the fallback is per-kernel, silent, and safe, because the
// scalar variant is always correct (design §4.2). The explicit-ISA overload
// exists so tests can exercise every selection on one host; production
// entry points use the SelectedIsa() overload below.
template <typename Fn>
[[nodiscard]] Fn Select(const KernelTable<Fn>& table, Isa isa) {
  CHECK(table.scalar != nullptr,
        "KernelTable has no scalar variant; the scalar slot is never null");
  switch (isa) {
    case Isa::kScalar:
      break;
    case Isa::kNeon:
      if (table.neon != nullptr) {
        return table.neon;
      }
      break;
    case Isa::kAvx2:
      if (table.avx2 != nullptr) {
        return table.avx2;
      }
      break;
  }
  return table.scalar;
}

// The entry for SelectedIsa() — what public kernel entry points call,
// caching the result in a function-local static so dispatch cost after the
// first call is one indirect call (design §4.2):
//
//   void AddF32(const float* a, const float* b, float* out, std::int64_t n) {
//     static const auto fn = Select(kAddF32Table);
//     fn(a, b, out, n);
//   }
template <typename Fn>
[[nodiscard]] Fn Select(const KernelTable<Fn>& table) {
  return Select(table, SelectedIsa());
}

namespace detail {

// Best ISA the host can execute, ignoring ENGINE_FORCE_ISA: kNeon on arm64
// (ASIMD is architecturally mandatory in AArch64 — no probing), kAvx2 on
// x86-64 iff cpuid reports AVX2 + FMA + F16C (all Haswell-2013+; F16C is
// required by the M3-T06 conversion kernels, so its availability is folded
// into the kAvx2 gate, design §4.1) and OSXSAVE/xgetbv confirm the OS
// preserves YMM state, else kScalar. Exposed for tests.
[[nodiscard]] Isa BestHostIsa();

// The resolution behind SelectedIsa(): `force_value` is the raw
// ENGINE_FORCE_ISA (nullptr or empty = unset, matching ENGINE_NUM_THREADS).
// An unknown value or an ISA the host cannot execute is fatal with an
// actionable message — the variable is developer/test configuration
// resolved once at startup, and silently falling back would let a
// forced-scalar test pass claim coverage it does not have (design §4.3).
// Exposed for the death tests: SelectedIsa()'s process-wide memoization
// makes the failure paths unreachable through the public entry once any
// kernel has dispatched.
[[nodiscard]] Isa ResolveIsa(const char* force_value, Isa best_host);

}  // namespace detail

}  // namespace engine::kernels
