#include "kernels/dispatch.h"

#include "core/check.h"

#include <cstdlib>
#include <string_view>

#if defined(ENGINE_ARCH_X86_64)
#include <cpuid.h>

#include <cstdint>
#endif

namespace engine::kernels {

namespace {

// "scalar, neon" / "scalar, avx2" / "scalar" — the available-ISA list for
// the fatal messages below. Scalar is available on every host; at most one
// vector ISA can be (the build compiles per-ISA TUs for one arch, §4.4).
constexpr std::string_view HostIsaList(Isa best_host) {
  switch (best_host) {
    case Isa::kScalar:
      return "scalar";
    case Isa::kNeon:
      return "scalar, neon";
    case Isa::kAvx2:
      return "scalar, avx2";
  }
  CHECK(false, "invalid Isa value {}", static_cast<int>(best_host));
}

#if defined(ENGINE_ARCH_X86_64)
// XCR0 via xgetbv, as inline asm so this TU needs no -mxsave flag — it is
// compiled without arch flags like every non-per-ISA TU (§4.4 rule 1). Only
// called after cpuid confirmed OSXSAVE, so the instruction is executable.
std::uint64_t Xgetbv0() {
  // Not const: both are written by the asm's output operands below, which
  // misc-const-correctness does not model (and const would not compile).
  // NOLINTBEGIN(misc-const-correctness)
  std::uint32_t eax = 0;
  std::uint32_t edx = 0;
  // NOLINTEND(misc-const-correctness)
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0U));
  return (static_cast<std::uint64_t>(edx) << 32U) | eax;
}
#endif

}  // namespace

namespace detail {

Isa BestHostIsa() {
#if defined(ENGINE_ARCH_ARM64)
  return Isa::kNeon;
#elif defined(ENGINE_ARCH_X86_64)
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if (__get_cpuid(1U, &eax, &ebx, &ecx, &edx) == 0) {
    return Isa::kScalar;
  }
  // Leaf 1 ECX: FMA, F16C, AVX (the YMM registers AVX2 operates on), and
  // OSXSAVE (xgetbv is executable and the OS uses XSAVE at all).
  const unsigned int required_ecx = static_cast<unsigned int>(bit_FMA) |
                                    static_cast<unsigned int>(bit_F16C) |
                                    static_cast<unsigned int>(bit_AVX) |
                                    static_cast<unsigned int>(bit_OSXSAVE);
  if ((ecx & required_ecx) != required_ecx) {
    return Isa::kScalar;
  }
  if (__get_cpuid_count(7U, 0U, &eax, &ebx, &ecx, &edx) == 0 ||
      (ebx & static_cast<unsigned int>(bit_AVX2)) == 0U) {
    return Isa::kScalar;
  }
  // XCR0 bits 1 (SSE/XMM) and 2 (AVX/YMM): the OS preserves the full YMM
  // state across context switches.
  if ((Xgetbv0() & 0x6U) != 0x6U) {
    return Isa::kScalar;
  }
  return Isa::kAvx2;
#else
  return Isa::kScalar;
#endif
}

Isa ResolveIsa(const char* force_value, Isa best_host) {
  if (force_value == nullptr || *force_value == '\0') {
    return best_host;
  }
  const std::string_view value(force_value);
  Isa forced = Isa::kScalar;
  if (value == IsaName(Isa::kScalar)) {
    forced = Isa::kScalar;
  } else if (value == IsaName(Isa::kNeon)) {
    forced = Isa::kNeon;
  } else if (value == IsaName(Isa::kAvx2)) {
    forced = Isa::kAvx2;
  } else {
    CHECK(false,
          "ENGINE_FORCE_ISA=\"{}\" is not a supported value (valid: scalar, "
          "neon, avx2; this host supports: {})",
          value, HostIsaList(best_host));
  }
  // Scalar is executable everywhere; a vector ISA only on the host that has
  // it. Fatal-not-fallback is deliberate (§4.3): a test suite whose forced
  // pass silently ran something else would claim coverage it does not have.
  CHECK(forced == Isa::kScalar || forced == best_host,
        "ENGINE_FORCE_ISA=\"{}\" is not executable on this host (this host "
        "supports: {})",
        value, HostIsaList(best_host));
  return forced;
}

}  // namespace detail

Isa SelectedIsa() {
  static const Isa isa = detail::ResolveIsa(std::getenv("ENGINE_FORCE_ISA"),
                                            detail::BestHostIsa());
  return isa;
}

}  // namespace engine::kernels
