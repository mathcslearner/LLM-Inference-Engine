#include "kernels/dispatch.h"

#include "kernels/probe.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

// SIMD dispatch tests (M3-T05; design: docs/design/cpu-backend.md §4, §9).
//
// This suite is registered with ctest twice (tests/unit/CMakeLists.txt,
// SCALAR_PASS): once as unit/ running the host's best ISA and once as
// unit-scalar/ with ENGINE_FORCE_ISA=scalar in the environment. The same
// sources assert correctly in both passes by reading the variable (§8.1: no
// #ifdef-forked expectations). The ENGINE_FORCE_ISA failure paths are
// tested through detail::ResolveIsa — SelectedIsa()'s process-wide
// memoization (and EXPECT_DEATH's fork inheriting it) makes them
// unreachable through the public entry; the happy path through the real
// environment variable is exactly what the unit-scalar/ pass exercises.

namespace engine::kernels {
namespace {

bool ForcedScalar() {
  const char* env = std::getenv("ENGINE_FORCE_ISA");
  return env != nullptr && std::string_view(env) == "scalar";
}

TEST(DispatchTest, BestHostIsaMatchesPlatform) {
#if defined(ENGINE_ARCH_ARM64)
  // NEON (ASIMD) is architecturally mandatory in AArch64 — no probing.
  EXPECT_EQ(detail::BestHostIsa(), Isa::kNeon);
#elif defined(ENGINE_ARCH_X86_64)
  // kAvx2 on any Haswell-2013+ host (every CI runner qualifies); kScalar
  // only on pre-AVX2 hardware or an OS not preserving YMM state.
  const Isa best = detail::BestHostIsa();
  EXPECT_TRUE(best == Isa::kAvx2 || best == Isa::kScalar) << IsaName(best);
#else
  EXPECT_EQ(detail::BestHostIsa(), Isa::kScalar);
#endif
}

TEST(DispatchTest, SelectedIsaHonorsForcedScalarEnvironment) {
  if (ForcedScalar()) {
    EXPECT_EQ(SelectedIsa(), Isa::kScalar);
  } else {
    EXPECT_EQ(SelectedIsa(), detail::BestHostIsa());
  }
}

TEST(DispatchTest, SelectedIsaIsStable) {
  EXPECT_EQ(SelectedIsa(), SelectedIsa());
}

TEST(DispatchTest, ProbeExecutesTheSelectedVariant) {
  // The probe's table populates every variant this build compiled, so the
  // executed variant must be exactly SelectedIsa(): the vector ISA in the
  // normal pass, scalar in the forced-scalar pass — §9's "the override
  // verifiably changes the selected variant".
  EXPECT_EQ(DispatchProbe(), SelectedIsa());
}

TEST(DispatchTest, IsaNamesMatchTheForceSpellings) {
  EXPECT_EQ(IsaName(Isa::kScalar), "scalar");
  EXPECT_EQ(IsaName(Isa::kNeon), "neon");
  EXPECT_EQ(IsaName(Isa::kAvx2), "avx2");
}

TEST(ResolveIsaTest, UnsetOrEmptyResolvesToBestHost) {
  EXPECT_EQ(detail::ResolveIsa(nullptr, Isa::kNeon), Isa::kNeon);
  EXPECT_EQ(detail::ResolveIsa(nullptr, Isa::kScalar), Isa::kScalar);
  EXPECT_EQ(detail::ResolveIsa("", Isa::kAvx2), Isa::kAvx2);
}

TEST(ResolveIsaTest, ScalarIsAllowedOnEveryHost) {
  EXPECT_EQ(detail::ResolveIsa("scalar", Isa::kScalar), Isa::kScalar);
  EXPECT_EQ(detail::ResolveIsa("scalar", Isa::kNeon), Isa::kScalar);
  EXPECT_EQ(detail::ResolveIsa("scalar", Isa::kAvx2), Isa::kScalar);
}

TEST(ResolveIsaTest, ForcingTheHostsOwnIsaIsAllowed) {
  EXPECT_EQ(detail::ResolveIsa("neon", Isa::kNeon), Isa::kNeon);
  EXPECT_EQ(detail::ResolveIsa("avx2", Isa::kAvx2), Isa::kAvx2);
}

TEST(ResolveIsaDeathTest, UnknownValueIsFatalWithActionableMessage) {
  EXPECT_DEATH(static_cast<void>(detail::ResolveIsa("avx512", Isa::kNeon)),
               "ENGINE_FORCE_ISA=\"avx512\" is not a supported value");
  // The message names the host's available ISAs.
  EXPECT_DEATH(static_cast<void>(detail::ResolveIsa("avx512", Isa::kNeon)),
               "this host supports: scalar, neon");
  // Values are case-sensitive: exactly scalar / neon / avx2 (§4.3).
  EXPECT_DEATH(static_cast<void>(detail::ResolveIsa("Scalar", Isa::kNeon)),
               "not a supported value");
}

TEST(ResolveIsaDeathTest, UnavailableIsaIsFatalWithActionableMessage) {
  EXPECT_DEATH(static_cast<void>(detail::ResolveIsa("neon", Isa::kAvx2)),
               "ENGINE_FORCE_ISA=\"neon\" is not executable on this host");
  EXPECT_DEATH(static_cast<void>(detail::ResolveIsa("avx2", Isa::kNeon)),
               "this host supports: scalar, neon");
  EXPECT_DEATH(static_cast<void>(detail::ResolveIsa("avx2", Isa::kScalar)),
               "this host supports: scalar");
}

int ScalarVariant() { return 1; }
int NeonVariant() { return 2; }
int Avx2Variant() { return 3; }
using IntFn = int (*)();

TEST(SelectTest, NullSlotsFallBackToScalarUnderAnySelection) {
  const KernelTable<IntFn> table = {.scalar = &ScalarVariant};
  EXPECT_EQ(Select(table, Isa::kScalar), &ScalarVariant);
  EXPECT_EQ(Select(table, Isa::kNeon), &ScalarVariant);
  EXPECT_EQ(Select(table, Isa::kAvx2), &ScalarVariant);
}

TEST(SelectTest, PopulatedSlotsAreSelected) {
  const KernelTable<IntFn> table = {
      .scalar = &ScalarVariant, .neon = &NeonVariant, .avx2 = &Avx2Variant};
  EXPECT_EQ(Select(table, Isa::kScalar), &ScalarVariant);
  EXPECT_EQ(Select(table, Isa::kNeon), &NeonVariant);
  EXPECT_EQ(Select(table, Isa::kAvx2), &Avx2Variant);
}

TEST(SelectTest, SingleArgumentOverloadUsesSelectedIsa) {
  const KernelTable<IntFn> table = {
      .scalar = &ScalarVariant, .neon = &NeonVariant, .avx2 = &Avx2Variant};
  EXPECT_EQ(Select(table), Select(table, SelectedIsa()));
}

TEST(SelectDeathTest, NullScalarSlotIsFatal) {
  const KernelTable<IntFn> table = {.scalar = nullptr};
  EXPECT_DEATH(static_cast<void>(Select(table, Isa::kScalar)),
               "no scalar variant");
}

}  // namespace
}  // namespace engine::kernels
