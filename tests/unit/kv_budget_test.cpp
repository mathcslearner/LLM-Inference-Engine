#include "kvcache/kv_budget.h"

#include "core/status.h"
#include "core/sysinfo.h"

#include <gtest/gtest.h>

#include <cstdint>

// KV-budget resolution tests (M8-T07; design: docs/design/paged-kv-cache.md
// §5). Pure arithmetic + string parsing — no model, no kernels, no SCALAR_PASS.
// Also exercises core::host_memory_bytes() (the fraction path's host-RAM
// helper, §5.3), which kv_budget consumes via the driver.

namespace {

using engine::core::IsFailedPrecondition;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::kvcache::BlocksForTokens;
using engine::kvcache::KvBudgetInputs;
using engine::kvcache::KvCacheMemorySpec;
using engine::kvcache::ParseKvCacheMemory;
using engine::kvcache::ResolveKvBudgetBytes;

// --- ParseKvCacheMemory: absolute spellings ---

TEST(KvBudgetTest, ParsesBareByteCount) {
  const auto spec = ParseKvCacheMemory("8000000000");
  ASSERT_TRUE(spec.ok()) << spec.status().ToString();
  EXPECT_EQ(spec->kind, KvCacheMemorySpec::Kind::kAbsoluteBytes);
  EXPECT_EQ(spec->bytes, 8000000000LL);
}

TEST(KvBudgetTest, ParsesBinaryUnits) {
  EXPECT_EQ(ParseKvCacheMemory("2GiB")->bytes, 2LL * 1024 * 1024 * 1024);
  EXPECT_EQ(ParseKvCacheMemory("1500MiB")->bytes, 1500LL * 1024 * 1024);
  EXPECT_EQ(ParseKvCacheMemory("512KiB")->bytes, 512LL * 1024);
  EXPECT_EQ(ParseKvCacheMemory("64B")->bytes, 64);
}

TEST(KvBudgetTest, ParsesDecimalUnits) {
  EXPECT_EQ(ParseKvCacheMemory("2GB")->bytes, 2LL * 1000 * 1000 * 1000);
  EXPECT_EQ(ParseKvCacheMemory("1500MB")->bytes, 1500LL * 1000 * 1000);
  EXPECT_EQ(ParseKvCacheMemory("512KB")->bytes, 512LL * 1000);
}

TEST(KvBudgetTest, ParsesFractionalUnitValue) {
  // "1.5GiB" → 1.5 · 2^30.
  EXPECT_EQ(ParseKvCacheMemory("1.5GiB")->bytes,
            static_cast<std::int64_t>(1.5 * 1024 * 1024 * 1024));
}

TEST(KvBudgetTest, UnitsAreCaseInsensitiveAndTrimmed) {
  EXPECT_EQ(ParseKvCacheMemory("  2gib ")->bytes, 2LL * 1024 * 1024 * 1024);
}

// --- ParseKvCacheMemory: fractions ---

TEST(KvBudgetTest, ParsesFraction) {
  const auto spec = ParseKvCacheMemory("0.6");
  ASSERT_TRUE(spec.ok());
  EXPECT_EQ(spec->kind, KvCacheMemorySpec::Kind::kFraction);
  EXPECT_DOUBLE_EQ(spec->fraction, 0.6);
}

TEST(KvBudgetTest, FractionOfExactlyOneIsAllowed) {
  const auto spec = ParseKvCacheMemory("1.0");
  ASSERT_TRUE(spec.ok());
  EXPECT_EQ(spec->kind, KvCacheMemorySpec::Kind::kFraction);
  EXPECT_DOUBLE_EQ(spec->fraction, 1.0);
}

// --- ParseKvCacheMemory: rejections ---

TEST(KvBudgetTest, RejectsEmpty) {
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("").status()));
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("   ").status()));
}

TEST(KvBudgetTest, RejectsGarbage) {
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("abc").status()));
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("2GiBB").status()));
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("1.2.3").status()));
}

TEST(KvBudgetTest, RejectsOutOfRangeFraction) {
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("1.5").status()));
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("0.0").status()));
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("-0.5").status()));
}

TEST(KvBudgetTest, RejectsNonPositiveByteCount) {
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("0").status()));
  EXPECT_TRUE(IsInvalidArgument(ParseKvCacheMemory("0B").status()));
}

// --- ResolveKvBudgetBytes ---

TEST(KvBudgetTest, AbsolutePassesThrough) {
  const KvCacheMemorySpec spec{.kind = KvCacheMemorySpec::Kind::kAbsoluteBytes,
                               .bytes = 4096,
                               .fraction = 0.0};
  // Absolute ignores host-RAM / weights / workspace entirely.
  const auto budget =
      ResolveKvBudgetBytes(spec, KvBudgetInputs{.host_ram_bytes = 0,
                                                .weights_bytes = 999,
                                                .workspace_bytes = 9});
  ASSERT_TRUE(budget.ok());
  EXPECT_EQ(*budget, 4096);
}

TEST(KvBudgetTest, FractionSubtractsWeightsAndWorkspace) {
  const KvCacheMemorySpec spec{
      .kind = KvCacheMemorySpec::Kind::kFraction, .bytes = 0, .fraction = 0.5};
  const auto budget =
      ResolveKvBudgetBytes(spec, KvBudgetInputs{.host_ram_bytes = 1000,
                                                .weights_bytes = 200,
                                                .workspace_bytes = 50});
  ASSERT_TRUE(budget.ok());
  // 0.5·1000 − 200 − 50 = 250.
  EXPECT_EQ(*budget, 250);
}

TEST(KvBudgetTest, FractionWithUnknownHostRamIsFailedPrecondition) {
  const KvCacheMemorySpec spec{
      .kind = KvCacheMemorySpec::Kind::kFraction, .bytes = 0, .fraction = 0.5};
  const auto budget =
      ResolveKvBudgetBytes(spec, KvBudgetInputs{.host_ram_bytes = 0});
  EXPECT_TRUE(IsFailedPrecondition(budget.status()));
}

TEST(KvBudgetTest, FractionExhaustedNamesTheThreeTerms) {
  const KvCacheMemorySpec spec{
      .kind = KvCacheMemorySpec::Kind::kFraction, .bytes = 0, .fraction = 0.5};
  const auto budget =
      ResolveKvBudgetBytes(spec, KvBudgetInputs{.host_ram_bytes = 1000,
                                                .weights_bytes = 400,
                                                .workspace_bytes = 200});
  // 500 − 400 − 200 < 0.
  ASSERT_TRUE(IsResourceExhausted(budget.status()));
  const std::string msg = budget.status().ToString();
  EXPECT_NE(msg.find("400"), std::string::npos);
  EXPECT_NE(msg.find("200"), std::string::npos);
  EXPECT_NE(msg.find("1000"), std::string::npos);
}

// --- BlocksForTokens ---

TEST(KvBudgetTest, BlocksForTokensCeils) {
  EXPECT_EQ(BlocksForTokens(1, 16), 1);
  EXPECT_EQ(BlocksForTokens(16, 16), 1);
  EXPECT_EQ(BlocksForTokens(17, 16), 2);
  EXPECT_EQ(BlocksForTokens(64, 8), 8);
  EXPECT_EQ(BlocksForTokens(65, 8), 9);
}

TEST(KvBudgetTest, BlocksForTokensNonPositiveIsOne) {
  EXPECT_EQ(BlocksForTokens(0, 16), 1);
  EXPECT_EQ(BlocksForTokens(-5, 16), 1);
}

// --- Host memory helper (§5.3) ---

TEST(KvBudgetTest, HostMemoryIsPositiveOnDevAndCi) {
  // macOS (sysctl) and Linux (sysconf) both report a positive figure; the CI
  // and dev machines are both supported platforms.
  EXPECT_GT(engine::core::host_memory_bytes(), 0);
}

}  // namespace
