#include "core/status.h"
#include "sampling/stages.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

// M7-T03 (design: docs/design/model-execution.md §15.2, stage 1): the
// repetition / frequency / presence penalties applied over the request's token
// history. Portable C++ — no ISA concern, no SCALAR_PASS. Every case uses
// dyadic (exactly fp32-representable) logits and penalties so the expected
// post-penalty logits can be hand-computed and asserted with exact equality —
// the acceptance criterion. History convention under test: repetition penalizes
// every token seen in prompt ∪ generated (once, not compounded); frequency and
// presence count *generated* tokens only.

namespace {

using engine::core::IsInvalidArgument;
using engine::sampling::detail::ApplyPenalties;

constexpr float kInf = std::numeric_limits<float>::infinity();

// A shared row and history reused across the single-penalty cases:
//   logits  = {2.0, -1.0, 0.5, 4.0, -0.5}
//   prompt  = {0}          (token 0 appears in the prompt only)
//   gen     = {1, 1, 3}    (token 1 twice, token 3 once)
// so seen = {0,1,3}, generated counts = {1:2, 3:1}.
[[nodiscard]] std::vector<float> BaseLogits() {
  return {2.0F, -1.0F, 0.5F, 4.0F, -0.5F};
}
const std::vector<std::int32_t> kPrompt = {0};
const std::vector<std::int32_t> kGenerated = {1, 1, 3};

[[nodiscard]] std::vector<float> Penalize(std::vector<float> logits, float rep,
                                          float pres, float freq) {
  const auto status = ApplyPenalties(std::span<float>{logits}, kPrompt,
                                     kGenerated, rep, pres, freq);
  EXPECT_TRUE(status.ok()) << status.ToString();
  return logits;
}

// --------------------------------------------------------- single penalty --

// Repetition (r = 2): positive logits of seen tokens are divided, negative ones
// multiplied. Token 0 (prompt-only) IS penalized; tokens 2/4 (unseen) are not.
TEST(PenaltiesTest, RepetitionOnly) {
  const std::vector<float> got = Penalize(BaseLogits(), 2.0F, 0.0F, 0.0F);
  const std::vector<float> want = {1.0F, -2.0F, 0.5F, 2.0F, -0.5F};
  EXPECT_EQ(got, want);
}

// Repetition with r < 1 boosts (divide by <1 grows positive logits) rather than
// suppresses — the HF sign convention holds for the whole positive range.
TEST(PenaltiesTest, RepetitionBelowOneBoosts) {
  const std::vector<float> got = Penalize(BaseLogits(), 0.5F, 0.0F, 0.0F);
  // token0 2.0/0.5=4.0, token1 -1.0*0.5=-0.5, token3 4.0/0.5=8.0.
  const std::vector<float> want = {4.0F, -0.5F, 0.5F, 8.0F, -0.5F};
  EXPECT_EQ(got, want);
}

// Frequency (f = 0.5): subtract f * occurrences, generated tokens only. Token 3
// once (-0.5), token 1 twice (-1.0); the prompt-only token 0 is untouched.
TEST(PenaltiesTest, FrequencyOnly) {
  const std::vector<float> got = Penalize(BaseLogits(), 1.0F, 0.0F, 0.5F);
  const std::vector<float> want = {2.0F, -2.0F, 0.5F, 3.5F, -0.5F};
  EXPECT_EQ(got, want);
}

// Presence (p = 0.25): a flat subtraction from every distinct generated token,
// regardless of count; the prompt-only token 0 is untouched.
TEST(PenaltiesTest, PresenceOnly) {
  const std::vector<float> got = Penalize(BaseLogits(), 1.0F, 0.25F, 0.0F);
  const std::vector<float> want = {2.0F, -1.25F, 0.5F, 3.75F, -0.5F};
  EXPECT_EQ(got, want);
}

// A negative presence penalty is a flat *boost* (subtracting a negative).
TEST(PenaltiesTest, NegativePresenceBoosts) {
  const std::vector<float> got = Penalize(BaseLogits(), 1.0F, -0.5F, 0.0F);
  const std::vector<float> want = {2.0F, -0.5F, 0.5F, 4.5F, -0.5F};
  EXPECT_EQ(got, want);
}

// ------------------------------------------------------ combined penalties --

// All three together, applied in the documented order repetition -> frequency
// -> presence:
//   after rep  {1.0, -2.0, 0.5, 2.0, -0.5}
//   after freq {1.0, -3.0, 0.5, 1.5, -0.5}   (token1 -1.0, token3 -0.5)
//   after pres {1.0, -3.25, 0.5, 1.25, -0.5} (token1/token3 -0.25 each)
TEST(PenaltiesTest, AllThreeComposeInOrder) {
  const std::vector<float> got = Penalize(BaseLogits(), 2.0F, 0.25F, 0.5F);
  const std::vector<float> want = {1.0F, -3.25F, 0.5F, 1.25F, -0.5F};
  EXPECT_EQ(got, want);
}

// ------------------------------------------------------------- edge cases --

// Defaults (r=1, p=0, f=0) are an exact no-op: the logits, including -inf and
// negative entries, are left bitwise unchanged even with a non-empty history.
TEST(PenaltiesTest, DefaultsAreExactNoOp) {
  std::vector<float> logits = {2.0F, -kInf, 0.5F, -3.0F};
  const std::vector<float> before = logits;
  const auto status = ApplyPenalties(std::span<float>{logits}, kPrompt,
                                     kGenerated, 1.0F, 0.0F, 0.0F);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(logits, before);
}

// A masked (-inf) logit stays -inf under every operation.
TEST(PenaltiesTest, MaskedLogitStaysNegInf) {
  std::vector<float> logits = {-kInf, 2.0F};
  const std::vector<std::int32_t> gen = {0, 1};
  const auto status =
      ApplyPenalties(std::span<float>{logits}, {}, gen, 2.0F, 0.25F, 0.5F);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(logits[0], -kInf);  // -inf < 0 -> -inf * r = -inf, then -f -p
  EXPECT_EQ(logits[1], 0.25F);  // 2.0/2=1.0, -0.5*1, -0.25 -> 0.25
}

// Empty history with active penalties changes nothing.
TEST(PenaltiesTest, EmptyHistoryIsNoOp) {
  std::vector<float> logits = BaseLogits();
  const std::vector<float> before = logits;
  const auto status =
      ApplyPenalties(std::span<float>{logits}, {}, {}, 2.0F, 0.25F, 0.5F);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(logits, before);
}

// A token seen twice in the prompt is still penalized only once (not
// compounded) by repetition.
TEST(PenaltiesTest, RepetitionAppliedOncePerDistinctToken) {
  std::vector<float> logits = {4.0F, 1.0F};
  const std::vector<std::int32_t> prompt = {0, 0, 0};  // token 0 thrice
  const auto status =
      ApplyPenalties(std::span<float>{logits}, prompt, {}, 2.0F, 0.0F, 0.0F);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(logits[0], 2.0F);  // 4.0/2 once, not 4.0/2/2/2
  EXPECT_EQ(logits[1], 1.0F);
}

// An out-of-range history id is InvalidArgument, and the logits are left
// untouched (validation is front-loaded before any mutation).
TEST(PenaltiesTest, OutOfRangeIdRejectedLogitsUntouched) {
  std::vector<float> logits = {1.0F, 2.0F, 3.0F};
  const std::vector<float> before = logits;
  const std::vector<std::int32_t> gen = {5};  // >= vocab (3)
  const auto status =
      ApplyPenalties(std::span<float>{logits}, {}, gen, 2.0F, 0.0F, 0.0F);
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_EQ(logits, before);
}

TEST(PenaltiesTest, NegativeIdRejected) {
  std::vector<float> logits = {1.0F, 2.0F};
  const std::vector<std::int32_t> prompt = {-1};
  const auto status =
      ApplyPenalties(std::span<float>{logits}, prompt, {}, 2.0F, 0.0F, 0.0F);
  EXPECT_TRUE(IsInvalidArgument(status));
}

}  // namespace
