#include "sampling/sampler.h"

#include "core/status.h"
#include "sampling/params.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

// M7-T01 (design: docs/design/model-execution.md §15): the Sampler skeleton and
// its greedy selection branch. Portable C++ — no ISA concern, no SCALAR_PASS.
// Covers: greedy argmax over a logits row (with lowest-index tie-break),
// NaN/empty-row error posture, and the Create-time Unimplemented guards for
// every not-yet-landed stage (temperature > 0, filters, penalties, stop
// conditions, logprobs).

namespace {

using engine::core::IsInternal;
using engine::core::IsInvalidArgument;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::sampling::SampleContext;
using engine::sampling::Sampler;
using engine::sampling::SamplingParams;

[[nodiscard]] Sampler GreedySampler() {
  StatusOr<Sampler> sampler = Sampler::Create(SamplingParams::Greedy(8));
  EXPECT_TRUE(sampler.ok()) << sampler.status().ToString();
  return *std::move(sampler);
}

// Empty history: greedy ignores the context, but the fields must be valid.
constexpr SampleContext kNoContext{};

// ----------------------------------------------------------- greedy path --

TEST(SamplerTest, GreedyPicksArgmax) {
  const Sampler sampler = GreedySampler();
  const std::vector<float> logits = {0.1F, 3.5F, -2.0F, 3.4F};
  StatusOr<std::int32_t> id = sampler.Sample(logits, kNoContext);
  ASSERT_TRUE(id.ok()) << id.status().ToString();
  EXPECT_EQ(*id, 1);
}

TEST(SamplerTest, GreedyTieBreaksToLowestIndex) {
  const Sampler sampler = GreedySampler();
  // Two positions share the maximum; the strict `>` scan keeps the first.
  const std::vector<float> logits = {2.0F, 5.0F, 5.0F, 1.0F};
  StatusOr<std::int32_t> id = sampler.Sample(logits, kNoContext);
  ASSERT_TRUE(id.ok()) << id.status().ToString();
  EXPECT_EQ(*id, 1);
}

TEST(SamplerTest, GreedySingleElementRow) {
  const Sampler sampler = GreedySampler();
  const std::vector<float> logits = {-4.2F};
  StatusOr<std::int32_t> id = sampler.Sample(logits, kNoContext);
  ASSERT_TRUE(id.ok()) << id.status().ToString();
  EXPECT_EQ(*id, 0);
}

TEST(SamplerTest, GreedyIsDeterministicAcrossCalls) {
  const Sampler sampler = GreedySampler();
  const std::vector<float> logits = {1.0F, 0.5F, 9.0F, 9.0F, 2.0F};
  const std::int32_t first = *sampler.Sample(logits, kNoContext);
  const std::int32_t second = *sampler.Sample(logits, kNoContext);
  EXPECT_EQ(first, 2);
  EXPECT_EQ(second, 2);
}

// ------------------------------------------------------------ error paths --

TEST(SamplerTest, EmptyLogitsRowIsInvalidArgument) {
  const Sampler sampler = GreedySampler();
  const std::span<const float> empty{};
  const StatusOr<std::int32_t> id = sampler.Sample(empty, kNoContext);
  ASSERT_FALSE(id.ok());
  EXPECT_TRUE(IsInvalidArgument(id.status()));
}

TEST(SamplerTest, NaNMaxIsInternal) {
  const Sampler sampler = GreedySampler();
  // NaN as the running maximum surfaces as Internal, not a bogus token. Because
  // the scan seeds best_val from index 0 and updates only on a strict `>`
  // (which NaN never satisfies), the max is NaN exactly when the leading logit
  // is NaN — the ported M5 argmax contract.
  const std::vector<float> logits = {std::numeric_limits<float>::quiet_NaN(),
                                     1.0F};
  const StatusOr<std::int32_t> id = sampler.Sample(logits, kNoContext);
  ASSERT_FALSE(id.ok());
  EXPECT_TRUE(IsInternal(id.status())) << id.status().ToString();
}

// --------------------------------------------------- Create-time guards --

// Each not-yet-implemented knob is rejected at Create with Unimplemented, so no
// requested behaviour is ever silently dropped.
TEST(SamplerTest, RejectsStochasticTemperature) {
  auto sampler = Sampler::Create([] {
    SamplingParams p = SamplingParams::Greedy(8);
    p.temperature = 0.7F;
    return p;
  }());
  ASSERT_FALSE(sampler.ok());
  EXPECT_TRUE(IsUnimplemented(sampler.status())) << sampler.status().ToString();
}

TEST(SamplerTest, RejectsTopK) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.top_k = 40;
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsTopP) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.top_p = 0.9F;
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsRepetitionPenalty) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.repetition_penalty = 1.1F;
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsPresencePenalty) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.presence_penalty = 0.5F;
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsFrequencyPenalty) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.frequency_penalty = 0.5F;
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsStopTokenIds) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.stop_token_ids = {7};
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsStopStrings) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.stop_strings = {"STOP"};
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

TEST(SamplerTest, RejectsLogprobs) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.logprobs = 5;
  EXPECT_TRUE(IsUnimplemented(Sampler::Create(p).status()));
}

// Invalid params surface as InvalidArgument (from ValidateSamplingParams),
// distinct from the Unimplemented guards above.
TEST(SamplerTest, RejectsInvalidParamsWithInvalidArgument) {
  const SamplingParams p = SamplingParams::Greedy(0);  // max_tokens must be > 0
  EXPECT_TRUE(IsInvalidArgument(Sampler::Create(p).status()));
}

}  // namespace
