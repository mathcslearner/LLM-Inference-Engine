#include "core/status.h"
#include "sampling/params.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>

// M7-T01 (design: docs/design/model-execution.md §15): SamplingParams defaults
// and validation. Portable C++ — no ISA concern, no SCALAR_PASS. Every
// out-of-range field is rejected with InvalidArgument naming the field; the
// boundary-valid values and the identity/greedy configurations are accepted.

namespace {

using engine::core::IsInvalidArgument;
using engine::sampling::kMaxLogprobs;
using engine::sampling::SamplingParams;
using engine::sampling::ValidateSamplingParams;

// A minimally valid params: defaults reject max_tokens == 0, so set it.
[[nodiscard]] SamplingParams Valid() {
  SamplingParams params;
  params.max_tokens = 16;
  return params;
}

// ------------------------------------------------------------- accepted --

TEST(SamplingParamsTest, DefaultsAreIdentityExceptMaxTokens) {
  const SamplingParams params;
  // The default configuration is "sample from the raw softmax": temperature 1,
  // filters/penalties off. Only max_tokens must be set by the caller.
  EXPECT_EQ(params.temperature, 1.0F);
  EXPECT_EQ(params.top_k, 0);
  EXPECT_EQ(params.top_p, 1.0F);
  EXPECT_EQ(params.repetition_penalty, 1.0F);
  EXPECT_EQ(params.presence_penalty, 0.0F);
  EXPECT_EQ(params.frequency_penalty, 0.0F);
  EXPECT_FALSE(params.seed.has_value());
  EXPECT_EQ(params.max_tokens, 0);
  EXPECT_TRUE(params.stop_token_ids.empty());
  EXPECT_TRUE(params.stop_strings.empty());
  EXPECT_EQ(params.logprobs, 0);
}

TEST(SamplingParamsTest, ValidParamsAccepted) {
  EXPECT_TRUE(ValidateSamplingParams(Valid()).ok());
}

TEST(SamplingParamsTest, GreedyIsValidAndTemperatureZero) {
  const SamplingParams params = SamplingParams::Greedy(32);
  EXPECT_EQ(params.temperature, 0.0F);
  EXPECT_EQ(params.max_tokens, 32);
  EXPECT_TRUE(ValidateSamplingParams(params).ok());
}

TEST(SamplingParamsTest, BoundaryValuesAccepted) {
  SamplingParams params = Valid();
  params.temperature = 0.0F;  // greedy boundary
  params.top_p = 1.0F;        // keep-everything boundary
  params.presence_penalty = 2.0F;
  params.frequency_penalty = -2.0F;
  params.logprobs = kMaxLogprobs;
  params.stop_token_ids = {0};  // 0 is a valid id
  params.stop_strings = {" "};  // non-empty
  EXPECT_TRUE(ValidateSamplingParams(params).ok());
}

// ------------------------------------------------------------- rejected --

// Assert validation fails with InvalidArgument whose message names `field`.
void ExpectRejected(const SamplingParams& params, const std::string& field) {
  const auto status = ValidateSamplingParams(params);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
  EXPECT_NE(status.message().find(field), std::string::npos)
      << status.message();
}

TEST(SamplingParamsTest, RejectsNegativeTemperature) {
  SamplingParams params = Valid();
  params.temperature = -0.5F;
  ExpectRejected(params, "temperature");
}

TEST(SamplingParamsTest, RejectsNonFiniteTemperature) {
  SamplingParams params = Valid();
  params.temperature = std::numeric_limits<float>::quiet_NaN();
  ExpectRejected(params, "temperature");
}

TEST(SamplingParamsTest, RejectsNegativeTopK) {
  SamplingParams params = Valid();
  params.top_k = -1;
  ExpectRejected(params, "top_k");
}

TEST(SamplingParamsTest, RejectsTopPZero) {
  SamplingParams params = Valid();
  params.top_p = 0.0F;
  ExpectRejected(params, "top_p");
}

TEST(SamplingParamsTest, RejectsTopPAboveOne) {
  SamplingParams params = Valid();
  params.top_p = 1.5F;
  ExpectRejected(params, "top_p");
}

TEST(SamplingParamsTest, RejectsNonFiniteTopP) {
  SamplingParams params = Valid();
  params.top_p = std::numeric_limits<float>::infinity();
  ExpectRejected(params, "top_p");
}

TEST(SamplingParamsTest, RejectsNonPositiveRepetitionPenalty) {
  SamplingParams params = Valid();
  params.repetition_penalty = 0.0F;
  ExpectRejected(params, "repetition_penalty");
}

TEST(SamplingParamsTest, RejectsPresencePenaltyOutOfRange) {
  SamplingParams params = Valid();
  params.presence_penalty = 2.5F;
  ExpectRejected(params, "presence_penalty");
}

TEST(SamplingParamsTest, RejectsFrequencyPenaltyOutOfRange) {
  SamplingParams params = Valid();
  params.frequency_penalty = -2.5F;
  ExpectRejected(params, "frequency_penalty");
}

TEST(SamplingParamsTest, RejectsNonPositiveMaxTokens) {
  SamplingParams params = Valid();
  params.max_tokens = 0;
  ExpectRejected(params, "max_tokens");
}

TEST(SamplingParamsTest, RejectsNegativeStopTokenId) {
  SamplingParams params = Valid();
  params.stop_token_ids = {5, -3};
  ExpectRejected(params, "stop_token_ids");
}

TEST(SamplingParamsTest, RejectsEmptyStopString) {
  SamplingParams params = Valid();
  params.stop_strings = {"ok", ""};
  ExpectRejected(params, "stop_strings");
}

TEST(SamplingParamsTest, RejectsNegativeLogprobs) {
  SamplingParams params = Valid();
  params.logprobs = -1;
  ExpectRejected(params, "logprobs");
}

TEST(SamplingParamsTest, RejectsLogprobsAboveCap) {
  SamplingParams params = Valid();
  params.logprobs = kMaxLogprobs + 1;
  ExpectRejected(params, "logprobs");
}

}  // namespace
