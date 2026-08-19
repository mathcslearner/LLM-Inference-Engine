#include "sampling/sampler.h"

#include "core/status.h"
#include "sampling/params.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
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

// Temperature/top-k/top-p are implemented in T02: Create now accepts them.
TEST(SamplerTest, AcceptsStochasticTemperature) {
  SamplingParams p;  // default temperature 1.0
  p.max_tokens = 8;
  EXPECT_TRUE(Sampler::Create(p).ok());
}

TEST(SamplerTest, AcceptsTopK) {
  SamplingParams p;
  p.max_tokens = 8;
  p.top_k = 40;
  EXPECT_TRUE(Sampler::Create(p).ok());
}

TEST(SamplerTest, AcceptsTopP) {
  SamplingParams p;
  p.max_tokens = 8;
  p.top_p = 0.9F;
  EXPECT_TRUE(Sampler::Create(p).ok());
}

// Each still-unimplemented knob is rejected at Create with Unimplemented, so no
// requested behaviour is ever silently dropped.
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

// ------------------------------------------------- stochastic sampling --

[[nodiscard]] Sampler MakeSampler(SamplingParams params) {
  StatusOr<Sampler> sampler = Sampler::Create(std::move(params));
  EXPECT_TRUE(sampler.ok()) << sampler.status().ToString();
  return *std::move(sampler);
}

// A stochastic sampler with an explicit seed and otherwise-default knobs.
[[nodiscard]] Sampler SeededSampler(std::uint64_t seed,
                                    float temperature = 1.0F,
                                    std::int32_t top_k = 0,
                                    float top_p = 1.0F) {
  SamplingParams p;
  p.max_tokens = 1 << 20;
  p.temperature = temperature;
  p.top_k = top_k;
  p.top_p = top_p;
  p.seed = seed;
  return MakeSampler(std::move(p));
}

// Softmax reference (fp64) for the expected sampling distribution.
[[nodiscard]] std::vector<double> SoftmaxRef(std::span<const float> logits,
                                             float temperature) {
  std::vector<double> p(logits.size());
  double max_l = -std::numeric_limits<double>::infinity();
  for (const float x : logits) {
    max_l = std::max(max_l, static_cast<double>(x) / temperature);
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp((static_cast<double>(logits[i]) / temperature) - max_l);
    sum += p[i];
  }
  for (double& v : p) {
    v /= sum;
  }
  return p;
}

// Draw `n` tokens from `sampler` at successive step indices (the RNG counter),
// tallying a histogram over `vocab`.
[[nodiscard]] std::vector<int> DrawHistogram(const Sampler& sampler,
                                             std::span<const float> logits,
                                             int vocab, int n) {
  std::vector<int> hist(static_cast<std::size_t>(vocab), 0);
  std::vector<std::int32_t> generated;  // only its size (the step) matters
  generated.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const SampleContext ctx{.prompt_ids = {}, .generated_ids = generated};
    StatusOr<std::int32_t> id = sampler.Sample(logits, ctx);
    EXPECT_TRUE(id.ok()) << id.status().ToString();
    ++hist[static_cast<std::size_t>(*id)];
    generated.push_back(0);
  }
  return hist;
}

[[nodiscard]] double ChiSquare(const std::vector<int>& hist,
                               const std::vector<double>& probs, int n) {
  double chi = 0.0;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    const double expected = static_cast<double>(n) * probs[i];
    const double diff = static_cast<double>(hist[i]) - expected;
    chi += (diff * diff) / expected;
  }
  return chi;
}

// The empirical distribution over 20k draws matches the softmax within the
// chi-square 0.001 critical value for dof = V-1 = 4 (18.47); the test is
// deterministic (fixed seed), so this is a fixed pass, not a flaky bound.
TEST(SamplerTest, ChiSquareMatchesSoftmaxAtUnitTemperature) {
  const std::vector<float> logits = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F};
  const Sampler sampler = SeededSampler(/*seed=*/42);
  const int n = 20000;
  const std::vector<int> hist =
      DrawHistogram(sampler, logits, static_cast<int>(logits.size()), n);
  const double chi = ChiSquare(hist, SoftmaxRef(logits, 1.0F), n);
  EXPECT_LT(chi, 18.47) << "chi-square = " << chi;
}

TEST(SamplerTest, ChiSquareMatchesSoftmaxAtLowTemperature) {
  const std::vector<float> logits = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F};
  const Sampler sampler = SeededSampler(/*seed=*/7, /*temperature=*/0.5F);
  const int n = 20000;
  const std::vector<int> hist =
      DrawHistogram(sampler, logits, static_cast<int>(logits.size()), n);
  const double chi = ChiSquare(hist, SoftmaxRef(logits, 0.5F), n);
  EXPECT_LT(chi, 18.47) << "chi-square = " << chi;
}

TEST(SamplerTest, TopKRestrictsSupportToKTokens) {
  const std::vector<float> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.0F};
  const Sampler sampler =
      SeededSampler(/*seed=*/3, /*temperature=*/1.0F, /*top_k=*/2);
  const std::vector<int> hist =
      DrawHistogram(sampler, logits, static_cast<int>(logits.size()), 4000);
  // Only the two highest-logit tokens (indices 0, 1) are ever sampled.
  EXPECT_GT(hist[0], 0);
  EXPECT_GT(hist[1], 0);
  for (std::size_t i = 2; i < logits.size(); ++i) {
    EXPECT_EQ(hist[i], 0) << "token " << i;
  }
}

TEST(SamplerTest, TopPRestrictsSupportToNucleus) {
  // Softmax over these logits: index 0 ~0.634, index 1 ~0.233, tail small.
  const std::vector<float> logits = {3.0F, 2.0F, 1.0F, 0.0F, -1.0F};
  const Sampler sampler =
      SeededSampler(/*seed=*/5, /*temperature=*/1.0F, /*top_k=*/0,
                    /*top_p=*/0.85F);
  const std::vector<int> hist =
      DrawHistogram(sampler, logits, static_cast<int>(logits.size()), 4000);
  // Cumulative reaches 0.85 by index 1 (0.634 + 0.233 = 0.867), so only tokens
  // 0 and 1 fall inside the nucleus.
  EXPECT_GT(hist[0], 0);
  EXPECT_GT(hist[1], 0);
  for (std::size_t i = 2; i < logits.size(); ++i) {
    EXPECT_EQ(hist[i], 0) << "token " << i;
  }
}

// Same seed ⇒ identical token sequence across two independent samplers.
TEST(SamplerTest, SameSeedProducesIdenticalSequence) {
  const std::vector<float> logits = {1.0F, 2.0F, 0.5F, -0.5F, 1.5F, 0.0F};
  const Sampler a = SeededSampler(/*seed=*/2024);
  const Sampler b = SeededSampler(/*seed=*/2024);
  std::vector<std::int32_t> seq_a;
  std::vector<std::int32_t> seq_b;
  for (int step = 0; step < 200; ++step) {
    const std::vector<std::int32_t> gen(static_cast<std::size_t>(step), 0);
    const SampleContext ctx{.prompt_ids = {}, .generated_ids = gen};
    seq_a.push_back(*a.Sample(logits, ctx));
    seq_b.push_back(*b.Sample(logits, ctx));
  }
  EXPECT_EQ(seq_a, seq_b);
}

// Different seeds ⇒ different sequences (independent streams).
TEST(SamplerTest, DifferentSeedsProduceDifferentSequences) {
  const std::vector<float> logits = {1.0F, 2.0F, 0.5F, -0.5F, 1.5F, 0.0F};
  const Sampler a = SeededSampler(/*seed=*/1);
  const Sampler b = SeededSampler(/*seed=*/2);
  std::vector<std::int32_t> seq_a;
  std::vector<std::int32_t> seq_b;
  for (int step = 0; step < 200; ++step) {
    const std::vector<std::int32_t> gen(static_cast<std::size_t>(step), 0);
    const SampleContext ctx{.prompt_ids = {}, .generated_ids = gen};
    seq_a.push_back(*a.Sample(logits, ctx));
    seq_b.push_back(*b.Sample(logits, ctx));
  }
  EXPECT_NE(seq_a, seq_b);
}

// The draw depends only on (seed, step) — not on the *contents* of the token
// history — so it is independent of batch composition.
TEST(SamplerTest, DrawDependsOnStepNotHistoryContents) {
  const std::vector<float> logits = {1.0F, 2.0F, 0.5F, -0.5F, 1.5F, 0.0F};
  const Sampler sampler = SeededSampler(/*seed=*/99);
  // Two histories of the same length (step index) but different contents.
  const std::vector<std::int32_t> gen_x = {7, 7, 7};
  const std::vector<std::int32_t> gen_y = {1, 2, 3};
  const std::vector<std::int32_t> prompt_x = {5};
  const std::vector<std::int32_t> prompt_y = {8, 9};
  const auto id_x = *sampler.Sample(
      logits, SampleContext{.prompt_ids = prompt_x, .generated_ids = gen_x});
  const auto id_y = *sampler.Sample(
      logits, SampleContext{.prompt_ids = prompt_y, .generated_ids = gen_y});
  EXPECT_EQ(id_x, id_y);
}

TEST(SamplerTest, SeedAccessorEchoesRequestSeed) {
  const Sampler sampler = SeededSampler(/*seed=*/123456789ULL);
  EXPECT_EQ(sampler.seed(), 123456789ULL);
}

TEST(SamplerTest, NulloptSeedsResolveToDistinctSeeds) {
  SamplingParams p;
  p.max_tokens = 8;  // temperature 1.0, seed nullopt
  const Sampler a = MakeSampler(p);
  const Sampler b = MakeSampler(p);
  EXPECT_NE(a.seed(), b.seed());
}

// ------------------------------------------ stochastic error posture --

TEST(SamplerTest, StochasticEmptyRowIsInvalidArgument) {
  const Sampler sampler = SeededSampler(/*seed=*/1);
  const std::span<const float> empty{};
  EXPECT_TRUE(IsInvalidArgument(sampler.Sample(empty, kNoContext).status()));
}

TEST(SamplerTest, StochasticNaNIsInternal) {
  const Sampler sampler = SeededSampler(/*seed=*/1);
  const std::vector<float> logits = {1.0F,
                                     std::numeric_limits<float>::quiet_NaN()};
  EXPECT_TRUE(IsInternal(sampler.Sample(logits, kNoContext).status()));
}

TEST(SamplerTest, StochasticPositiveInfIsInternal) {
  const Sampler sampler = SeededSampler(/*seed=*/1);
  const std::vector<float> logits = {1.0F,
                                     std::numeric_limits<float>::infinity()};
  EXPECT_TRUE(IsInternal(sampler.Sample(logits, kNoContext).status()));
}

TEST(SamplerTest, StochasticNeverSamplesMaskedNegInfToken) {
  const std::vector<float> logits = {
      2.0F, -std::numeric_limits<float>::infinity(), 1.0F};
  const Sampler sampler = SeededSampler(/*seed=*/11);
  const std::vector<int> hist =
      DrawHistogram(sampler, logits, static_cast<int>(logits.size()), 2000);
  EXPECT_EQ(hist[1], 0);  // the -inf token is never drawn
}

}  // namespace
