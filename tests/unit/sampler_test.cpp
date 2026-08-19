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
// NaN/empty-row error posture, penalties (T03) affecting greedy selection, the
// stochastic branch (T02), and logprobs (T05) via `SampleWithLogprobs`. As of
// T05 `Create` no longer rejects any knob with Unimplemented.

namespace {

using engine::core::IsInternal;
using engine::core::IsInvalidArgument;
using engine::core::StatusOr;
using engine::sampling::SampleContext;
using engine::sampling::Sampler;
using engine::sampling::SampleResult;
using engine::sampling::SamplingParams;
using engine::sampling::StepLogprobs;

[[nodiscard]] Sampler GreedySampler() {
  StatusOr<Sampler> sampler = Sampler::Create(SamplingParams::Greedy(8));
  EXPECT_TRUE(sampler.ok()) << sampler.status().ToString();
  return *std::move(sampler);
}

// Build a sampler from arbitrary params, asserting Create succeeds.
[[nodiscard]] Sampler MakeSampler(SamplingParams params) {
  StatusOr<Sampler> sampler = Sampler::Create(std::move(params));
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

// The penalty knobs are implemented (T03): Create accepts non-default values,
// where before it rejected them with Unimplemented.
TEST(SamplerTest, AcceptsPenalties) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.repetition_penalty = 1.3F;
  p.presence_penalty = 0.5F;
  p.frequency_penalty = 0.5F;
  EXPECT_TRUE(Sampler::Create(p).ok());
}

// Penalties feed the greedy argmax: repetition demotes a previously generated
// token below an un-generated one, flipping the selection. logits {3.0, 4.0}
// argmax to 1; with token 1 in the history and r=2 it becomes 4.0/2=2.0 < 3.0,
// so the argmax moves to 0.
TEST(SamplerTest, GreedyArgmaxShiftsUnderRepetitionPenalty) {
  const std::vector<float> logits = {3.0F, 4.0F};
  const std::vector<std::int32_t> generated = {1};
  const SampleContext ctx{.prompt_ids = {}, .generated_ids = generated};

  const Sampler plain = GreedySampler();
  StatusOr<std::int32_t> without = plain.Sample(logits, ctx);
  ASSERT_TRUE(without.ok()) << without.status().ToString();
  EXPECT_EQ(*without, 1);  // no penalty: raw argmax

  SamplingParams p = SamplingParams::Greedy(8);
  p.repetition_penalty = 2.0F;
  const Sampler penalized = MakeSampler(std::move(p));
  StatusOr<std::int32_t> with = penalized.Sample(logits, ctx);
  ASSERT_TRUE(with.ok()) << with.status().ToString();
  EXPECT_EQ(*with, 0);  // penalty demotes token 1 below token 0
}

// A greedy sample with a default-only history is bit-identical whether or not
// penalties are configured, and the penalized path leaves the argmax unmoved
// when the history is empty.
TEST(SamplerTest, GreedyPenaltyNoOpWithEmptyHistory) {
  const std::vector<float> logits = {0.1F, 3.5F, -2.0F, 3.4F};
  SamplingParams p = SamplingParams::Greedy(8);
  p.repetition_penalty = 2.0F;
  p.presence_penalty = 1.0F;
  p.frequency_penalty = 1.0F;
  const Sampler penalized = MakeSampler(std::move(p));
  StatusOr<std::int32_t> id = penalized.Sample(logits, kNoContext);
  ASSERT_TRUE(id.ok()) << id.status().ToString();
  EXPECT_EQ(*id, 1);  // unchanged from the un-penalized GreedyPicksArgmax case
}

// An out-of-range history id surfaces as InvalidArgument through Sample (the
// penalty stage validates before touching logits).
TEST(SamplerTest, GreedyPenaltyRejectsOutOfRangeHistory) {
  const std::vector<float> logits = {1.0F, 2.0F, 3.0F};
  const std::vector<std::int32_t> generated = {9};  // >= vocab
  const SampleContext ctx{.prompt_ids = {}, .generated_ids = generated};
  SamplingParams p = SamplingParams::Greedy(8);
  p.frequency_penalty = 0.5F;
  const Sampler penalized = MakeSampler(std::move(p));
  EXPECT_TRUE(IsInvalidArgument(penalized.Sample(logits, ctx).status()));
}

// Stop conditions are the generation loop's StopChecker (M7-T04), not the
// sampler's: Create accepts them and ignores them (no Unimplemented guard).
TEST(SamplerTest, AcceptsStopTokenIds) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.stop_token_ids = {7};
  EXPECT_TRUE(Sampler::Create(p).ok());
}

TEST(SamplerTest, AcceptsStopStrings) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.stop_strings = {"STOP"};
  EXPECT_TRUE(Sampler::Create(p).ok());
}

// Logprobs land in T05: Create now accepts the knob (the last Unimplemented
// guard is gone — every field is implemented).
TEST(SamplerTest, AcceptsLogprobs) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.logprobs = 5;
  EXPECT_TRUE(Sampler::Create(p).ok());
}

// Invalid params surface as InvalidArgument (from ValidateSamplingParams),
// distinct from the Unimplemented guards above.
TEST(SamplerTest, RejectsInvalidParamsWithInvalidArgument) {
  const SamplingParams p = SamplingParams::Greedy(0);  // max_tokens must be > 0
  EXPECT_TRUE(IsInvalidArgument(Sampler::Create(p).status()));
}

// ------------------------------------------------- stochastic sampling --

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

// ---------------------------------------------------------------- logprobs --

// Full-vocab natural-log softmax reference (fp64) at a given temperature.
[[nodiscard]] std::vector<double> LogSoftmaxRef(std::span<const float> logits,
                                                float temperature) {
  double max_l = -std::numeric_limits<double>::infinity();
  for (const float x : logits) {
    max_l = std::max(max_l, static_cast<double>(x) / temperature);
  }
  double sum = 0.0;
  for (const float x : logits) {
    sum += std::exp((static_cast<double>(x) / temperature) - max_l);
  }
  const double log_z = std::log(sum);
  std::vector<double> lp(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    lp[i] = ((static_cast<double>(logits[i]) / temperature) - max_l) - log_z;
  }
  return lp;
}

// Unwrap the StepLogprobs a SampleWithLogprobs result must carry, asserting it
// is present. `value_or` (not `value`) keeps clang-tidy's optional-access check
// happy while EXPECT_TRUE flags a genuinely absent value.
[[nodiscard]] StepLogprobs TakeLogprobs(const SampleResult& result) {
  EXPECT_TRUE(result.logprobs.has_value());
  return result.logprobs.value_or(StepLogprobs{});
}

// With logprobs disabled (the default), SampleWithLogprobs returns the token
// only — no StepLogprobs — and the token equals what Sample returns.
TEST(SamplerTest, LogprobsDisabledYieldsNoStepLogprobs) {
  const Sampler sampler = GreedySampler();  // logprobs == 0
  const std::vector<float> logits = {0.1F, 3.5F, -2.0F, 3.4F};
  StatusOr<SampleResult> r = sampler.SampleWithLogprobs(logits, kNoContext);
  ASSERT_TRUE(r.ok()) << r.status().ToString();
  EXPECT_EQ(r->token, 1);
  EXPECT_FALSE(r->logprobs.has_value());
  EXPECT_EQ(r->token, *sampler.Sample(logits, kNoContext));
}

// Acceptance criterion: the greedy chosen-token logprob equals the max logprob
// (== top[0]), and top[0] is the argmax token.
TEST(SamplerTest, GreedyChosenLogprobEqualsMax) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.logprobs = 4;
  const Sampler sampler = MakeSampler(std::move(p));
  const std::vector<float> logits = {0.1F, 3.5F, -2.0F, 3.4F, 1.0F};
  StatusOr<SampleResult> r = sampler.SampleWithLogprobs(logits, kNoContext);
  ASSERT_TRUE(r.ok()) << r.status().ToString();
  const StepLogprobs lp = TakeLogprobs(*r);
  ASSERT_FALSE(lp.top.empty());
  EXPECT_EQ(lp.top.front().id, r->token);
  EXPECT_FLOAT_EQ(lp.chosen_logprob, lp.top.front().logprob);
  // The chosen logprob matches the full-vocab log-softmax at the argmax.
  const std::vector<double> ref = LogSoftmaxRef(logits, /*temperature=*/1.0F);
  EXPECT_NEAR(lp.chosen_logprob,
              static_cast<float>(ref[static_cast<std::size_t>(r->token)]),
              1e-5F);
}

// Greedy logprobs are computed from the penalized row, so a penalty that moves
// the argmax also moves which logprob is reported as chosen.
TEST(SamplerTest, GreedyLogprobsUsePenalizedRow) {
  SamplingParams p = SamplingParams::Greedy(8);
  p.repetition_penalty = 2.0F;
  p.logprobs = 2;
  const Sampler sampler = MakeSampler(std::move(p));
  const std::vector<float> logits = {3.0F, 4.0F};
  const std::vector<std::int32_t> generated = {1};
  const SampleContext ctx{.prompt_ids = {}, .generated_ids = generated};
  StatusOr<SampleResult> r = sampler.SampleWithLogprobs(logits, ctx);
  ASSERT_TRUE(r.ok()) << r.status().ToString();
  EXPECT_EQ(r->token, 0);  // penalty demotes token 1 below token 0
  const StepLogprobs lp = TakeLogprobs(*r);
  // Reference over the penalized logits {3.0, 4.0/2 = 2.0}.
  const std::vector<float> penalized = {3.0F, 2.0F};
  const std::vector<double> ref = LogSoftmaxRef(penalized, 1.0F);
  EXPECT_NEAR(lp.chosen_logprob, static_cast<float>(ref[0]), 1e-5F);
  ASSERT_FALSE(lp.top.empty());
  EXPECT_EQ(lp.top.front().id, 0);
}

// Computing logprobs must not perturb the stochastic draw: the token stream is
// identical to the token-only path at every step.
TEST(SamplerTest, StochasticLogprobsDoNotChangeToken) {
  SamplingParams p;
  p.max_tokens = 1 << 20;
  p.temperature = 0.8F;
  p.top_k = 3;
  p.seed = 555;
  p.logprobs = 5;
  const Sampler sampler = MakeSampler(std::move(p));
  const std::vector<float> logits = {1.0F, 2.0F, 0.5F, -0.5F, 1.5F, 0.0F};
  for (int step = 0; step < 50; ++step) {
    const std::vector<std::int32_t> gen(static_cast<std::size_t>(step), 0);
    const SampleContext ctx{.prompt_ids = {}, .generated_ids = gen};
    const std::int32_t token_only = *sampler.Sample(logits, ctx);
    StatusOr<SampleResult> r = sampler.SampleWithLogprobs(logits, ctx);
    ASSERT_TRUE(r.ok()) << r.status().ToString();
    EXPECT_EQ(r->token, token_only) << "step " << step;
    const StepLogprobs lp = TakeLogprobs(*r);
    EXPECT_TRUE(std::isfinite(lp.chosen_logprob));
  }
}

// Even with top-k active, the reported logprobs are the full-vocabulary
// (temperature-scaled) distribution — taken before the top-k mask — so a
// token's logprob matches the untruncated log-softmax and the set covers the
// whole vocab.
TEST(SamplerTest, StochasticLogprobsAreFullVocabPreTruncation) {
  SamplingParams p;
  p.max_tokens = 1 << 20;
  p.temperature = 1.0F;
  p.top_k = 2;  // draws restricted, but logprobs describe all tokens
  p.seed = 321;
  p.logprobs = 6;  // >= vocab, so top lists every token
  const Sampler sampler = MakeSampler(std::move(p));
  const std::vector<float> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F, 0.0F};
  const SampleContext ctx = kNoContext;
  StatusOr<SampleResult> r = sampler.SampleWithLogprobs(logits, ctx);
  ASSERT_TRUE(r.ok()) << r.status().ToString();
  const StepLogprobs lp = TakeLogprobs(*r);
  // All six tokens are listed (none masked to -inf pre-truncation), summing to
  // 1 in prob space.
  EXPECT_EQ(lp.top.size(), logits.size());
  double mass = 0.0;
  for (const auto& t : lp.top) {
    mass += std::exp(static_cast<double>(t.logprob));
  }
  EXPECT_NEAR(mass, 1.0, 1e-5);
  // The chosen token's logprob matches the untruncated log-softmax.
  const std::vector<double> ref = LogSoftmaxRef(logits, 1.0F);
  EXPECT_NEAR(lp.chosen_logprob,
              static_cast<float>(ref[static_cast<std::size_t>(r->token)]),
              1e-5F);
}

}  // namespace
