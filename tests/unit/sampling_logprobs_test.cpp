#include "sampling/logprobs.h"
#include "sampling/stages.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

// M7-T05 (design: docs/design/model-execution.md §15.2 stage 6): the logprobs
// stages `LogSoftmax` + `ExtractLogprobs`. Portable C++ — no ISA concern, no
// SCALAR_PASS. Covers the acceptance criteria: the log-probs sum to 1 in prob
// space over the full vocabulary, top-N ordering matches a full-sort reference,
// and (with the sampler tests) the greedy chosen logprob equals the max.

namespace {

using engine::sampling::StepLogprobs;
using engine::sampling::TokenLogprob;
using engine::sampling::detail::ExtractLogprobs;
using engine::sampling::detail::LogSoftmax;

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// Reference natural-log softmax in double (independent of the implementation).
[[nodiscard]] std::vector<double> LogSoftmaxRef(std::span<const float> logits) {
  double max_l = -std::numeric_limits<double>::infinity();
  for (const float x : logits) {
    max_l = std::max(max_l, static_cast<double>(x));
  }
  double sum = 0.0;
  for (const float x : logits) {
    sum += std::exp(static_cast<double>(x) - max_l);
  }
  const double log_z = std::log(sum);
  std::vector<double> lp(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    lp[i] = std::isfinite(logits[i])
                ? (static_cast<double>(logits[i]) - max_l) - log_z
                : kNegInf;
  }
  return lp;
}

// A deterministic pseudo-random logits row (no RNG dependency).
[[nodiscard]] std::vector<float> MakeRow(std::size_t n, unsigned seed) {
  std::vector<float> row(n);
  unsigned state = seed;
  for (float& x : row) {
    state = state * 1664525U + 1013904223U;  // LCG
    // Map to roughly [-8, 8).
    x = (static_cast<float>(state >> 8U) / static_cast<float>(1U << 24U)) *
            16.0F -
        8.0F;
  }
  return row;
}

[[nodiscard]] double SumExp(const std::vector<double>& lp) {
  double s = 0.0;
  for (const double v : lp) {
    s += std::exp(v);
  }
  return s;
}

// ------------------------------------------------------------ LogSoftmax --

// M7-T06 note: `LogSoftmax` now exponentiates with the shared vector
// `kernels::ExpF32` polynomial (≤2 ulp of `expf`) rather than `std::exp`, so
// re-exponentiating the log-probs recovers 1 only to fp32-exp precision
// (~2.4e-7 relative). `SumExp` here still uses `std::exp(double)`, so the small
// gap is exactly the normalizer's poly-vs-double difference — hence the 1e-5
// tolerance (was 1e-12) on the "sums to one" assertions.
TEST(LogprobsTest, LogSoftmaxSumsToOneOverFullVocab) {
  const std::vector<float> row = MakeRow(512, /*seed=*/1234);
  std::vector<double> lp;
  LogSoftmax(row, lp);
  EXPECT_NEAR(SumExp(lp), 1.0, 1e-5);
}

TEST(LogprobsTest, LogSoftmaxUniformRow) {
  const std::vector<float> row = {0.0F, 0.0F, 0.0F, 0.0F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  const double expected = -std::log(4.0);
  for (const double v : lp) {
    EXPECT_NEAR(v, expected, 1e-12);
  }
  EXPECT_NEAR(SumExp(lp), 1.0, 1e-12);
}

// Agrees with a full `std::exp(double)` log-softmax to fp32-exp precision: the
// per-entry gap is `log(Z_double / Z_poly) ≈ ε_avg ≤ 2.4e-7` (the vector-exp
// polynomial's relative error, M7-T06), so 1e-5 (was 1e-9 when both used
// double exp).
TEST(LogprobsTest, LogSoftmaxMatchesDoubleReference) {
  const std::vector<float> row = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F, 3.5F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  const std::vector<double> ref = LogSoftmaxRef(row);
  ASSERT_EQ(lp.size(), ref.size());
  for (std::size_t i = 0; i < lp.size(); ++i) {
    EXPECT_NEAR(lp[i], ref[i], 1e-5) << "index " << i;
  }
}

// exp(lp[i]) recovers the softmax probability p_i = i/10 for logits = ln(i).
TEST(LogprobsTest, LogSoftmaxRecoversKnownProbabilities) {
  std::vector<float> row(4);
  for (std::size_t i = 0; i < row.size(); ++i) {
    // Runtime argument (not a literal) — logits = ln(1..4) so softmax = i/10.
    row[i] = std::log(static_cast<float>(i + 1));
  }
  std::vector<double> lp;
  LogSoftmax(row, lp);
  for (std::size_t i = 0; i < lp.size(); ++i) {
    const double expected_p = static_cast<double>(i + 1) / 10.0;
    EXPECT_NEAR(std::exp(lp[i]), expected_p, 1e-6) << "index " << i;
  }
}

// A large constant offset must not change the distribution (max-subtraction).
TEST(LogprobsTest, LogSoftmaxStableUnderLargeOffset) {
  const std::vector<float> base = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F};
  std::vector<float> shifted = base;
  for (float& x : shifted) {
    x += 1.0e4F;
  }
  std::vector<double> lp_base;
  std::vector<double> lp_shifted;
  LogSoftmax(base, lp_base);
  LogSoftmax(shifted, lp_shifted);
  EXPECT_NEAR(SumExp(lp_shifted), 1.0, 1e-5);  // fp32-exp class (M7-T06)
  for (std::size_t i = 0; i < lp_base.size(); ++i) {
    EXPECT_NEAR(lp_base[i], lp_shifted[i], 1e-4) << "index " << i;
  }
}

// Masked (`-inf`) logits map to `-inf` log-probs; the finite tail still
// normalizes to 1.
TEST(LogprobsTest, LogSoftmaxMaskedEntriesAreNegInf) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const std::vector<float> row = {2.0F, neg_inf, 1.0F, neg_inf, 0.0F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  EXPECT_TRUE(std::isinf(lp[1]) && lp[1] < 0.0);
  EXPECT_TRUE(std::isinf(lp[3]) && lp[3] < 0.0);
  EXPECT_TRUE(std::isfinite(lp[0]));
  EXPECT_NEAR(SumExp(lp), 1.0,
              1e-5);  // exp(-inf)==0 exactly; fp32-exp class (M7-T06)
}

// ---------------------------------------------------------- ExtractLogprobs --

// Full-sort reference for the top-N ordering (descending logprob, ascending
// id).
[[nodiscard]] std::vector<std::int32_t> TopIdsRef(const std::vector<double>& lp,
                                                  std::size_t n) {
  std::vector<std::int32_t> order(lp.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, [&](std::int32_t a, std::int32_t b) {
    const double la = lp[static_cast<std::size_t>(a)];
    const double lb = lp[static_cast<std::size_t>(b)];
    if (la != lb) {
      return la > lb;
    }
    return a < b;
  });
  std::vector<std::int32_t> top;
  for (std::size_t i = 0; i < order.size() && top.size() < n; ++i) {
    if (!std::isfinite(lp[static_cast<std::size_t>(order[i])])) {
      break;
    }
    top.push_back(order[i]);
  }
  return top;
}

TEST(LogprobsTest, ExtractLogprobsChosenLogprobIsChosenEntry) {
  const std::vector<float> row = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  const StepLogprobs out = ExtractLogprobs(lp, /*chosen=*/3, /*top_n=*/2);
  EXPECT_FLOAT_EQ(out.chosen_logprob, static_cast<float>(lp[3]));
}

TEST(LogprobsTest, ExtractLogprobsTopOrderingMatchesReference) {
  const std::vector<float> row = MakeRow(512, /*seed=*/9876);
  std::vector<double> lp;
  LogSoftmax(row, lp);
  for (const std::int32_t n : {1, 5, 20}) {
    const StepLogprobs out = ExtractLogprobs(lp, /*chosen=*/0, n);
    const std::vector<std::int32_t> ref =
        TopIdsRef(lp, static_cast<std::size_t>(n));
    ASSERT_EQ(out.top.size(), ref.size()) << "n=" << n;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      EXPECT_EQ(out.top[i].id, ref[i]) << "n=" << n << " rank " << i;
      EXPECT_FLOAT_EQ(out.top[i].logprob,
                      static_cast<float>(lp[static_cast<std::size_t>(ref[i])]))
          << "n=" << n << " rank " << i;
    }
    // Descending logprob order.
    for (std::size_t i = 1; i < out.top.size(); ++i) {
      EXPECT_GE(out.top[i - 1].logprob, out.top[i].logprob);
    }
  }
}

TEST(LogprobsTest, ExtractLogprobsBreaksTiesByAscendingId) {
  // Equal logits ⇒ equal log-probs; ties resolve to ascending id.
  const std::vector<float> row = {1.0F, 1.0F, 1.0F, 1.0F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  const StepLogprobs out = ExtractLogprobs(lp, /*chosen=*/0, /*top_n=*/3);
  ASSERT_EQ(out.top.size(), 3U);
  EXPECT_EQ(out.top[0].id, 0);
  EXPECT_EQ(out.top[1].id, 1);
  EXPECT_EQ(out.top[2].id, 2);
}

TEST(LogprobsTest, ExtractLogprobsExcludesMaskedTokens) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const std::vector<float> row = {2.0F, neg_inf, 1.0F, neg_inf, 0.0F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  // Ask for more than the 3 finite tokens; only finite ones appear.
  const StepLogprobs out = ExtractLogprobs(lp, /*chosen=*/0, /*top_n=*/5);
  ASSERT_EQ(out.top.size(), 3U);
  for (const TokenLogprob& t : out.top) {
    EXPECT_TRUE(std::isfinite(t.logprob));
    EXPECT_NE(t.id, 1);
    EXPECT_NE(t.id, 3);
  }
}

TEST(LogprobsTest, ExtractLogprobsClampsToVocab) {
  const std::vector<float> row = {2.0F, 1.0F, 0.0F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  const StepLogprobs out = ExtractLogprobs(lp, /*chosen=*/0, /*top_n=*/50);
  EXPECT_EQ(out.top.size(), 3U);  // capped at the vocabulary size
}

TEST(LogprobsTest, ExtractLogprobsZeroNKeepsChosenOnly) {
  const std::vector<float> row = {2.0F, 1.0F, 0.0F};
  std::vector<double> lp;
  LogSoftmax(row, lp);
  const StepLogprobs out = ExtractLogprobs(lp, /*chosen=*/1, /*top_n=*/0);
  EXPECT_TRUE(out.top.empty());
  EXPECT_FLOAT_EQ(out.chosen_logprob, static_cast<float>(lp[1]));
}

}  // namespace
