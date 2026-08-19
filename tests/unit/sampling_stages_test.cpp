#include "core/status.h"
#include "sampling/stages.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

// M7-T02 (design: docs/design/model-execution.md §15.2): exact-behaviour tests
// for the individual stochastic-sampling stages, the acceptance criterion
// "top-k/top-p masks verified exactly". Portable C++ — no ISA concern, no
// SCALAR_PASS. Statistical (chi-square) coverage of the assembled pipeline
// lives in sampler_test.cpp.

namespace {

using engine::core::IsInternal;
using engine::sampling::detail::ApplyTemperature;
using engine::sampling::detail::ApplyTopK;
using engine::sampling::detail::ApplyTopP;
using engine::sampling::detail::CheckFinite;
using engine::sampling::detail::SelectByCdf;
using engine::sampling::detail::Softmax;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

[[nodiscard]] bool IsNegInf(float x) { return std::isinf(x) && x < 0.0F; }

// ----------------------------------------------------------- CheckFinite --

TEST(StagesTest, CheckFiniteAcceptsNormalRow) {
  const std::vector<float> logits = {1.0F, -2.0F, 3.5F};
  EXPECT_TRUE(CheckFinite(logits).ok());
}

TEST(StagesTest, CheckFiniteAcceptsMaskedRow) {
  // -inf entries (masked tokens) are fine as long as one finite logit remains.
  const std::vector<float> logits = {kNegInf, 2.0F, kNegInf};
  EXPECT_TRUE(CheckFinite(logits).ok());
}

TEST(StagesTest, CheckFiniteRejectsNaN) {
  const std::vector<float> logits = {1.0F,
                                     std::numeric_limits<float>::quiet_NaN()};
  EXPECT_TRUE(IsInternal(CheckFinite(logits)));
}

TEST(StagesTest, CheckFiniteRejectsPositiveInf) {
  const std::vector<float> logits = {1.0F, kPosInf};
  EXPECT_TRUE(IsInternal(CheckFinite(logits)));
}

TEST(StagesTest, CheckFiniteRejectsAllNegInf) {
  const std::vector<float> logits = {kNegInf, kNegInf};
  EXPECT_TRUE(IsInternal(CheckFinite(logits)));
}

// -------------------------------------------------------- ApplyTemperature --

TEST(StagesTest, TemperatureScalesByReciprocal) {
  std::vector<float> logits = {2.0F, -4.0F, 1.0F};
  ApplyTemperature(logits, 2.0F);
  EXPECT_FLOAT_EQ(logits[0], 1.0F);
  EXPECT_FLOAT_EQ(logits[1], -2.0F);
  EXPECT_FLOAT_EQ(logits[2], 0.5F);
}

// -------------------------------------------------------------- ApplyTopK --

TEST(StagesTest, TopKKeepsHighestMasksRest) {
  std::vector<float> logits = {1.0F, 3.0F, 2.0F, 5.0F, 4.0F};
  ApplyTopK(logits, 2);
  // Keeps the two largest (indices 3, 4); the rest go to -inf.
  EXPECT_TRUE(IsNegInf(logits[0]));
  EXPECT_TRUE(IsNegInf(logits[1]));
  EXPECT_TRUE(IsNegInf(logits[2]));
  EXPECT_FLOAT_EQ(logits[3], 5.0F);
  EXPECT_FLOAT_EQ(logits[4], 4.0F);
}

TEST(StagesTest, TopKThresholdTiesAreAllKept) {
  // k = 1 but two tokens share the maximum: the threshold (5) keeps both.
  std::vector<float> logits = {5.0F, 5.0F, 3.0F, 1.0F};
  ApplyTopK(logits, 1);
  EXPECT_FLOAT_EQ(logits[0], 5.0F);
  EXPECT_FLOAT_EQ(logits[1], 5.0F);
  EXPECT_TRUE(IsNegInf(logits[2]));
  EXPECT_TRUE(IsNegInf(logits[3]));
}

TEST(StagesTest, TopKDisabledAndOversizedAreNoOps) {
  const std::vector<float> original = {1.0F, 3.0F, 2.0F};
  for (const std::int32_t k : {0, 3, 100}) {
    std::vector<float> logits = original;
    ApplyTopK(logits, k);
    EXPECT_EQ(logits, original) << "k=" << k;
  }
}

// --------------------------------------------------------------- Softmax --

TEST(StagesTest, SoftmaxNormalizesAndMapsNegInfToZero) {
  const std::vector<float> logits = {0.0F, kNegInf, 0.0F};
  std::vector<double> probs;
  Softmax(logits, probs);
  ASSERT_EQ(probs.size(), 3U);
  EXPECT_DOUBLE_EQ(probs[0], 0.5);
  EXPECT_DOUBLE_EQ(probs[1], 0.0);  // masked token: exactly zero
  EXPECT_DOUBLE_EQ(probs[2], 0.5);
}

TEST(StagesTest, SoftmaxSumsToOne) {
  const std::vector<float> logits = {2.0F, 1.0F, -0.5F, 3.0F, 0.0F};
  std::vector<double> probs;
  Softmax(logits, probs);
  double sum = 0.0;
  for (const double p : probs) {
    sum += p;
  }
  EXPECT_NEAR(sum, 1.0, 1e-12);
}

// -------------------------------------------------------------- ApplyTopP --

TEST(StagesTest, TopPKeepsCrossingTokenInclusive) {
  std::vector<double> probs = {0.5, 0.3, 0.15, 0.05};
  ApplyTopP(probs, 0.6F);
  // 0.5 < 0.6, +0.3 = 0.8 >= 0.6 -> keep the first two, zero the rest.
  EXPECT_DOUBLE_EQ(probs[0], 0.5);
  EXPECT_DOUBLE_EQ(probs[1], 0.3);
  EXPECT_DOUBLE_EQ(probs[2], 0.0);
  EXPECT_DOUBLE_EQ(probs[3], 0.0);
}

TEST(StagesTest, TopPTinyThresholdKeepsOnlyArgmax) {
  std::vector<double> probs = {0.5, 0.3, 0.2};
  ApplyTopP(probs, 0.1F);
  EXPECT_DOUBLE_EQ(probs[0], 0.5);
  EXPECT_DOUBLE_EQ(probs[1], 0.0);
  EXPECT_DOUBLE_EQ(probs[2], 0.0);
}

TEST(StagesTest, TopPOneIsNoOp) {
  const std::vector<double> original = {0.5, 0.3, 0.15, 0.05};
  std::vector<double> probs = original;
  ApplyTopP(probs, 1.0F);
  EXPECT_EQ(probs, original);
}

TEST(StagesTest, TopPBreaksTiesByAscendingIndex) {
  // Two tokens tie at 0.4; the descending sort visits index 0 before index 1.
  std::vector<double> probs = {0.4, 0.4, 0.2};
  ApplyTopP(probs, 0.5F);
  // 0.4 < 0.5, +0.4 = 0.8 >= 0.5 -> keep indices 0 and 1, zero index 2.
  EXPECT_DOUBLE_EQ(probs[0], 0.4);
  EXPECT_DOUBLE_EQ(probs[1], 0.4);
  EXPECT_DOUBLE_EQ(probs[2], 0.0);
}

// ------------------------------------------------------------- SelectByCdf --

TEST(StagesTest, SelectByCdfWalksCumulativeMass) {
  const std::vector<double> probs = {0.2, 0.5, 0.3};
  EXPECT_EQ(SelectByCdf(probs, 0.0), 0);   // first positive index
  EXPECT_EQ(SelectByCdf(probs, 0.1), 0);   // within [0, 0.2)
  EXPECT_EQ(SelectByCdf(probs, 0.3), 1);   // within [0.2, 0.7)
  EXPECT_EQ(SelectByCdf(probs, 0.75), 2);  // within [0.7, 1.0)
}

TEST(StagesTest, SelectByCdfNeverPicksMaskedToken) {
  const std::vector<double> probs = {0.5, 0.0, 0.5};  // index 1 masked
  EXPECT_EQ(SelectByCdf(probs, 0.4), 0);
  EXPECT_EQ(SelectByCdf(probs, 0.6), 2);  // skips the zero-mass index 1
}

TEST(StagesTest, SelectByCdfClampsAtTopOfMass) {
  const std::vector<double> probs = {0.5, 0.0, 0.5};
  // u just below 1: falls through to the last positive index.
  EXPECT_EQ(SelectByCdf(probs, 0.999999999), 2);
}

}  // namespace
