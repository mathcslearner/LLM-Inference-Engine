#include "core/check.h"

#include <gtest/gtest.h>

namespace {

TEST(CheckTest, TrueConditionIsANoOp) {
  int evaluations = 0;
  CHECK(++evaluations == 1);
  EXPECT_EQ(evaluations, 1);
}

TEST(CheckTest, MessageArgumentsNotEvaluatedOnSuccess) {
  int evaluations = 0;
  CHECK(true, "never formatted {}", ++evaluations);
  EXPECT_EQ(evaluations, 0);
}

TEST(CheckDeathTest, FalseConditionAborts) {
  EXPECT_DEATH(CHECK(1 == 2),
               "CHECK\\(1 == 2\\) failed at .*check_test\\.cpp:[0-9]+ in ");
}

TEST(CheckDeathTest, MessageIsFormattedIntoFailure) {
  const int index = 7;
  const int size = 4;
  EXPECT_DEATH(CHECK(index < size, "index {} out of {}", index, size),
               "CHECK\\(index < size\\) failed at .*: index 7 out of 4");
}

#ifdef NDEBUG
TEST(DcheckTest, CompilesToNothingInRelease) {
  int evaluations = 0;
  DCHECK(++evaluations == 1);
  DCHECK(false, "never formatted {}", ++evaluations);
  EXPECT_EQ(evaluations, 0);
}
#else
TEST(DcheckTest, EvaluatesConditionInDebug) {
  int evaluations = 0;
  DCHECK(++evaluations == 1);
  EXPECT_EQ(evaluations, 1);
}

TEST(DcheckDeathTest, FalseConditionAbortsInDebug) {
  EXPECT_DEATH(DCHECK(2 < 1, "message {}", 42),
               "DCHECK\\(2 < 1\\) failed at .*: message 42");
}
#endif

}  // namespace
