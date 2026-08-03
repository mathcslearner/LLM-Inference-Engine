#include "common/paths.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace {

TEST(UnitSmoke, GoogleTestRuns) { EXPECT_EQ(2 + 2, 4); }

// Exercises the harness plumbing end to end: the shared helper library links
// and its injected fixtures path points at the real source-tree directory.
TEST(UnitSmoke, FixturesDirExists) {
  const std::filesystem::path dir = engine::testing::FixturesDir();
  EXPECT_TRUE(std::filesystem::exists(dir)) << dir;
  EXPECT_TRUE(std::filesystem::is_directory(dir)) << dir;
}

}  // namespace
