#include "core/logging.h"

#include <gtest/gtest.h>

#include <regex>
#include <sstream>
#include <string>

namespace {

using engine::core::GetLogLevel;
using engine::core::kMinLogLevel;
using engine::core::LogLevel;
using engine::core::SetLogLevel;
using engine::core::SetLogStreamForTesting;
using engine::core::SetModuleLogLevel;
using engine::core::detail::ApplyLogLevelSpec;
using engine::core::detail::ResetForTesting;

// Redirects log output to an in-memory stream and restores pristine filter
// state around every test. Runtime-filtering tests use LOG_INFO and above:
// those are compiled in under every supported ENGINE_MIN_LOG_LEVEL default
// (kDebug in Debug, kInfo in Release), so the tests behave identically in
// both build types.
class LoggingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetForTesting();
    SetLogStreamForTesting(&stream_);
  }

  void TearDown() override {
    SetLogStreamForTesting(nullptr);
    ResetForTesting();
  }

  [[nodiscard]] std::string Output() const { return stream_.str(); }

  [[nodiscard]] bool OutputContains(const std::string& needle) const {
    return Output().find(needle) != std::string::npos;
  }

  std::ostringstream stream_;
};

TEST_F(LoggingTest, DefaultGlobalLevelIsInfo) {
  EXPECT_EQ(GetLogLevel(), LogLevel::kInfo);
}

TEST_F(LoggingTest, GlobalLevelFiltersBySeverity) {
  SetLogLevel(LogLevel::kWarn);
  LOG_INFO("core", "suppressed-info {}", 1);
  LOG_WARN("core", "emitted-warn");
  LOG_ERROR("core", "emitted-error");
  EXPECT_FALSE(OutputContains("suppressed-info"));
  EXPECT_TRUE(OutputContains("emitted-warn"));
  EXPECT_TRUE(OutputContains("emitted-error"));
}

TEST_F(LoggingTest, LevelIsSwitchableAtRuntime) {
  SetLogLevel(LogLevel::kError);
  LOG_INFO("core", "before-switch");
  EXPECT_FALSE(OutputContains("before-switch"));

  SetLogLevel(LogLevel::kInfo);
  LOG_INFO("core", "after-switch");
  EXPECT_TRUE(OutputContains("after-switch"));
}

TEST_F(LoggingTest, OffSuppressesEverything) {
  SetLogLevel(LogLevel::kOff);
  LOG_ERROR("core", "silenced-error");
  EXPECT_EQ(Output(), "");
}

TEST_F(LoggingTest, ModuleOverrideLowersLevelBelowGlobal) {
  SetLogLevel(LogLevel::kError);
  SetModuleLogLevel("kvcache", LogLevel::kInfo);
  LOG_INFO("kvcache", "kvcache-info");
  LOG_INFO("scheduler", "scheduler-info");
  EXPECT_TRUE(OutputContains("kvcache-info"));
  EXPECT_FALSE(OutputContains("scheduler-info"));
}

TEST_F(LoggingTest, ModuleOverrideRaisesLevelAboveGlobal) {
  SetLogLevel(LogLevel::kInfo);
  SetModuleLogLevel("noisy", LogLevel::kError);
  LOG_INFO("noisy", "noisy-info");
  LOG_INFO("other", "other-info");
  EXPECT_FALSE(OutputContains("noisy-info"));
  EXPECT_TRUE(OutputContains("other-info"));
}

TEST_F(LoggingTest, MessageCarriesTimestampLevelModuleAndSourceLocation) {
  LOG_INFO("sampling", "value = {}", 42);
  // [2026-08-03 12:34:56.789] [info] [sampling] [logging_test.cpp:NN] ...
  const std::regex line_format(
      R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] )"
      R"(\[info\] \[sampling\] \[logging_test\.cpp:\d+\] value = 42)");
  EXPECT_TRUE(std::regex_search(Output(), line_format))
      << "log line did not match expected format: " << Output();
}

TEST_F(LoggingTest, ApplySpecSetsGlobalLevel) {
  EXPECT_TRUE(ApplyLogLevelSpec("error"));
  EXPECT_EQ(GetLogLevel(), LogLevel::kError);
}

TEST_F(LoggingTest, ApplySpecIsCaseInsensitiveAndTrimsWhitespace) {
  EXPECT_TRUE(ApplyLogLevelSpec("  WARN  "));
  EXPECT_EQ(GetLogLevel(), LogLevel::kWarn);
}

TEST_F(LoggingTest, ApplySpecSetsPerModuleOverrides) {
  EXPECT_TRUE(ApplyLogLevelSpec("error, kvcache=info"));
  EXPECT_EQ(GetLogLevel(), LogLevel::kError);
  LOG_INFO("kvcache", "kvcache-info");
  LOG_INFO("core", "core-info");
  EXPECT_TRUE(OutputContains("kvcache-info"));
  EXPECT_FALSE(OutputContains("core-info"));
}

TEST_F(LoggingTest, ApplySpecRejectsInvalidEntriesButAppliesValidOnes) {
  EXPECT_FALSE(ApplyLogLevelSpec("chatty,error"));
  EXPECT_EQ(GetLogLevel(), LogLevel::kError);
  // The invalid entry is reported on the "core" logger.
  EXPECT_TRUE(OutputContains("chatty"));
}

TEST_F(LoggingTest, ApplySpecLeavesLevelUntouchedOnInvalidInput) {
  EXPECT_FALSE(ApplyLogLevelSpec("verbose"));
  EXPECT_EQ(GetLogLevel(), LogLevel::kInfo);
}

TEST_F(LoggingTest, SuppressedMessageDoesNotEvaluateArguments) {
  SetLogLevel(LogLevel::kError);
  int evaluations = 0;
  LOG_INFO("core", "value {}", ++evaluations);
  EXPECT_EQ(evaluations, 0);
}

TEST_F(LoggingTest, DebugCompilesToNothingAboveCompileTimeFloor) {
  SetLogLevel(LogLevel::kDebug);
  int evaluations = 0;
  LOG_DEBUG("core", "debug-value {}", ++evaluations);
  if constexpr (LogLevel::kDebug >= kMinLogLevel) {
    // Debug builds: the statement is live and its arguments evaluate.
    EXPECT_EQ(evaluations, 1);
    EXPECT_TRUE(OutputContains("debug-value 1"));
  } else {
    // Release builds: stripped at compile time — no evaluation, no output.
    EXPECT_EQ(evaluations, 0);
    EXPECT_EQ(Output(), "");
  }
}

}  // namespace
