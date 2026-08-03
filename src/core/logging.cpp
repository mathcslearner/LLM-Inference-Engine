#include "core/logging.h"

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace engine::core {
namespace {

constexpr LogLevel kDefaultLevel = LogLevel::kInfo;

// timestamp · level (colored on terminals) · module · file:line · message.
constexpr const char* kPattern =
    "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] [%s:%#] %v";

spdlog::level::level_enum ToSpdlogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return spdlog::level::debug;
    case LogLevel::kInfo:
      return spdlog::level::info;
    case LogLevel::kWarn:
      return spdlog::level::warn;
    case LogLevel::kError:
      return spdlog::level::err;
    case LogLevel::kOff:
      return spdlog::level::off;
  }
  return spdlog::level::info;  // Unreachable; the switch is exhaustive.
}

std::string_view TrimWhitespace(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  return text;
}

std::optional<LogLevel> ParseLogLevel(std::string_view token) {
  std::string lower(token);
  std::ranges::transform(lower, lower.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  if (lower == "debug") {
    return LogLevel::kDebug;
  }
  if (lower == "info") {
    return LogLevel::kInfo;
  }
  if (lower == "warn" || lower == "warning") {
    return LogLevel::kWarn;
  }
  if (lower == "error" || lower == "err") {
    return LogLevel::kError;
  }
  if (lower == "off") {
    return LogLevel::kOff;
  }
  return std::nullopt;
}

// Owns the sink, the per-module loggers, and the runtime filter state.
// Deliberately leaked (never destroyed) so logging stays valid during static
// destruction; all state behind Instance() so the ENGINE_LOG_LEVEL read
// happens exactly once, at first use of any logging API.
class LogRegistry {
 public:
  static LogRegistry& Instance() {
    static auto* instance = new LogRegistry();  // Intentionally leaked.
    return *instance;
  }

  bool ShouldLog(std::string_view module, LogLevel level) {
    if (level < min_effective_level_.load(std::memory_order_relaxed)) {
      return false;
    }
    if (!has_module_overrides_.load(std::memory_order_relaxed)) {
      // No overrides: the min effective level IS the global level, and this
      // message is at or above it.
      return true;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = module_levels_.find(module);
    const LogLevel effective =
        it == module_levels_.end()
            ? global_level_.load(std::memory_order_relaxed)
            : it->second;
    return level >= effective;
  }

  void Log(std::string_view module, LogLevel level, detail::SourceLocation loc,
           std::string_view message) {
    std::shared_ptr<spdlog::logger> logger;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      auto it = loggers_.find(module);
      if (it == loggers_.end()) {
        auto created =
            std::make_shared<spdlog::logger>(std::string(module), sink_);
        // Filtering is done by ShouldLog; the spdlog logger passes
        // everything through.
        created->set_level(spdlog::level::trace);
        it = loggers_.emplace(std::string(module), std::move(created)).first;
      }
      logger = it->second;
    }
    logger->log(spdlog::source_loc{loc.file, loc.line, loc.function},
                ToSpdlogLevel(level),
                spdlog::string_view_t(message.data(), message.size()));
  }

  void SetGlobalLevel(LogLevel level) {
    const std::lock_guard<std::mutex> lock(mutex_);
    global_level_.store(level, std::memory_order_relaxed);
    RecomputeMinEffectiveLevelLocked();
  }

  [[nodiscard]] LogLevel global_level() const {
    return global_level_.load(std::memory_order_relaxed);
  }

  void SetModuleLevel(std::string_view module, LogLevel level) {
    const std::lock_guard<std::mutex> lock(mutex_);
    module_levels_.insert_or_assign(std::string(module), level);
    RecomputeMinEffectiveLevelLocked();
  }

  bool ApplySpec(std::string_view spec) {
    bool all_valid = true;
    while (!spec.empty()) {
      const auto comma = spec.find(',');
      const std::string_view entry = TrimWhitespace(spec.substr(0, comma));
      spec = comma == std::string_view::npos ? std::string_view{}
                                             : spec.substr(comma + 1);
      if (entry.empty()) {
        continue;
      }
      const auto equals = entry.find('=');
      if (equals == std::string_view::npos) {
        const auto level = ParseLogLevel(entry);
        if (level.has_value()) {
          SetGlobalLevel(*level);
          continue;
        }
      } else {
        const std::string_view module = TrimWhitespace(entry.substr(0, equals));
        const auto level =
            ParseLogLevel(TrimWhitespace(entry.substr(equals + 1)));
        if (!module.empty() && level.has_value()) {
          SetModuleLevel(module, *level);
          continue;
        }
      }
      all_valid = false;
      if (ShouldLog("core", LogLevel::kWarn)) {
        Log("core", LogLevel::kWarn,
            detail::SourceLocation{
                .file = __FILE__, .line = __LINE__, .function = __func__},
            fmt::format("ignoring invalid log level entry \"{}\" "
                        "(expected \"level\" or \"module=level\")",
                        entry));
      }
    }
    return all_valid;
  }

  void SetStreamForTesting(std::ostream* stream) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stream != nullptr) {
      sink_ = std::make_shared<spdlog::sinks::ostream_sink_mt>(
          *stream, /*force_flush=*/true);
    } else {
      sink_ = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    }
    sink_->set_pattern(kPattern);
    for (auto& entry : loggers_) {
      entry.second->sinks() = {sink_};
    }
  }

  void ResetForTesting() {
    const std::lock_guard<std::mutex> lock(mutex_);
    global_level_.store(kDefaultLevel, std::memory_order_relaxed);
    module_levels_.clear();
    RecomputeMinEffectiveLevelLocked();
  }

 private:
  LogRegistry()
      : sink_(std::make_shared<spdlog::sinks::stderr_color_sink_mt>()) {
    sink_->set_pattern(kPattern);
    // Reading the environment here (not lazily per call) keeps precedence
    // simple: the env var seeds the levels, later programmatic calls win.
    const char* spec = std::getenv("ENGINE_LOG_LEVEL");
    if (spec != nullptr) {
      ApplySpec(spec);
    }
  }

  // Caches min(global, all module overrides) so ShouldLog can reject
  // suppressed messages with one atomic load. Callers hold mutex_.
  void RecomputeMinEffectiveLevelLocked() {
    LogLevel min_level = global_level_.load(std::memory_order_relaxed);
    for (const auto& entry : module_levels_) {
      min_level = std::min(min_level, entry.second);
    }
    min_effective_level_.store(min_level, std::memory_order_relaxed);
    has_module_overrides_.store(!module_levels_.empty(),
                                std::memory_order_relaxed);
  }

  std::atomic<LogLevel> global_level_{kDefaultLevel};
  std::atomic<LogLevel> min_effective_level_{kDefaultLevel};
  std::atomic<bool> has_module_overrides_{false};

  std::mutex mutex_;
  std::shared_ptr<spdlog::sinks::sink> sink_;
  // std::less<> enables lookup by string_view without allocating.
  std::map<std::string, LogLevel, std::less<>> module_levels_;
  std::map<std::string, std::shared_ptr<spdlog::logger>, std::less<>> loggers_;
};

}  // namespace

void SetLogLevel(LogLevel level) {
  LogRegistry::Instance().SetGlobalLevel(level);
}

LogLevel GetLogLevel() { return LogRegistry::Instance().global_level(); }

void SetModuleLogLevel(std::string_view module, LogLevel level) {
  LogRegistry::Instance().SetModuleLevel(module, level);
}

void SetLogStreamForTesting(std::ostream* stream) {
  LogRegistry::Instance().SetStreamForTesting(stream);
}

namespace detail {

bool ShouldLog(std::string_view module, LogLevel level) {
  return LogRegistry::Instance().ShouldLog(module, level);
}

void LogMessage(std::string_view module, LogLevel level, SourceLocation loc,
                std::string_view message) {
  LogRegistry::Instance().Log(module, level, loc, message);
}

bool ApplyLogLevelSpec(std::string_view spec) {
  return LogRegistry::Instance().ApplySpec(spec);
}

void ResetForTesting() { LogRegistry::Instance().ResetForTesting(); }

}  // namespace detail
}  // namespace engine::core
