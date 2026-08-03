#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <iosfwd>
#include <string_view>

// Logging subsystem (M0-T06): a thin wrapper over spdlog.
//
// Policy: this header is the project's entire logging surface. No other file
// may include or use spdlog directly — spdlog is an implementation detail of
// logging.cpp and can be swapped without touching callers.
//
// Usage:
//
//   LOG_INFO("kvcache", "allocated {} blocks in {} ms", n_blocks, elapsed);
//
// The first argument is the module name (a stable, lowercase identifier —
// by convention the module's directory name under src/). Each module name
// gets its own named logger, so output is attributable and levels can be
// tuned per module. The remaining arguments are a fmt format string and its
// values; format strings are checked at compile time.
//
// Every message carries a timestamp, level, module name, and source
// location:
//
//   [2026-08-03 12:34:56.789] [info] [kvcache] [block_pool.cpp:42] ...
//
// Filtering happens at two layers:
//
//   - Compile time: statements below `kMinLogLevel` compile to nothing —
//     arguments are not evaluated and no code is emitted. The floor is
//     kDebug in Debug builds and kInfo in Release; override it with
//     -DENGINE_MIN_LOG_LEVEL=ENGINE_LOG_LEVEL_<LEVEL>.
//   - Run time: a global level (SetLogLevel) with optional per-module
//     overrides (SetModuleLogLevel). Suppressed statements cost one atomic
//     load; the message is only formatted when it will actually be emitted.
//
// The environment variable ENGINE_LOG_LEVEL is read once, at first use of
// any logging API. It holds a comma-separated list of `level` (global) and
// `module=level` entries, e.g. `ENGINE_LOG_LEVEL="warn,kvcache=debug"`.
// Levels: debug, info, warn, error, off (case-insensitive). Programmatic
// calls made afterwards override it; invalid entries are ignored with a
// warning.

// Numeric level values. These must be macros, not enumerators: their job is
// to be usable on the compiler command line, where the preprocessor expands
// -DENGINE_MIN_LOG_LEVEL=ENGINE_LOG_LEVEL_WARN. Must stay in
// ascending-severity order; LogLevel mirrors them.
// NOLINTBEGIN(modernize-macro-to-enum)
#define ENGINE_LOG_LEVEL_DEBUG 0
#define ENGINE_LOG_LEVEL_INFO 1
#define ENGINE_LOG_LEVEL_WARN 2
#define ENGINE_LOG_LEVEL_ERROR 3
#define ENGINE_LOG_LEVEL_OFF 4
// NOLINTEND(modernize-macro-to-enum)

#ifndef ENGINE_MIN_LOG_LEVEL
#ifdef NDEBUG
#define ENGINE_MIN_LOG_LEVEL ENGINE_LOG_LEVEL_INFO
#else
#define ENGINE_MIN_LOG_LEVEL ENGINE_LOG_LEVEL_DEBUG
#endif
#endif

namespace engine::core {

// Severity levels, ascending. kOff is not a message level — it is only a
// filter setting that suppresses everything.
enum class LogLevel : std::uint8_t {
  kDebug = ENGINE_LOG_LEVEL_DEBUG,
  kInfo = ENGINE_LOG_LEVEL_INFO,
  kWarn = ENGINE_LOG_LEVEL_WARN,
  kError = ENGINE_LOG_LEVEL_ERROR,
  kOff = ENGINE_LOG_LEVEL_OFF,
};

// Compile-time floor: LOG_* statements below this level are removed entirely
// (see ENGINE_MIN_LOG_LEVEL above).
inline constexpr LogLevel kMinLogLevel =
    static_cast<LogLevel>(ENGINE_MIN_LOG_LEVEL);

// Sets the global runtime log level. Thread-safe; takes effect immediately.
void SetLogLevel(LogLevel level);

// Returns the current global runtime log level (default: kInfo).
[[nodiscard]] LogLevel GetLogLevel();

// Overrides the runtime level for one module, in either direction: a module
// can be made more verbose or quieter than the global level. Thread-safe.
void SetModuleLogLevel(std::string_view module, LogLevel level);

// Test-only: redirects all log output to `stream` (unformatted-by-terminal,
// flushed per message); pass nullptr to restore the default stderr sink.
// Not for production use.
void SetLogStreamForTesting(std::ostream* stream);

namespace detail {

// Call-site location captured by the LOG_* macros.
struct SourceLocation {
  const char* file;
  int line;
  const char* function;
};

// Runtime filter check. Cheap when the message is suppressed: one relaxed
// atomic load unless per-module overrides are in play.
[[nodiscard]] bool ShouldLog(std::string_view module, LogLevel level);

// Emits an already-formatted message through the module's logger.
void LogMessage(std::string_view module, LogLevel level, SourceLocation loc,
                std::string_view message);

// Applies an ENGINE_LOG_LEVEL-syntax spec (see file comment). Returns false
// if any entry was invalid (valid entries are still applied). Used for the
// environment variable and directly testable.
bool ApplyLogLevelSpec(std::string_view spec);

// Test-only: restores the default global level and clears all per-module
// overrides. Does not touch the sink.
void ResetForTesting();

}  // namespace detail
}  // namespace engine::core

// Implementation detail shared by the LOG_* macros below. `do { } while (0)`
// makes the macro a single statement; the `if constexpr` removes the whole
// body — including argument evaluation — when `level_` is below the
// compile-time floor.
#define ENGINE_LOG_IMPL(level_, module_, ...)                              \
  do {                                                                     \
    if constexpr ((level_) >= ::engine::core::kMinLogLevel) {              \
      const ::std::string_view engine_log_module_view_ = (module_);        \
      if (::engine::core::detail::ShouldLog(engine_log_module_view_,       \
                                            (level_))) {                   \
        ::engine::core::detail::LogMessage(                                \
            engine_log_module_view_, (level_),                             \
            ::engine::core::detail::SourceLocation{                        \
                .file = __FILE__, .line = __LINE__, .function = __func__}, \
            ::fmt::format(__VA_ARGS__));                                   \
      }                                                                    \
    }                                                                      \
  } while (0)

// Leveled logging macros: LOG_<LEVEL>(module, format, args...).
#define LOG_DEBUG(module_, ...) \
  ENGINE_LOG_IMPL(::engine::core::LogLevel::kDebug, module_, __VA_ARGS__)
#define LOG_INFO(module_, ...) \
  ENGINE_LOG_IMPL(::engine::core::LogLevel::kInfo, module_, __VA_ARGS__)
#define LOG_WARN(module_, ...) \
  ENGINE_LOG_IMPL(::engine::core::LogLevel::kWarn, module_, __VA_ARGS__)
#define LOG_ERROR(module_, ...) \
  ENGINE_LOG_IMPL(::engine::core::LogLevel::kError, module_, __VA_ARGS__)
