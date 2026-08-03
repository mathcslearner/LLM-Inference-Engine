#pragma once

#include <fmt/format.h>

#include <string_view>

// Fatal assertion macros (M0-T07).
//
// Policy (ADR-003, M0-T08): CHECK is for programmer errors — invariants that
// cannot fail in a correct program (out-of-range internal index, violated
// precondition between our own modules). A CHECK failure prints the failed
// condition with its source location and aborts the process; it is never an
// error-handling strategy. Conditions that *can* legitimately fail at runtime
// (bad user input, missing file, exhausted memory) are recoverable errors and
// must return Status/StatusOr (see core/status.h) instead.
//
// Usage:
//
//   CHECK(block_index < pool_size_);
//   CHECK(block_index < pool_size_, "block {} out of {}", block_index,
//         pool_size_);
//
// The optional message is a fmt format literal plus arguments, checked at
// compile time (like LOG_*); it is only formatted when the check fails. The
// format string must be a string literal.
//
// DCHECK is CHECK in Debug builds; under NDEBUG it compiles to nothing — the
// condition and message arguments are type-checked but never evaluated. Use
// it for invariants too hot to verify in Release (per-token loops, kernels).
//
// CHECK_OK / DCHECK_OK live in core/status.h (they need the Status type).

namespace engine::core::detail {

// Prints "<condition> failed at <file>:<line> in <function>[: <message>]" to
// stderr and aborts. Deliberately independent of the logging subsystem so a
// CHECK works during static initialization and inside logging itself.
[[noreturn]] void CheckFailed(const char* condition, const char* file, int line,
                              const char* function, std::string_view message);

}  // namespace engine::core::detail

// Implementation detail shared by CHECK/DCHECK (and CHECK_OK in status.h).
// The message is formatted only on failure; `"" __VA_ARGS__` collapses to ""
// when no message is given and concatenates with the format literal when one
// is (which is why the format string must be a literal).
#define ENGINE_CHECK_IMPL(cond_, cond_text_, ...)                         \
  do {                                                                    \
    if (!(cond_)) [[unlikely]] {                                          \
      ::engine::core::detail::CheckFailed(cond_text_, __FILE__, __LINE__, \
                                          __func__,                       \
                                          ::fmt::format("" __VA_ARGS__)); \
    }                                                                     \
  } while (0)

// Fatal assertion: aborts with source location if `cond_` is false.
#define CHECK(cond_, ...) \
  ENGINE_CHECK_IMPL((cond_), "CHECK(" #cond_ ")" __VA_OPT__(, ) __VA_ARGS__)

// Debug-only CHECK. Under NDEBUG the `if (false)` keeps the condition and
// arguments type-checked (no unused-variable warnings) but never evaluated,
// and the whole statement compiles to nothing.
#ifdef NDEBUG
#define DCHECK(cond_, ...)                                                \
  do {                                                                    \
    if (false) {                                                          \
      ENGINE_CHECK_IMPL((cond_),                                          \
                        "DCHECK(" #cond_ ")" __VA_OPT__(, ) __VA_ARGS__); \
    }                                                                     \
  } while (0)
#else
#define DCHECK(cond_, ...) \
  ENGINE_CHECK_IMPL((cond_), "DCHECK(" #cond_ ")" __VA_OPT__(, ) __VA_ARGS__)
#endif
