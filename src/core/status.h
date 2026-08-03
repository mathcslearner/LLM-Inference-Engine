#pragma once

#include "core/check.h"

#include <fmt/format.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

// Error-handling primitives (M0-T07).
//
// Policy (ADR-003, M0-T08):
//
//   - Recoverable errors — anything that can legitimately fail at runtime
//     (bad request parameters, missing files, exhausted memory or KV blocks)
//     — are returned as `Status`, or `StatusOr<T>` when the operation also
//     produces a value. Callers propagate them with RETURN_IF_ERROR /
//     ASSIGN_OR_RETURN or handle them explicitly.
//   - Programmer errors — invariants that cannot fail in a correct program —
//     use CHECK/DCHECK (core/check.h), which abort. Never CHECK on input.
//   - Exceptions are not used across module boundaries. Third-party code may
//     throw internally, but every engine module's public API is
//     exception-free: it reports failure exclusively through Status.
//
// Usage:
//
//   StatusOr<Config> LoadConfig(std::string_view path);
//
//   Status Init(std::string_view path) {
//     ASSIGN_OR_RETURN(Config config, LoadConfig(path));
//     RETURN_IF_ERROR(Validate(config));
//     return OkStatus();
//   }
//
// Error construction composes messages with fmt (compile-time checked):
//
//   return NotFoundError("no weight named {}", name);
//
// The factory format string must be a literal; for messages built at
// runtime, construct `Status(StatusCode::kNotFound, runtime_string)`.

namespace engine::core {

// Error-code taxonomy. Values are stable — they will eventually cross API
// and process boundaries (server responses, metrics), so append new codes at
// the end and never renumber.
enum class StatusCode : std::uint8_t {
  kOk = 0,
  kCancelled = 1,           // Operation cancelled, typically by the caller.
  kInvalidArgument = 2,     // Caller-supplied argument is malformed.
  kNotFound = 3,            // Requested entity does not exist.
  kAlreadyExists = 4,       // Entity the caller tried to create exists.
  kFailedPrecondition = 5,  // System not in a state required for the call.
  kOutOfRange = 6,          // Argument outside the valid range.
  kResourceExhausted = 7,   // A quota or pool (e.g. KV blocks) is exhausted.
  kOutOfMemory = 8,         // Host or device memory allocation failed.
  kUnimplemented = 9,       // Operation not implemented / not supported.
  kUnavailable = 10,        // Transient failure; retrying may succeed.
  kInternal = 11,           // Invariant broken; not caller-recoverable.
};

// Canonical upper-snake name, e.g. "INVALID_ARGUMENT".
[[nodiscard]] std::string_view StatusCodeToString(StatusCode code);

// A Status is either OK or an (error code, message) pair. It is cheap to
// copy and move; an OK status carries no message (a message passed with
// kOk is discarded, so all OK statuses compare equal). [[nodiscard]]: a
// dropped Status is a swallowed error.
class [[nodiscard]] Status {
 public:
  // Constructs an OK status.
  Status() = default;

  Status(StatusCode code, std::string_view message)
      : code_(code),
        message_(code == StatusCode::kOk ? std::string_view{} : message) {}

  [[nodiscard]] bool ok() const { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] std::string_view message() const { return message_; }

  // "OK", or "<CODE>: <message>", e.g. "NOT_FOUND: no weight named wq".
  [[nodiscard]] std::string ToString() const;

  friend bool operator==(const Status&, const Status&) = default;

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

// Streams ToString(); makes gtest failure output readable.
std::ostream& operator<<(std::ostream& os, const Status& status);

[[nodiscard]] inline Status OkStatus() { return {}; }

// Per-code factories. The format string is a fmt literal, checked at compile
// time; use the Status constructor directly for runtime-built messages.
template <typename... Args>
[[nodiscard]] Status CancelledError(fmt::format_string<Args...> format,
                                    Args&&... args) {
  return Status(StatusCode::kCancelled,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status InvalidArgumentError(fmt::format_string<Args...> format,
                                          Args&&... args) {
  return Status(StatusCode::kInvalidArgument,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status NotFoundError(fmt::format_string<Args...> format,
                                   Args&&... args) {
  return Status(StatusCode::kNotFound,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status AlreadyExistsError(fmt::format_string<Args...> format,
                                        Args&&... args) {
  return Status(StatusCode::kAlreadyExists,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status FailedPreconditionError(fmt::format_string<Args...> format,
                                             Args&&... args) {
  return Status(StatusCode::kFailedPrecondition,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status OutOfRangeError(fmt::format_string<Args...> format,
                                     Args&&... args) {
  return Status(StatusCode::kOutOfRange,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status ResourceExhaustedError(fmt::format_string<Args...> format,
                                            Args&&... args) {
  return Status(StatusCode::kResourceExhausted,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status OutOfMemoryError(fmt::format_string<Args...> format,
                                      Args&&... args) {
  return Status(StatusCode::kOutOfMemory,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status UnimplementedError(fmt::format_string<Args...> format,
                                        Args&&... args) {
  return Status(StatusCode::kUnimplemented,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status UnavailableError(fmt::format_string<Args...> format,
                                      Args&&... args) {
  return Status(StatusCode::kUnavailable,
                fmt::format(format, std::forward<Args>(args)...));
}
template <typename... Args>
[[nodiscard]] Status InternalError(fmt::format_string<Args...> format,
                                   Args&&... args) {
  return Status(StatusCode::kInternal,
                fmt::format(format, std::forward<Args>(args)...));
}

// Per-code predicates, mirroring the factories.
[[nodiscard]] inline bool IsCancelled(const Status& s) {
  return s.code() == StatusCode::kCancelled;
}
[[nodiscard]] inline bool IsInvalidArgument(const Status& s) {
  return s.code() == StatusCode::kInvalidArgument;
}
[[nodiscard]] inline bool IsNotFound(const Status& s) {
  return s.code() == StatusCode::kNotFound;
}
[[nodiscard]] inline bool IsAlreadyExists(const Status& s) {
  return s.code() == StatusCode::kAlreadyExists;
}
[[nodiscard]] inline bool IsFailedPrecondition(const Status& s) {
  return s.code() == StatusCode::kFailedPrecondition;
}
[[nodiscard]] inline bool IsOutOfRange(const Status& s) {
  return s.code() == StatusCode::kOutOfRange;
}
[[nodiscard]] inline bool IsResourceExhausted(const Status& s) {
  return s.code() == StatusCode::kResourceExhausted;
}
[[nodiscard]] inline bool IsOutOfMemory(const Status& s) {
  return s.code() == StatusCode::kOutOfMemory;
}
[[nodiscard]] inline bool IsUnimplemented(const Status& s) {
  return s.code() == StatusCode::kUnimplemented;
}
[[nodiscard]] inline bool IsUnavailable(const Status& s) {
  return s.code() == StatusCode::kUnavailable;
}
[[nodiscard]] inline bool IsInternal(const Status& s) {
  return s.code() == StatusCode::kInternal;
}

// A StatusOr<T> holds exactly one of: a value of type T, or a non-OK Status
// explaining why there is no value. There is no empty state — construct it
// from a value or from an error (constructing from an OK status is a CHECK
// failure).
//
// Access: `ok()` then `*so` / `so->member` (DCHECK'd), or `so.value()`
// (CHECK'd). Move the value out of an rvalue: `T v = std::move(so).value();`.
//
// Copyable iff T is copyable; always movable for movable T (including
// move-only types like std::unique_ptr). A moved-from StatusOr is valid but
// unspecified, like any moved-from value.
template <typename T>
class [[nodiscard]] StatusOr {
  static_assert(!std::is_same_v<std::remove_cvref_t<T>, Status>,
                "StatusOr<Status> is meaningless; use Status");
  static_assert(!std::is_reference_v<T>,
                "StatusOr<T&> is unsupported; use a pointer or "
                "std::reference_wrapper");

 public:
  StatusOr() = delete;  // No empty state: value or error, always.

  // Implicit from a non-OK Status, so `return NotFoundError(...)` works in a
  // function returning StatusOr<T>.
  // NOLINTNEXTLINE(google-explicit-constructor)
  StatusOr(Status status) : rep_(std::in_place_index<0>, std::move(status)) {
    CHECK(!std::get<0>(rep_).ok(),
          "StatusOr must not be constructed from an OK status");
  }

  // Implicit from anything T is implicitly constructible from, so
  // `return "name";` works in a function returning StatusOr<std::string>.
  template <typename U = T>
    requires std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T> &&
             (!std::is_same_v<std::remove_cvref_t<U>, Status>) &&
             (!std::is_same_v<std::remove_cvref_t<U>, StatusOr>)
  // NOLINTNEXTLINE(google-explicit-constructor)
  StatusOr(U&& value) : rep_(std::in_place_index<1>, std::forward<U>(value)) {}

  [[nodiscard]] bool ok() const { return rep_.index() == 1; }

  // The error, or OkStatus() if a value is held. By value: error paths are
  // cold, and it keeps StatusOr from having to store an OK status.
  [[nodiscard]] Status status() const& {
    if (ok()) {
      return OkStatus();
    }
    return std::get<0>(rep_);
  }
  [[nodiscard]] Status status() && {
    if (ok()) {
      return OkStatus();
    }
    return std::move(std::get<0>(rep_));
  }

  // The value; CHECK-fails if an error is held.
  [[nodiscard]] const T& value() const& {
    CHECK(ok(), "StatusOr::value() on error: {}", status().ToString());
    return std::get<1>(rep_);
  }
  [[nodiscard]] T& value() & {
    CHECK(ok(), "StatusOr::value() on error: {}", status().ToString());
    return std::get<1>(rep_);
  }
  [[nodiscard]] T&& value() && {
    CHECK(ok(), "StatusOr::value() on error: {}", status().ToString());
    return std::get<1>(std::move(rep_));
  }

  // Unchecked-in-Release access; the caller has already tested ok().
  [[nodiscard]] const T& operator*() const& {
    DCHECK(ok());
    return std::get<1>(rep_);
  }
  [[nodiscard]] T& operator*() & {
    DCHECK(ok());
    return std::get<1>(rep_);
  }
  [[nodiscard]] T&& operator*() && {
    DCHECK(ok());
    return std::get<1>(std::move(rep_));
  }
  [[nodiscard]] const T* operator->() const {
    DCHECK(ok());
    return &std::get<1>(rep_);
  }
  [[nodiscard]] T* operator->() {
    DCHECK(ok());
    return &std::get<1>(rep_);
  }

 private:
  // index 0 = error Status, index 1 = value. std::variant supplies correct
  // copy/move/destroy semantics; its valueless-by-exception state is
  // unreachable here because engine types do not throw (ADR-003).
  std::variant<Status, T> rep_;
};

namespace detail {

// Uniform status extraction so CHECK_OK accepts Status and StatusOr alike.
// By value: returning a reference into the argument invites dangling when
// called on a temporary, and an OK Status copies without allocating.
[[nodiscard]] inline Status GetStatus(const Status& status) { return status; }
template <typename T>
[[nodiscard]] Status GetStatus(const StatusOr<T>& status_or) {
  return status_or.status();
}

}  // namespace detail
}  // namespace engine::core

// Evaluates `expr_` (a Status expression) exactly once and returns it from
// the enclosing function if it is an error. The enclosing function must
// return Status.
#define RETURN_IF_ERROR(expr_)                           \
  do {                                                   \
    ::engine::core::Status engine_rie_status_ = (expr_); \
    if (!engine_rie_status_.ok()) {                      \
      return engine_rie_status_;                         \
    }                                                    \
  } while (0)

// Evaluates `rexpr_` (a StatusOr<T> expression) exactly once; on error
// returns its Status from the enclosing function (which must return Status),
// otherwise assigns the value to `lhs_`, which may declare a new variable:
//
//   ASSIGN_OR_RETURN(auto config, LoadConfig(path));
//
// Expands to multiple statements, so it cannot be the body of an unbraced
// `if`/`for`. `lhs_` cannot contain unparenthesized commas (e.g. a
// std::pair<A, B> declaration) — declare such variables first and assign.
// NOLINTBEGIN(bugprone-macro-parentheses): `lhs_` may be a declaration and
// `statusor_` a declarator name; neither can be parenthesized.
#define ASSIGN_OR_RETURN(lhs_, rexpr_) \
  ENGINE_ASSIGN_OR_RETURN_IMPL(        \
      ENGINE_STATUS_CONCAT(engine_statusor_, __LINE__), lhs_, rexpr_)

#define ENGINE_ASSIGN_OR_RETURN_IMPL(statusor_, lhs_, rexpr_) \
  auto statusor_ = (rexpr_);                                  \
  if (!statusor_.ok()) {                                      \
    return ::std::move(statusor_).status();                   \
  }                                                           \
  lhs_ = ::std::move(statusor_).value()
// NOLINTEND(bugprone-macro-parentheses)

#define ENGINE_STATUS_CONCAT(a_, b_) ENGINE_STATUS_CONCAT_IMPL(a_, b_)
#define ENGINE_STATUS_CONCAT_IMPL(a_, b_) a_##b_

// CHECK_OK(expr): fatal unless `expr` — a Status or StatusOr<T> — is OK.
// For programmer errors only, like CHECK (see core/check.h); DCHECK_OK is
// the Debug-only variant.
#define CHECK_OK(expr_) ENGINE_CHECK_OK_IMPL(expr_, "CHECK_OK(" #expr_ ")")
#ifdef NDEBUG
#define DCHECK_OK(expr_)                                    \
  do {                                                      \
    if (false) {                                            \
      ENGINE_CHECK_OK_IMPL(expr_, "DCHECK_OK(" #expr_ ")"); \
    }                                                       \
  } while (0)
#else
#define DCHECK_OK(expr_) ENGINE_CHECK_OK_IMPL(expr_, "DCHECK_OK(" #expr_ ")")
#endif

// `auto&&` binds to both lvalues and temporaries (extending the latter's
// lifetime across the branch); the status is re-extracted for the message
// only on the cold failure path.
#define ENGINE_CHECK_OK_IMPL(expr_, text_)                                     \
  do {                                                                         \
    if (auto&& engine_check_ok_val_ = (expr_);                                 \
        !::engine::core::detail::GetStatus(engine_check_ok_val_).ok())         \
        [[unlikely]] {                                                         \
      ::engine::core::detail::CheckFailed(                                     \
          text_, __FILE__, __LINE__, __func__,                                 \
          ::engine::core::detail::GetStatus(engine_check_ok_val_).ToString()); \
    }                                                                          \
  } while (0)
