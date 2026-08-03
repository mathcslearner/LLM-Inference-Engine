#include "core/status.h"

#include <fmt/format.h>

#include <ostream>
#include <string>
#include <string_view>

namespace engine::core {

std::string_view StatusCodeToString(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kCancelled:
      return "CANCELLED";
    case StatusCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case StatusCode::kNotFound:
      return "NOT_FOUND";
    case StatusCode::kAlreadyExists:
      return "ALREADY_EXISTS";
    case StatusCode::kFailedPrecondition:
      return "FAILED_PRECONDITION";
    case StatusCode::kOutOfRange:
      return "OUT_OF_RANGE";
    case StatusCode::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case StatusCode::kOutOfMemory:
      return "OUT_OF_MEMORY";
    case StatusCode::kUnimplemented:
      return "UNIMPLEMENTED";
    case StatusCode::kUnavailable:
      return "UNAVAILABLE";
    case StatusCode::kInternal:
      return "INTERNAL";
  }
  return "UNKNOWN";  // Unreachable; the switch is exhaustive.
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  return fmt::format("{}: {}", StatusCodeToString(code_), message_);
}

std::ostream& operator<<(std::ostream& os, const Status& status) {
  return os << status.ToString();
}

}  // namespace engine::core
