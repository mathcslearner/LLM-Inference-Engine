#include "core/check.h"

#include <fmt/format.h>

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace engine::core::detail {

void CheckFailed(const char* condition, const char* file, int line,
                 const char* function, std::string_view message) {
  // stderr, not the logging subsystem: this must work during static
  // initialization, after logging shutdown, and inside logging itself.
  fmt::print(stderr, "{} failed at {}:{} in {}{}{}\n", condition, file, line,
             function, message.empty() ? "" : ": ", message);
  std::fflush(stderr);
  std::abort();
}

}  // namespace engine::core::detail
