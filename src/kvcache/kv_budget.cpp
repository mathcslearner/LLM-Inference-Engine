#include "kvcache/kv_budget.h"

#include "core/check.h"
#include "core/status.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace engine::kvcache {

namespace {

// Trims ASCII whitespace from both ends.
std::string_view Trim(std::string_view s) {
  std::size_t begin = 0;
  std::size_t end = s.size();
  while (begin < end &&
         (std::isspace(static_cast<unsigned char>(s[begin])) != 0)) {
    ++begin;
  }
  while (end > begin &&
         (std::isspace(static_cast<unsigned char>(s[end - 1])) != 0)) {
    --end;
  }
  return s.substr(begin, end - begin);
}

// Case-insensitive suffix test.
bool EndsWithIgnoreCase(std::string_view s, std::string_view suffix) {
  if (s.size() < suffix.size()) {
    return false;
  }
  const std::string_view tail = s.substr(s.size() - suffix.size());
  for (std::size_t i = 0; i < suffix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(tail[i])) !=
        std::tolower(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  return true;
}

// Recognized unit suffix → (multiplier, suffix length). Longest match first so
// "GiB" is not shadowed by "B". Returns multiplier 0 when no unit is present.
struct Unit {
  std::int64_t multiplier = 0;
  std::size_t length = 0;
};

Unit MatchUnit(std::string_view s) {
  constexpr std::int64_t kKi = 1024;
  constexpr std::int64_t kMi = 1024LL * 1024;
  constexpr std::int64_t kGi = 1024LL * 1024 * 1024;
  constexpr std::int64_t kK = 1000;
  constexpr std::int64_t kM = 1000LL * 1000;
  constexpr std::int64_t kG = 1000LL * 1000 * 1000;
  if (EndsWithIgnoreCase(s, "GiB")) {
    return Unit{.multiplier = kGi, .length = 3};
  }
  if (EndsWithIgnoreCase(s, "MiB")) {
    return Unit{.multiplier = kMi, .length = 3};
  }
  if (EndsWithIgnoreCase(s, "KiB")) {
    return Unit{.multiplier = kKi, .length = 3};
  }
  if (EndsWithIgnoreCase(s, "GB")) {
    return Unit{.multiplier = kG, .length = 2};
  }
  if (EndsWithIgnoreCase(s, "MB")) {
    return Unit{.multiplier = kM, .length = 2};
  }
  if (EndsWithIgnoreCase(s, "KB")) {
    return Unit{.multiplier = kK, .length = 2};
  }
  if (EndsWithIgnoreCase(s, "B")) {
    return Unit{.multiplier = 1, .length = 1};
  }
  return Unit{.multiplier = 0, .length = 0};
}

// Parses a non-negative decimal, requiring the whole string to be consumed.
[[nodiscard]] core::StatusOr<double> ParseNonNegativeDouble(
    std::string_view s) {
  if (s.empty()) {
    return core::InvalidArgumentError("--kv-cache-memory: empty numeric value");
  }
  const std::string owned(s);
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(owned, &consumed);
  } catch (...) {
    return core::InvalidArgumentError("--kv-cache-memory: '{}' is not a number",
                                      s);
  }
  if (consumed != owned.size()) {
    return core::InvalidArgumentError(
        "--kv-cache-memory: '{}' has trailing characters", s);
  }
  if (!(value >= 0.0) || std::isinf(value) || std::isnan(value)) {
    return core::InvalidArgumentError(
        "--kv-cache-memory: '{}' is not a finite non-negative number", s);
  }
  return value;
}

}  // namespace

core::StatusOr<KvCacheMemorySpec> ParseKvCacheMemory(std::string_view text) {
  const std::string_view s = Trim(text);
  if (s.empty()) {
    return core::InvalidArgumentError("--kv-cache-memory: empty value");
  }

  const Unit unit = MatchUnit(s);
  if (unit.multiplier != 0) {
    const std::string_view num = Trim(s.substr(0, s.size() - unit.length));
    ASSIGN_OR_RETURN(const double value, ParseNonNegativeDouble(num));
    const double bytes = value * static_cast<double>(unit.multiplier);
    if (!(bytes >= 1.0)) {
      return core::InvalidArgumentError(
          "--kv-cache-memory: '{}' is a non-positive byte budget", text);
    }
    return KvCacheMemorySpec{
        .kind = KvCacheMemorySpec::Kind::kAbsoluteBytes,
        .bytes = static_cast<std::int64_t>(std::llround(bytes)),
        .fraction = 0.0};
  }

  // No unit: a value containing '.' is a fraction of host RAM; a bare integer
  // is an absolute byte count.
  if (s.find('.') != std::string_view::npos) {
    ASSIGN_OR_RETURN(const double fraction, ParseNonNegativeDouble(s));
    if (!(fraction > 0.0) || fraction > 1.0) {
      return core::InvalidArgumentError(
          "--kv-cache-memory: fraction '{}' must be in (0, 1]", text);
    }
    return KvCacheMemorySpec{.kind = KvCacheMemorySpec::Kind::kFraction,
                             .bytes = 0,
                             .fraction = fraction};
  }

  ASSIGN_OR_RETURN(const double value, ParseNonNegativeDouble(s));
  if (!(value >= 1.0)) {
    return core::InvalidArgumentError(
        "--kv-cache-memory: '{}' is a non-positive byte budget", text);
  }
  return KvCacheMemorySpec{
      .kind = KvCacheMemorySpec::Kind::kAbsoluteBytes,
      .bytes = static_cast<std::int64_t>(std::llround(value)),
      .fraction = 0.0};
}

core::StatusOr<std::int64_t> ResolveKvBudgetBytes(
    const KvCacheMemorySpec& spec, const KvBudgetInputs& inputs) {
  if (spec.kind == KvCacheMemorySpec::Kind::kAbsoluteBytes) {
    return spec.bytes;
  }
  if (inputs.host_ram_bytes <= 0) {
    return core::FailedPreconditionError(
        "--kv-cache-memory: host RAM is unknown on this platform; pass an "
        "absolute budget (e.g. 2GiB) instead of a fraction");
  }
  const double ceiling =
      spec.fraction * static_cast<double>(inputs.host_ram_bytes);
  const std::int64_t budget = static_cast<std::int64_t>(std::llround(ceiling)) -
                              inputs.weights_bytes - inputs.workspace_bytes;
  if (budget <= 0) {
    return core::ResourceExhaustedError(
        "--kv-cache-memory: fraction {:.3f} of {} B host RAM leaves nothing "
        "for KV after {} B weights + {} B workspace",
        spec.fraction, inputs.host_ram_bytes, inputs.weights_bytes,
        inputs.workspace_bytes);
  }
  return budget;
}

std::int64_t BlocksForTokens(std::int64_t tokens, int block_size) {
  CHECK(block_size > 0, "BlocksForTokens: block_size must be > 0, got {}",
        block_size);
  if (tokens <= 0) {
    return 1;
  }
  return (tokens + block_size - 1) / block_size;
}

}  // namespace engine::kvcache
