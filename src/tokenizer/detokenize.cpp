#include "tokenizer/detokenize.h"

#include "core/status.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace engine::tokenizer {

DetokenizerStream::DetokenizerStream(const Tokenizer& tokenizer,
                                     bool skip_special_tokens)
    : tokenizer_(&tokenizer), skip_special_tokens_(skip_special_tokens) {}

core::StatusOr<std::string> DetokenizerStream::push(std::int32_t id) {
  const auto token = tokenizer_->token_bytes(id);
  if (!token.has_value()) {
    return core::InvalidArgumentError("id {} is out of range [0, {})", id,
                                      tokenizer_->vocab_size());
  }
  if (skip_special_tokens_ && tokenizer_->is_special(id)) {
    return std::string();
  }
  pending_.append(*token);
  // Drain everything provably complete or invalid; keep only a trailing
  // prefix that future bytes could still complete.
  std::string out;
  std::size_t pos = 0;
  while (pos < pending_.size()) {
    const auto part =
        classify_utf8_prefix(std::string_view(pending_).substr(pos));
    if (part.kind == Utf8Prefix::kIncomplete) {
      break;
    }
    if (part.kind == Utf8Prefix::kComplete) {
      out.append(pending_, pos, part.length);
    } else {
      append_utf8(out, 0xFFFD);
    }
    pos += part.length;
  }
  pending_.erase(0, pos);
  return out;
}

std::string DetokenizerStream::finish() {
  if (pending_.empty()) {
    return {};
  }
  // The carry is a single incomplete sequence — one maximal subpart.
  pending_.clear();
  std::string out;
  append_utf8(out, 0xFFFD);
  return out;
}

}  // namespace engine::tokenizer
