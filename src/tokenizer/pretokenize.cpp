#include "tokenizer/pretokenize.h"

#include "tokenizer/unicode.h"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace engine::tokenizer {
namespace {

// The two Split regexes the target families ship (fixture-pinned, §6.2).
// They differ only in the digit-run alternative: cl100k groups 1–3 digits,
// Qwen 2 isolates every digit.
constexpr std::string_view kCl100kPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr std::string_view kQwen2Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// Sentinel for a byte that did not decode as UTF-8: outside every category
// class, always a pre-token of its own (the caller normally routes invalid
// runs around the matcher; this keeps the function total regardless).
constexpr char32_t kInvalidByte = 0x110000;

bool IsCrLf(char32_t codepoint) {
  return codepoint == U'\r' || codepoint == U'\n';
}

// The [^\s\p{L}\p{N}] class of alternative 4 (with the sentinel carved
// out — it is not a character).
bool IsPunctClass(char32_t codepoint) {
  return codepoint != kInvalidByte && !is_whitespace(codepoint) &&
         !is_letter(codepoint) && !is_number(codepoint);
}

// ASCII-only case fold for the (?i:...) contraction group. The suffixes
// are ASCII letters; full Unicode simple folding would additionally match
// exotica like 'ſ' (LATIN SMALL LETTER LONG S) for 's — no golden vector
// exercises that, and the goldens are the contract (§6.4).
char32_t AsciiFold(char32_t codepoint) {
  return codepoint >= U'A' && codepoint <= U'Z' ? codepoint + 0x20 : codepoint;
}

// Alternative 1: (?i:'s|'t|'re|'ve|'m|'ll|'d). 0 = no match.
std::size_t MatchContraction(std::span<const char32_t> cps, std::size_t i) {
  const std::size_t n = cps.size();
  if (cps[i] != U'\'' || i + 1 >= n) {
    return 0;
  }
  const char32_t second = AsciiFold(cps[i + 1]);
  if (second == U's' || second == U't' || second == U'm' || second == U'd') {
    return 2;
  }
  if (i + 2 < n) {
    const char32_t third = AsciiFold(cps[i + 2]);
    if ((second == U'r' && third == U'e') ||
        (second == U'v' && third == U'e') ||
        (second == U'l' && third == U'l')) {
      return 3;
    }
  }
  return 0;
}

// Alternative 2: [^\r\n\p{L}\p{N}]?\p{L}+ — a letter run, optionally
// preceded by one codepoint outside {CR, LF, letters, numbers} (this is
// what glues a leading space or tab onto a word: " b", "\td"). 0 = no
// match.
std::size_t MatchLetterRun(std::span<const char32_t> cps, std::size_t i) {
  const std::size_t n = cps.size();
  std::size_t j = i;
  if (!is_letter(cps[j])) {
    const bool prefix_ok = cps[j] != kInvalidByte && !IsCrLf(cps[j]) &&
                           !is_number(cps[j]) && j + 1 < n &&
                           is_letter(cps[j + 1]);
    if (!prefix_ok) {
      return 0;
    }
    ++j;
  }
  while (j < n && is_letter(cps[j])) {
    ++j;
  }
  return j - i;
}

// Alternative 4: ' '? [^\s\p{L}\p{N}]+ [\r\n]* — punctuation with an
// optional leading space and any trailing newlines. (If the space is not
// followed by punctuation the optional prefix backtracks away, and a bare
// space is \s, so the alternative fails as a whole.) 0 = no match.
std::size_t MatchPunctuationRun(std::span<const char32_t> cps, std::size_t i) {
  const std::size_t n = cps.size();
  std::size_t j = i;
  if (cps[i] == U' ' && i + 1 < n && IsPunctClass(cps[i + 1])) {
    j = i + 1;
  }
  if (!IsPunctClass(cps[j])) {
    return 0;
  }
  while (j < n && IsPunctClass(cps[j])) {
    ++j;
  }
  while (j < n && IsCrLf(cps[j])) {
    ++j;
  }
  return j - i;
}

// Alternatives 5-7 over the maximal \s+ run from i: \s*[\r\n]+, then
// \s+(?!\S), then \s+. 0 = cps[i] is not whitespace.
std::size_t MatchWhitespaceRun(std::span<const char32_t> cps, std::size_t i) {
  const std::size_t n = cps.size();
  if (!is_whitespace(cps[i])) {
    return 0;
  }
  std::size_t j = i;
  while (j < n && is_whitespace(cps[j])) {
    ++j;
  }
  // 5. \s*[\r\n]+ — greedy \s* backtracks to the last CR/LF in the run,
  // so the match extends exactly through that character.
  std::size_t last_crlf = n;
  for (std::size_t k = i; k < j; ++k) {
    if (IsCrLf(cps[k])) {
      last_crlf = k;
    }
  }
  if (last_crlf != n) {
    return last_crlf - i + 1;
  }
  // 6. \s+(?!\S) — at end of input the whole run matches; before a
  // non-space the lookahead forces one whitespace to be left for the next
  // token's optional prefix (and a run of one fails entirely).
  if (j == n) {
    return j - i;
  }
  if (j - i > 1) {
    return j - i - 1;
  }
  // 7. \s+
  return 1;
}

// The regex's alternatives, tried in pattern order at codepoint `i`;
// returns the match length in codepoints (>= 1; every codepoint is matched
// by some alternative — at minimum 4's punctuation class or 7's \s+).
std::size_t MatchAt(std::span<const char32_t> cps, std::size_t i,
                    std::size_t max_digit_run) {
  if (cps[i] == kInvalidByte) {
    return 1;
  }
  if (const std::size_t len = MatchContraction(cps, i); len > 0) {
    return len;
  }
  if (const std::size_t len = MatchLetterRun(cps, i); len > 0) {
    return len;
  }
  // Alternative 3: \p{N}{1,k}.
  if (is_number(cps[i])) {
    std::size_t j = i + 1;
    while (j < cps.size() && j - i < max_digit_run && is_number(cps[j])) {
      ++j;
    }
    return j - i;
  }
  if (const std::size_t len = MatchPunctuationRun(cps, i); len > 0) {
    return len;
  }
  if (const std::size_t len = MatchWhitespaceRun(cps, i); len > 0) {
    return len;
  }
  return 1;  // unreachable: alternative 4 accepts any non-\s\p{L}\p{N} cp
}

}  // namespace

core::StatusOr<SplitSpec> select_split_pattern(std::string_view pattern) {
  if (pattern == kCl100kPattern) {
    return SplitSpec{.max_digit_run = 3};
  }
  if (pattern == kQwen2Pattern) {
    return SplitSpec{.max_digit_run = 1};
  }
  return core::UnimplementedError(
      "pre_tokenizer Split pattern \"{}\" is not one of the known "
      "GPT-2/cl100k-family patterns this matcher was validated against "
      "(design §6.4)",
      pattern);
}

std::vector<std::string_view> pretokenize(std::string_view text,
                                          SplitSpec spec) {
  // Decode once, keeping byte offsets so spans can be returned as views.
  std::vector<char32_t> cps;
  std::vector<std::size_t> offsets;
  cps.reserve(text.size());
  offsets.reserve(text.size() + 1);
  std::size_t pos = 0;
  while (pos < text.size()) {
    offsets.push_back(pos);
    const auto codepoint = decode_utf8(text, pos);
    if (codepoint.has_value()) {
      cps.push_back(*codepoint);
    } else {
      cps.push_back(kInvalidByte);
      ++pos;
    }
  }
  offsets.push_back(text.size());

  std::vector<std::string_view> spans;
  std::size_t i = 0;
  while (i < cps.size()) {
    const std::size_t len =
        MatchAt(cps, i, static_cast<std::size_t>(spec.max_digit_run));
    spans.push_back(text.substr(offsets[i], offsets[i + len] - offsets[i]));
    i += len;
  }
  return spans;
}

}  // namespace engine::tokenizer
