#pragma once

#include "core/status.h"

#include <string_view>
#include <vector>

// Pre-tokenization for byte-level BPE (M4-T09; design:
// docs/design/model-loading.md §6.4 step 3). The Split regexes the target
// families ship are two fixed members of the GPT-2/cl100k pattern family;
// rather than pull in a regex engine that can express \p{L} plus lookahead,
// a hand-written matcher implements the family, parameterized by the one
// knob on which the two differ, and is selected by matching the pattern
// string from tokenizer.json verbatim. Fidelity is pinned by the golden
// vectors — that is the whole test strategy for the lookahead and category
// edge cases.

namespace engine::tokenizer {

// The pattern-family parameters. Both known patterns are, alternative by
// alternative: case-insensitive contraction suffixes, an optional
// non-letter prefix glued to a letter run, a digit run, punctuation with an
// optional leading space and trailing newlines, whitespace-then-newlines,
// and the trailing-whitespace lookahead \s+(?!\S).
struct SplitSpec {
  // \p{N}{1,3} (Llama 3 / cl100k) vs \p{N} (Qwen 2).
  int max_digit_run = 1;
};

// Matches `pattern` against the known pattern strings. `Unimplemented`
// naming the pattern otherwise (§6.2's whitelist discipline: a pattern this
// matcher was never validated against must fail loudly, not approximately).
[[nodiscard]] core::StatusOr<SplitSpec> select_split_pattern(
    std::string_view pattern);

// Splits one segment into pre-token spans (views into `text`, in order,
// concatenating back to exactly `text`). `text` is one added-token-free,
// normalized segment; it should be valid UTF-8 — encode routes invalid
// byte runs around the matcher (§6.4) — but undecodable bytes are handled
// totally anyway: each becomes its own span.
[[nodiscard]] std::vector<std::string_view> pretokenize(std::string_view text,
                                                        SplitSpec spec);

}  // namespace engine::tokenizer
