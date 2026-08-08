#pragma once

#include "core/status.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Byte-level BPE alphabet and merge loop (M4-T08/T09; design:
// docs/design/model-loading.md §6.2, §6.4 steps 4–5). The GPT-2
// `bytes_to_unicode` bijection maps every byte value to a printable
// codepoint, so vocab and merge entries in `tokenizer.json` are strings
// over a 256-symbol alphabet with no whitespace or control characters: the
// 188 bytes in the printable-latin ranges `!`..`~`, `¡`..`¬`, `®`..`ÿ` map
// to themselves, and the remaining 68 map to U+0100 + n in ascending byte
// order. Encode (M4-T09) maps raw UTF-8 bytes *into* the alphabet before
// the merge loop; decode (M4-T10) and vocab loading (M4-T08) map alphabet
// strings *back* to raw bytes.

namespace engine::tokenizer {

// The codepoint the byte-level alphabet maps `byte` to. Total: every byte
// value has exactly one image.
[[nodiscard]] char32_t byte_to_alphabet(std::uint8_t byte);

// Inverse of `byte_to_alphabet`; nullopt for codepoints outside the
// 256-codepoint alphabet.
[[nodiscard]] std::optional<std::uint8_t> alphabet_to_byte(char32_t codepoint);

// Maps raw bytes into the merge alphabet: the UTF-8 encoding of each
// byte's alphabet codepoint. Total over arbitrary bytes — the input need
// not be valid UTF-8 (design §6.4).
[[nodiscard]] std::string map_bytes_to_alphabet(std::string_view raw);

// Inverse: an alphabet-space string (e.g. a vocab token, "Ġworld") → the
// raw bytes it stands for (" world"). `InvalidArgument` if the input is
// not valid UTF-8 or contains a codepoint outside the alphabet — for a
// vocab entry that means the file is not byte-level BPE output.
[[nodiscard]] core::StatusOr<std::string> unmap_alphabet_to_bytes(
    std::string_view token);

// The BPE merge loop (§6.4 step 5) over one pre-token already mapped into
// the merge alphabet: starting from single alphabet codepoints, repeatedly
// merges the adjacent pair with the lowest rank until no pair has one.
// Returns the final symbols as substrings of `word`, in order (their
// concatenation is `word`; the caller maps symbols to ids). `merge_ranks`
// is keyed "left right". Ranks are total, so the only possible tie is
// between occurrences of the *same* pair, which merge left-to-right —
// matching HF exactly; no other tie-breaking exists (§6.4).
[[nodiscard]] std::vector<std::string_view> bpe_split(
    std::string_view word,
    const std::unordered_map<std::string, std::int32_t>& merge_ranks);

}  // namespace engine::tokenizer
