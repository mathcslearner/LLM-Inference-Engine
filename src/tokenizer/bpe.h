#pragma once

#include "core/status.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// Byte-level BPE alphabet (M4-T08; design: docs/design/model-loading.md
// §6.2, §6.4 step 4). The GPT-2 `bytes_to_unicode` bijection maps every
// byte value to a printable codepoint, so vocab and merge entries in
// `tokenizer.json` are strings over a 256-symbol alphabet with no
// whitespace or control characters: the 188 bytes in the printable-latin
// ranges `!`..`~`, `¡`..`¬`, `®`..`ÿ` map to themselves, and the remaining
// 68 map to U+0100 + n in ascending byte order. Encode (M4-T09) maps raw
// UTF-8 bytes *into* the alphabet before the merge loop; decode (M4-T10)
// and vocab loading (M4-T08) map alphabet strings *back* to raw bytes.
// The merge loop itself joins this file in M4-T09.

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

}  // namespace engine::tokenizer
