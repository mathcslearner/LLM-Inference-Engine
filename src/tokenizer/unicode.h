#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// Unicode machinery for encode (M4-T09; design: docs/design/model-loading.md
// §6.3): the character categories the pre-tokenization pattern needs
// (\p{L}, \p{N}, \s) and NFC normalization (Qwen's normalizer), plus the
// UTF-8 codec the tokenizer module shares. The category and normalization
// tables are generated from pinned Unicode 16.0.0 data files by
// tools/gen_unicode and committed as src/tokenizer/unicode_data.inc — no
// ICU, no system-versioned data (§6.3's dependency decision). Module-
// internal: nothing outside src/tokenizer/ should need these.

namespace engine::tokenizer {

// General category L* (Lu, Ll, Lt, Lm, Lo) — the pattern's \p{L}.
[[nodiscard]] bool is_letter(char32_t codepoint);

// General category N* (Nd, Nl, No) — the pattern's \p{N}.
[[nodiscard]] bool is_number(char32_t codepoint);

// The White_Space property — what \s means in HF's regex engine (Rust
// regex/fancy-regex Unicode semantics), notably including U+00A0.
[[nodiscard]] bool is_whitespace(char32_t codepoint);

// Canonical composition (NFC), byte-identical to HF `tokenizers`' NFC
// normalizer. Total: bytes that do not decode as UTF-8 are copied through
// verbatim and act as normalization barriers (encode only feeds this valid
// segments, §6.4 — the total contract just avoids a redundant error path).
// The common case (ASCII, or any text already in NFC by quick-check) is
// returned without running the full algorithm.
[[nodiscard]] std::string nfc_normalize(std::string_view text);

// --- UTF-8 codec (shared by bpe.cpp, pretokenize.cpp, and NFC) ------------

// Appends the UTF-8 encoding of `codepoint` (any scalar value < 0x110000).
void append_utf8(std::string& out, char32_t codepoint);

// Decodes the UTF-8 sequence starting at `pos`, advancing `pos` past it.
// nullopt on a malformed sequence (bad lead byte, truncated or invalid
// continuation), leaving `pos` unchanged. Overlong encodings and
// surrogate/out-of-range codepoints are not rejected — byte-level BPE
// treats them like any other bytes and the callers' table lookups miss.
[[nodiscard]] std::optional<char32_t> decode_utf8(std::string_view text,
                                                  std::size_t& pos);

// --- strict UTF-8 classification (decode side, M4-T10) ---------------------
//
// decode_utf8 above is deliberately lenient; decode output must instead
// match HF exactly: `tokenizers` materializes the concatenated token bytes
// as a Rust String via from_utf8_lossy — strict well-formedness (RFC 3629:
// overlong forms, surrogates, and > U+10FFFF are all invalid) with the
// WHATWG "maximal subpart" policy, where each longest prefix of a valid
// sequence that cannot be completed becomes exactly one U+FFFD. The
// malformed-tail decode goldens pin this (design §6.5 amendment).
// classify_utf8_prefix is the incremental primitive (DetokenizerStream
// buffers on kIncomplete); append_utf8_lossy is the batch loop over it.

enum class Utf8Prefix : std::uint8_t {
  kComplete,    // bytes[0..length) is one well-formed sequence
  kIncomplete,  // all of `bytes` is a proper prefix of some well-formed
                // sequence — further bytes could still complete it
  kInvalid,     // bytes[0..length) is a maximal subpart: replace it with
                // one U+FFFD and rescan at `length`
};

struct Utf8Classification {
  Utf8Prefix kind;
  std::size_t length;  // bytes consumed (kIncomplete: bytes.size())
};

// Classifies the leading UTF-8 sequence of `bytes` (must be non-empty).
[[nodiscard]] Utf8Classification classify_utf8_prefix(std::string_view bytes);

// Appends `bytes` to `out` with every maximal invalid subpart replaced by
// U+FFFD (Rust String::from_utf8_lossy semantics). A trailing incomplete
// sequence is invalid here — batch conversion has no more bytes coming.
void append_utf8_lossy(std::string& out, std::string_view bytes);

}  // namespace engine::tokenizer
