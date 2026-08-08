#include "tokenizer/bpe.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace engine::tokenizer {
namespace {

// The three byte ranges that map to themselves (GPT-2 bytes_to_unicode):
// the printable latin-1 characters minus space, DEL-adjacent controls, and
// the two gaps 0xA0 (NBSP) and 0xAD (soft hyphen).
constexpr bool IsPrintableLatin(unsigned byte) {
  return (byte >= U'!' && byte <= U'~') || (byte >= 0xA1 && byte <= 0xAC) ||
         (byte >= 0xAE && byte <= 0xFF);
}

// Highest alphabet codepoint: 68 non-printable bytes are remapped to
// U+0100 + n, so the inverse table covers [0, 0x100 + 68).
constexpr std::size_t kInverseSize = 0x144;

struct AlphabetTables {
  std::array<char32_t, 256> to_alphabet;
  std::array<std::int16_t, kInverseSize> to_byte;  // -1: not in the alphabet
};

constexpr AlphabetTables BuildTables() {
  AlphabetTables tables{};
  for (auto& entry : tables.to_byte) {
    entry = -1;
  }
  char32_t next_remapped = 0x100;
  for (unsigned byte = 0; byte < 256; ++byte) {
    const char32_t codepoint =
        IsPrintableLatin(byte) ? static_cast<char32_t>(byte) : next_remapped++;
    tables.to_alphabet[byte] = codepoint;
    tables.to_byte[codepoint] = static_cast<std::int16_t>(byte);
  }
  return tables;
}

constexpr AlphabetTables kTables = BuildTables();

// Appends the UTF-8 encoding of `codepoint` (< 0x800 — every alphabet
// codepoint fits in two bytes).
void AppendUtf8(std::string& out, char32_t codepoint) {
  if (codepoint < 0x80) {
    out.push_back(static_cast<char>(codepoint));
  } else {
    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
}

// Decodes the UTF-8 sequence starting at `pos`, advancing `pos` past it.
// nullopt on a malformed sequence (bad lead byte, truncated or invalid
// continuation). Overlong encodings are not rejected here — they decode to
// codepoints outside the alphabet and fail the caller's lookup instead.
std::optional<char32_t> DecodeUtf8(std::string_view text, std::size_t& pos) {
  const auto lead = static_cast<unsigned char>(text[pos]);
  std::size_t length = 0;
  char32_t codepoint = 0;
  if (lead < 0x80) {
    length = 1;
    codepoint = lead;
  } else if ((lead & 0xE0) == 0xC0) {
    length = 2;
    codepoint = lead & 0x1FU;
  } else if ((lead & 0xF0) == 0xE0) {
    length = 3;
    codepoint = lead & 0x0FU;
  } else if ((lead & 0xF8) == 0xF0) {
    length = 4;
    codepoint = lead & 0x07U;
  } else {
    return std::nullopt;
  }
  if (pos + length > text.size()) {
    return std::nullopt;
  }
  for (std::size_t i = 1; i < length; ++i) {
    const auto cont = static_cast<unsigned char>(text[pos + i]);
    if ((cont & 0xC0) != 0x80) {
      return std::nullopt;
    }
    codepoint = (codepoint << 6) | (cont & 0x3FU);
  }
  pos += length;
  return codepoint;
}

}  // namespace

char32_t byte_to_alphabet(std::uint8_t byte) {
  return kTables.to_alphabet[byte];
}

std::optional<std::uint8_t> alphabet_to_byte(char32_t codepoint) {
  if (codepoint >= kInverseSize || kTables.to_byte[codepoint] < 0) {
    return std::nullopt;
  }
  return static_cast<std::uint8_t>(kTables.to_byte[codepoint]);
}

std::string map_bytes_to_alphabet(std::string_view raw) {
  std::string out;
  out.reserve(raw.size() * 2);  // worst case: every byte remaps to 2 UTF-8
  for (const char byte : raw) {
    AppendUtf8(out, byte_to_alphabet(static_cast<std::uint8_t>(byte)));
  }
  return out;
}

core::StatusOr<std::string> unmap_alphabet_to_bytes(std::string_view token) {
  std::string out;
  out.reserve(token.size());
  std::size_t pos = 0;
  while (pos < token.size()) {
    const auto codepoint = DecodeUtf8(token, pos);
    if (!codepoint.has_value()) {
      return core::InvalidArgumentError(
          "alphabet string is not valid UTF-8 at byte offset {}", pos);
    }
    const auto byte = alphabet_to_byte(*codepoint);
    if (!byte.has_value()) {
      return core::InvalidArgumentError(
          "codepoint U+{:04X} is not in the byte-level BPE alphabet",
          static_cast<std::uint32_t>(*codepoint));
    }
    out.push_back(static_cast<char>(*byte));
  }
  return out;
}

}  // namespace engine::tokenizer
