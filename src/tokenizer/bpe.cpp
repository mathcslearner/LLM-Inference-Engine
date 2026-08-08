#include "tokenizer/bpe.h"

#include "tokenizer/unicode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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
    append_utf8(out, byte_to_alphabet(static_cast<std::uint8_t>(byte)));
  }
  return out;
}

core::StatusOr<std::string> unmap_alphabet_to_bytes(std::string_view token) {
  std::string out;
  out.reserve(token.size());
  std::size_t pos = 0;
  while (pos < token.size()) {
    const auto codepoint = decode_utf8(token, pos);
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

std::vector<std::string_view> bpe_split(
    std::string_view word,
    const std::unordered_map<std::string, std::int32_t>& merge_ranks) {
  // Initial symbols: single alphabet codepoints (UTF-8 boundaries — every
  // alphabet string is valid UTF-8, one or two bytes per codepoint).
  std::vector<std::string_view> symbols;
  std::size_t pos = 0;
  while (pos < word.size()) {
    const std::size_t start = pos;
    ++pos;
    while (pos < word.size() &&
           (static_cast<unsigned char>(word[pos]) & 0xC0) == 0x80) {
      ++pos;
    }
    symbols.push_back(word.substr(start, pos - start));
  }
  // Repeatedly merge the leftmost occurrence of the lowest-ranked adjacent
  // pair. The rescan is O(symbols²) rank lookups per pre-token — pre-tokens
  // are short (a word or a whitespace run), and simplicity guarantees no
  // tie-breaking beyond rank order can creep in.
  std::string key;
  while (symbols.size() > 1) {
    auto best_rank = std::numeric_limits<std::int32_t>::max();
    std::size_t best = symbols.size();
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      key.assign(symbols[i]);
      key += ' ';
      key.append(symbols[i + 1]);
      const auto it = merge_ranks.find(key);
      if (it != merge_ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best == symbols.size()) {
      break;
    }
    // Adjacent symbols are contiguous substrings of `word`, so the merged
    // symbol is just the widened view.
    symbols[best] = std::string_view(
        symbols[best].data(), symbols[best].size() + symbols[best + 1].size());
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
  }
  return symbols;
}

}  // namespace engine::tokenizer
