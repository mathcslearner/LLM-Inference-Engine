#include "tokenizer/unicode.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace engine::tokenizer {
namespace {

// Row types for the generated tables (tools/gen_unicode, Unicode 16.0.0 —
// design §6.3). All range arrays are sorted by `lo` and non-overlapping;
// kCompEntries is sorted by `key`.
struct CodepointRange {
  char32_t lo;
  char32_t hi;
};

struct CccRange {
  char32_t lo;
  char32_t hi;
  std::uint8_t ccc;
};

struct DecompEntry {
  char32_t cp;
  std::uint16_t offset;  // into kDecompPool
  std::uint8_t len;
};

struct CompEntry {
  std::uint64_t key;  // (uint64_t(starter) << 32) | combining
  char32_t composed;
};

#include "tokenizer/unicode_data.inc"

bool InRanges(std::span<const CodepointRange> ranges, char32_t codepoint) {
  const auto it = std::ranges::upper_bound(
      ranges, codepoint, {}, [](const CodepointRange& r) { return r.lo; });
  return it != ranges.begin() && codepoint <= std::prev(it)->hi;
}

// Canonical combining class; 0 for everything outside the table.
std::uint8_t CombiningClass(char32_t codepoint) {
  const auto* it = std::ranges::upper_bound(
      kCccRanges, codepoint, {}, [](const CccRange& r) { return r.lo; });
  if (it == std::begin(kCccRanges) || codepoint > std::prev(it)->hi) {
    return 0;
  }
  return std::prev(it)->ccc;
}

// The full (recursively expanded) canonical decomposition, or an empty span
// for codepoints that decompose to themselves. Hangul is handled
// algorithmically by the caller, never through this table.
std::span<const char32_t> CanonicalDecomposition(char32_t codepoint) {
  const auto* it = std::ranges::lower_bound(
      kDecompEntries, codepoint, {}, [](const DecompEntry& e) { return e.cp; });
  if (it == std::end(kDecompEntries) || it->cp != codepoint) {
    return {};
  }
  return {&kDecompPool[it->offset], static_cast<std::size_t>(it->len)};
}

// --- Hangul (Unicode §3.12, algorithmic — excluded from the tables) --------

constexpr char32_t kHangulSBase = 0xAC00;
constexpr char32_t kHangulLBase = 0x1100;
constexpr char32_t kHangulVBase = 0x1161;
constexpr char32_t kHangulTBase = 0x11A7;
constexpr std::uint32_t kHangulLCount = 19;
constexpr std::uint32_t kHangulVCount = 21;
constexpr std::uint32_t kHangulTCount = 28;
constexpr std::uint32_t kHangulNCount = kHangulVCount * kHangulTCount;
constexpr std::uint32_t kHangulSCount = kHangulLCount * kHangulNCount;

void DecomposeHangul(char32_t syllable, std::vector<char32_t>& out) {
  const std::uint32_t index = syllable - kHangulSBase;
  out.push_back(kHangulLBase + (index / kHangulNCount));
  out.push_back(kHangulVBase + ((index % kHangulNCount) / kHangulTCount));
  if (index % kHangulTCount != 0) {
    out.push_back(kHangulTBase + (index % kHangulTCount));
  }
}

std::optional<char32_t> ComposePair(char32_t first, char32_t second) {
  // L + V → LV syllable.
  if (first >= kHangulLBase && first < kHangulLBase + kHangulLCount &&
      second >= kHangulVBase && second < kHangulVBase + kHangulVCount) {
    const std::uint32_t lv_index =
        ((first - kHangulLBase) * kHangulVCount) + (second - kHangulVBase);
    return kHangulSBase + (lv_index * kHangulTCount);
  }
  // LV syllable + T → LVT syllable.
  if (first >= kHangulSBase && first < kHangulSBase + kHangulSCount &&
      (first - kHangulSBase) % kHangulTCount == 0 && second > kHangulTBase &&
      second < kHangulTBase + kHangulTCount) {
    return first + (second - kHangulTBase);
  }
  const std::uint64_t key = (static_cast<std::uint64_t>(first) << 32U) | second;
  const auto* it = std::ranges::lower_bound(
      kCompEntries, key, {}, [](const CompEntry& e) { return e.key; });
  if (it == std::end(kCompEntries) || it->key != key) {
    return std::nullopt;
  }
  return it->composed;
}

// --- NFC over one decoded segment ------------------------------------------

// True when `codepoint` cannot require any normalization work: NFC quick
// check Yes *and* combining class 0 (a QC=Yes mark can still need
// reordering, so QC alone is not enough for the fast path).
bool IsNfcInert(char32_t codepoint) {
  return CombiningClass(codepoint) == 0 &&
         !InRanges(kNfcNoOrMaybeRanges, codepoint);
}

void NormalizeSegment(std::vector<char32_t>& buffer, std::string& out) {
  if (buffer.empty()) {
    return;
  }
  // 1. Full canonical decomposition.
  std::vector<char32_t> decomposed;
  decomposed.reserve(buffer.size());
  for (const char32_t codepoint : buffer) {
    if (codepoint >= kHangulSBase && codepoint < kHangulSBase + kHangulSCount) {
      DecomposeHangul(codepoint, decomposed);
      continue;
    }
    const auto expansion = CanonicalDecomposition(codepoint);
    if (expansion.empty()) {
      decomposed.push_back(codepoint);
    } else {
      decomposed.insert(decomposed.end(), expansion.begin(), expansion.end());
    }
  }
  // 2. Canonical ordering: stable insertion of non-starters by combining
  // class (starters are barriers — ccc 0 never satisfies the swap test).
  for (std::size_t i = 1; i < decomposed.size(); ++i) {
    const std::uint8_t ccc = CombiningClass(decomposed[i]);
    if (ccc == 0) {
      continue;
    }
    std::size_t j = i;
    while (j > 0 && CombiningClass(decomposed[j - 1]) > ccc) {
      std::swap(decomposed[j - 1], decomposed[j]);
      --j;
    }
  }
  // 3. Canonical composition (the standard streaming algorithm: compose
  // each character with the last starter unless a character of equal or
  // higher combining class blocks it).
  char32_t starter = decomposed[0];
  std::size_t starter_pos = 0;
  std::size_t write_pos = 1;
  // 256 = "no starter yet / blocked": the first character may itself be a
  // non-starter, which nothing may compose with.
  std::uint32_t last_ccc = CombiningClass(starter) == 0 ? 0 : 256;
  for (std::size_t i = 1; i < decomposed.size(); ++i) {
    const char32_t current = decomposed[i];
    const std::uint8_t ccc = CombiningClass(current);
    const auto composed = ComposePair(starter, current);
    if (composed.has_value() && (last_ccc < ccc || last_ccc == 0)) {
      decomposed[starter_pos] = *composed;
      starter = *composed;
      continue;
    }
    if (ccc == 0) {
      starter_pos = write_pos;
      starter = current;
    }
    last_ccc = ccc;
    decomposed[write_pos++] = current;
  }
  decomposed.resize(write_pos);
  for (const char32_t codepoint : decomposed) {
    append_utf8(out, codepoint);
  }
  buffer.clear();
}

}  // namespace

bool is_letter(char32_t codepoint) {
  return InRanges(kLetterRanges, codepoint);
}

bool is_number(char32_t codepoint) {
  return InRanges(kNumberRanges, codepoint);
}

bool is_whitespace(char32_t codepoint) {
  return InRanges(kWhitespaceRanges, codepoint);
}

std::string nfc_normalize(std::string_view text) {
  // Fast path: every codepoint NFC-inert means the text is already NFC.
  bool inert = true;
  std::size_t pos = 0;
  while (pos < text.size()) {
    if (static_cast<unsigned char>(text[pos]) < 0x80) {
      ++pos;  // ASCII is always inert
      continue;
    }
    const auto codepoint = decode_utf8(text, pos);
    if (!codepoint.has_value() || !IsNfcInert(*codepoint)) {
      inert = false;
      break;
    }
  }
  if (inert) {
    return std::string(text);
  }

  // Full algorithm, segmented at undecodable bytes: such a byte is copied
  // through verbatim and no canonical process crosses it.
  std::string out;
  out.reserve(text.size());
  std::vector<char32_t> buffer;
  pos = 0;
  while (pos < text.size()) {
    const auto codepoint = decode_utf8(text, pos);
    if (codepoint.has_value()) {
      buffer.push_back(*codepoint);
      continue;
    }
    NormalizeSegment(buffer, out);
    out.push_back(text[pos]);
    ++pos;
  }
  NormalizeSegment(buffer, out);
  return out;
}

void append_utf8(std::string& out, char32_t codepoint) {
  if (codepoint < 0x80) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6U)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3FU)));
  } else if (codepoint < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (codepoint >> 12U)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3FU)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (codepoint >> 18U)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3FU)));
  }
}

std::optional<char32_t> decode_utf8(std::string_view text, std::size_t& pos) {
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
    codepoint = (codepoint << 6U) | (cont & 0x3FU);
  }
  pos += length;
  return codepoint;
}

Utf8Classification classify_utf8_prefix(std::string_view bytes) {
  const auto lead = static_cast<unsigned char>(bytes[0]);
  if (lead < 0x80) {
    return {.kind = Utf8Prefix::kComplete, .length = 1};
  }
  // Continuation range of the *second* byte per lead (WHATWG utf-8 decode
  // table): the restrictions reject overlong forms (E0/F0), surrogates
  // (ED), and > U+10FFFF (F4); later continuations are always 80–BF.
  std::size_t length = 0;
  unsigned char second_lo = 0x80;
  unsigned char second_hi = 0xBF;
  if (lead >= 0xC2 && lead <= 0xDF) {
    length = 2;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    length = 3;
    if (lead == 0xE0) {
      second_lo = 0xA0;
    } else if (lead == 0xED) {
      second_hi = 0x9F;
    }
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    length = 4;
    if (lead == 0xF0) {
      second_lo = 0x90;
    } else if (lead == 0xF4) {
      second_hi = 0x8F;
    }
  } else {
    // Continuation byte with no lead (80–BF), overlong lead (C0/C1), or
    // out-of-range lead (F5–FF): a one-byte maximal subpart.
    return {.kind = Utf8Prefix::kInvalid, .length = 1};
  }
  for (std::size_t i = 1; i < length; ++i) {
    if (i >= bytes.size()) {
      return {.kind = Utf8Prefix::kIncomplete, .length = bytes.size()};
    }
    const auto cont = static_cast<unsigned char>(bytes[i]);
    const unsigned char lo = i == 1 ? second_lo : 0x80;
    const unsigned char hi = i == 1 ? second_hi : 0xBF;
    if (cont < lo || cont > hi) {
      // The maximal subpart ends before the offending byte, which is
      // reclassified on its own at the next scan position.
      return {.kind = Utf8Prefix::kInvalid, .length = i};
    }
  }
  return {.kind = Utf8Prefix::kComplete, .length = length};
}

void append_utf8_lossy(std::string& out, std::string_view bytes) {
  std::size_t pos = 0;
  while (pos < bytes.size()) {
    const auto part = classify_utf8_prefix(bytes.substr(pos));
    if (part.kind == Utf8Prefix::kComplete) {
      out.append(bytes.substr(pos, part.length));
    } else {
      append_utf8(out, 0xFFFD);
    }
    pos += part.length;  // kIncomplete consumes the whole remainder
  }
}

}  // namespace engine::tokenizer
