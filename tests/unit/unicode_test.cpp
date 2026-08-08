#include "tokenizer/unicode.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// M4-T09 (design: docs/design/model-loading.md §6.3): the generated
// category tables (\p{L}, \p{N}, White_Space) and NFC normalization. Every
// NFC case was cross-checked against Python 3's unicodedata at the same
// pinned Unicode version (16.0.0); the real-text path is additionally
// pinned by the qwen2 nfc_decomposed golden vector (tokenizer_encode_test).
// All non-ASCII text is spelled with explicit \u escapes so the source is
// unambiguous about which side is decomposed.

namespace {

using engine::tokenizer::append_utf8;
using engine::tokenizer::decode_utf8;
using engine::tokenizer::is_letter;
using engine::tokenizer::is_number;
using engine::tokenizer::is_whitespace;
using engine::tokenizer::nfc_normalize;

// --- categories -------------------------------------------------------------

TEST(UnicodeCategoryTest, Letters) {
  EXPECT_TRUE(is_letter(U'A'));
  EXPECT_TRUE(is_letter(U'z'));
  EXPECT_TRUE(is_letter(0xE9));     // Ll (e-acute)
  EXPECT_TRUE(is_letter(0x3A9));    // Lu (Greek omega)
  EXPECT_TRUE(is_letter(0x2B0));    // Lm (modifier small h)
  EXPECT_TRUE(is_letter(0x4E00));   // Lo (CJK ideograph)
  EXPECT_TRUE(is_letter(0xAC00));   // Lo (Hangul syllable)
  EXPECT_TRUE(is_letter(0x1D400));  // Lu (mathematical bold A)
  EXPECT_FALSE(is_letter(U'0'));
  EXPECT_FALSE(is_letter(U' '));
  EXPECT_FALSE(is_letter(U'!'));
  EXPECT_FALSE(is_letter(0x301));     // Mn (combining acute)
  EXPECT_FALSE(is_letter(0x10FFFF));  // unassigned
}

TEST(UnicodeCategoryTest, Numbers) {
  EXPECT_TRUE(is_number(U'0'));
  EXPECT_TRUE(is_number(U'9'));
  EXPECT_TRUE(is_number(0x660));   // Nd (Arabic-Indic zero)
  EXPECT_TRUE(is_number(0x2160));  // Nl (Roman numeral one)
  EXPECT_TRUE(is_number(0xB2));    // No (superscript two)
  EXPECT_TRUE(is_number(0xBD));    // No (vulgar fraction one half)
  EXPECT_FALSE(is_number(U'a'));
  EXPECT_FALSE(is_number(U'-'));  // sign is not part of \p{N}
  EXPECT_FALSE(is_number(U'.'));
}

TEST(UnicodeCategoryTest, WhitespaceIsTheWhiteSpaceProperty) {
  EXPECT_TRUE(is_whitespace(U' '));
  EXPECT_TRUE(is_whitespace(U'\t'));
  EXPECT_TRUE(is_whitespace(U'\r'));
  EXPECT_TRUE(is_whitespace(U'\n'));
  EXPECT_TRUE(is_whitespace(0x0B));    // vertical tab
  EXPECT_TRUE(is_whitespace(0xA0));    // NBSP — \s in Rust regex semantics
  EXPECT_TRUE(is_whitespace(0x3000));  // ideographic space
  EXPECT_TRUE(is_whitespace(0x2028));  // line separator
  EXPECT_FALSE(is_whitespace(U'a'));
  EXPECT_FALSE(is_whitespace(0x200B));  // zero-width space: not White_Space
  EXPECT_FALSE(is_whitespace(0x301));
}

// --- UTF-8 codec ------------------------------------------------------------

TEST(Utf8CodecTest, RoundTripsAtEncodingBoundaries) {
  const std::vector<char32_t> boundaries = {0x0,   0x7F,   0x80,    0x7FF,
                                            0x800, 0xFFFF, 0x10000, 0x10FFFF};
  for (const char32_t codepoint : boundaries) {
    std::string encoded;
    append_utf8(encoded, codepoint);
    std::size_t pos = 0;
    const auto decoded = decode_utf8(encoded, pos);
    EXPECT_EQ(decoded, codepoint) << static_cast<std::uint32_t>(codepoint);
    EXPECT_EQ(pos, encoded.size());
  }
}

TEST(Utf8CodecTest, RejectsMalformedSequencesWithoutAdvancing) {
  // Lone continuation byte, bad lead byte, truncated 3-byte sequence,
  // continuation byte missing mid-sequence.
  for (const std::string_view bad :
       {"\x80", "\xFF", "\xE2\x82", "\xE2\x41\x41"}) {
    std::size_t pos = 0;
    EXPECT_EQ(decode_utf8(bad, pos), std::nullopt) << bad;
    EXPECT_EQ(pos, 0U);
  }
}

// --- NFC --------------------------------------------------------------------

TEST(NfcTest, AsciiAndComposedTextPassThrough) {
  EXPECT_EQ(nfc_normalize(""), "");
  EXPECT_EQ(nfc_normalize("plain ASCII text 123!"), "plain ASCII text 123!");
  // Already NFC (composed e-acute), and CJK: both hit the fast path.
  EXPECT_EQ(nfc_normalize("caf\u00E9"), "caf\u00E9");
  EXPECT_EQ(nfc_normalize("\u4F60\u597D\uFF0C\u4E16\u754C"),
            "\u4F60\u597D\uFF0C\u4E16\u754C");
}

TEST(NfcTest, ComposesCanonically) {
  // e + combining acute composes.
  EXPECT_EQ(nfc_normalize("cafe\u0301"), "caf\u00E9");
  // Reordering: dot above (ccc 230) before dot below (ccc 220) must
  // swap; q composes with neither.
  EXPECT_EQ(nfc_normalize("q\u0307\u0323"), "q\u0323\u0307");
  // Singleton: Angstrom sign maps to A-with-ring.
  EXPECT_EQ(nfc_normalize("\u212B"), "\u00C5");
  // d-with-dot-above + dot below: decompose, reorder, recompose the
  // pair that now touches the starter.
  EXPECT_EQ(nfc_normalize("\u1E0B\u0323"), "\u1E0D\u0307");
  // e + dot below + acute: the dot below composes, the acute remains.
  EXPECT_EQ(nfc_normalize("e\u0323\u0301"), "\u1EB9\u0301");
  // A second identical mark must not compose again (blocking).
  EXPECT_EQ(nfc_normalize("e\u0301\u0301"), "\u00E9\u0301");
  // a + ogonek (ccc 202) + acute (ccc 230): the ogonek composes first.
  EXPECT_EQ(nfc_normalize("a\u0328\u0301"), "\u0105\u0301");
}

TEST(NfcTest, HonorsCompositionExclusions) {
  // U+0958 (DEVANAGARI QA) is composition-excluded: it decomposes to
  // ka + nukta and must not recompose...
  EXPECT_EQ(nfc_normalize("\u0958"), "\u0915\u093C");
  // ...from either spelling.
  EXPECT_EQ(nfc_normalize("\u0915\u093C"), "\u0915\u093C");
  // Musical symbols are excluded too (and fully decompose to three).
  EXPECT_EQ(nfc_normalize("\U0001D160"), "\U0001D158\U0001D165\U0001D16E");
}

TEST(NfcTest, ComposesHangulAlgorithmically) {
  // L + V.
  EXPECT_EQ(nfc_normalize("\u1100\u1161"), "\uAC00");
  // L + V + T.
  EXPECT_EQ(nfc_normalize("\u1100\u1161\u11A8"), "\uAC01");
  // LV + T.
  EXPECT_EQ(nfc_normalize("\uAC00\u11A8"), "\uAC01");
  // Already-composed syllables are untouched.
  EXPECT_EQ(nfc_normalize("\uD55C\uAD6D\uC5B4"), "\uD55C\uAD6D\uC5B4");
}

TEST(NfcTest, IsTotalOverInvalidBytes) {
  // Undecodable bytes pass through verbatim and act as barriers; the valid
  // decomposed text around them still normalizes. (Adjacent literals keep
  // hex escapes from swallowing following characters.)
  const std::string input =
      "e\u0301"
      "\xFF"
      "e\u0301";
  const std::string expected =
      "\u00E9"
      "\xFF"
      "\u00E9";
  EXPECT_EQ(nfc_normalize(input), expected);
}

TEST(NfcTest, IsIdempotent) {
  const std::vector<std::string> inputs = {
      "cafe\u0301",         "q\u0307\u0323", "\u212B",
      "\u1100\u1161\u11A8", "\u0958",        "e\u0323\u0301",
  };
  for (const std::string& input : inputs) {
    const std::string once = nfc_normalize(input);
    EXPECT_EQ(nfc_normalize(once), once) << input;
  }
}

}  // namespace
