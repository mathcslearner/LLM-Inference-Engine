#include "tokenizer/bpe.h"

#include "core/status.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>

// The GPT-2 byte-level alphabet bijection (M4-T08; design:
// docs/design/model-loading.md §6.2). The 256-entry map is otherwise only
// covered indirectly through encode/decode goldens — this pins the
// bijection itself: total, injective, and exactly inverted, for every byte
// value.

namespace {

using engine::tokenizer::alphabet_to_byte;
using engine::tokenizer::byte_to_alphabet;
using engine::tokenizer::map_bytes_to_alphabet;
using engine::tokenizer::unmap_alphabet_to_bytes;

TEST(BpeAlphabetTest, All256BytesRoundTripThroughDistinctCodepoints) {
  std::set<char32_t> images;
  for (int value = 0; value < 256; ++value) {
    const auto byte = static_cast<std::uint8_t>(value);
    const char32_t codepoint = byte_to_alphabet(byte);
    EXPECT_TRUE(images.insert(codepoint).second)
        << "byte " << value << " maps to an already-used codepoint";
    // Whole-optional comparison: no dereference, so no
    // bugprone-unchecked-optional-access blind spot around gtest asserts.
    EXPECT_EQ(alphabet_to_byte(codepoint), std::optional<std::uint8_t>{byte})
        << "byte " << value;
  }
  EXPECT_EQ(images.size(), 256U);
}

TEST(BpeAlphabetTest, SpotChecksMatchTheGpt2Convention) {
  // Printable latin maps to itself; the 68 remapped bytes go to U+0100+n
  // in ascending byte order — 0x00 is the first, space (0x20) is Ġ.
  EXPECT_EQ(byte_to_alphabet('!'), U'!');
  EXPECT_EQ(byte_to_alphabet('~'), U'~');
  EXPECT_EQ(byte_to_alphabet(0x00), char32_t{0x100});
  EXPECT_EQ(byte_to_alphabet(' '), U'Ġ');
  EXPECT_EQ(alphabet_to_byte(U'Ġ'), std::optional<std::uint8_t>{' '});
}

TEST(BpeAlphabetTest, CodepointsOutsideTheAlphabetHaveNoPreimage) {
  EXPECT_FALSE(alphabet_to_byte(U'\0').has_value());
  EXPECT_FALSE(alphabet_to_byte(char32_t{0x1F600}).has_value());  // 😀
}

TEST(BpeAlphabetTest, StringMappingRoundTripsEveryByteValue) {
  std::string all_bytes;
  for (int value = 0; value < 256; ++value) {
    all_bytes.push_back(static_cast<char>(value));
  }
  const std::string mapped = map_bytes_to_alphabet(all_bytes);
  const auto unmapped = unmap_alphabet_to_bytes(mapped);
  ASSERT_TRUE(unmapped.ok()) << unmapped.status().ToString();
  EXPECT_EQ(*unmapped, all_bytes);
}

}  // namespace
