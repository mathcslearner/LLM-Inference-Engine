#include "tokenizer/pretokenize.h"

#include "core/status.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

// M4-T09 (design: docs/design/model-loading.md §6.4 step 3): pattern
// selection against the two known Split regexes, and the hand-written
// matcher's alternative-order semantics. The matcher's end-to-end fidelity
// is pinned by the golden encode vectors (tokenizer_encode_test); these
// unit cases document *why* each split falls out of the pattern.

namespace {

using engine::core::IsUnimplemented;
using engine::tokenizer::pretokenize;
using engine::tokenizer::select_split_pattern;
using engine::tokenizer::SplitSpec;

constexpr std::string_view kLlama3Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr std::string_view kQwen2Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

std::vector<std::string> Split(std::string_view text, int max_digit_run) {
  std::vector<std::string> out;
  for (const std::string_view span :
       pretokenize(text, SplitSpec{.max_digit_run = max_digit_run})) {
    out.emplace_back(span);
  }
  return out;
}

using Expected = std::vector<std::string>;

TEST(SelectSplitPatternTest, KnowsBothFamilyPatterns) {
  const auto llama3 = select_split_pattern(kLlama3Pattern);
  ASSERT_TRUE(llama3.ok()) << llama3.status();
  EXPECT_EQ(llama3->max_digit_run, 3);
  const auto qwen2 = select_split_pattern(kQwen2Pattern);
  ASSERT_TRUE(qwen2.ok()) << qwen2.status();
  EXPECT_EQ(qwen2->max_digit_run, 1);
}

TEST(SelectSplitPatternTest, RejectsUnknownPatternsByName) {
  const auto spec = select_split_pattern("[a-z]+");
  ASSERT_FALSE(spec.ok());
  EXPECT_TRUE(IsUnimplemented(spec.status())) << spec.status();
  EXPECT_NE(spec.status().message().find("[a-z]+"), std::string_view::npos)
      << spec.status();
}

TEST(PretokenizeTest, WordsPunctuationAndSpacePrefixes) {
  // A leading space attaches to a word via alternative 2's optional
  // prefix, and to punctuation via alternative 4's literal ' ?'.
  EXPECT_EQ(Split("Hello, world!", 3), (Expected{"Hello", ",", " world", "!"}));
  EXPECT_EQ(Split("a b", 3), (Expected{"a", " b"}));
  // Any single non-CR/LF non-letter/number codepoint glues to a following
  // letter run — including a tab.
  EXPECT_EQ(Split("c\td", 3), (Expected{"c", "\td"}));
}

TEST(PretokenizeTest, DigitRunBoundDiffersBetweenFamilies) {
  EXPECT_EQ(Split("1234567", 3), (Expected{"123", "456", "7"}));
  EXPECT_EQ(Split("1234567", 1), (Expected{"1", "2", "3", "4", "5", "6", "7"}));
  // A space before a digit stands alone: no alternative attaches it.
  EXPECT_EQ(Split("1 23", 3), (Expected{"1", " ", "23"}));
  EXPECT_EQ(Split("x9", 3), (Expected{"x", "9"}));  // letters never take digits
}

TEST(PretokenizeTest, ContractionsAreCaseInsensitive) {
  EXPECT_EQ(Split("isn't", 3), (Expected{"isn", "'t"}));
  EXPECT_EQ(Split("ISN'T", 3), (Expected{"ISN", "'T"}));
  EXPECT_EQ(Split("we're I'll he'd", 3),
            (Expected{"we", "'re", " I", "'ll", " he", "'d"}));
  // Not a contraction suffix: the apostrophe glues to the following letter
  // run via alternative 2's optional prefix instead.
  EXPECT_EQ(Split("o'clock", 3), (Expected{"o", "'clock"}));
}

TEST(PretokenizeTest, TrailingWhitespaceLookahead) {
  // \s+(?!\S): a run before a non-space keeps its last whitespace for the
  // next token's prefix...
  EXPECT_EQ(Split("a  b", 3), (Expected{"a", " ", " b"}));
  EXPECT_EQ(Split("e   \t f", 3), (Expected{"e", "   \t", " f"}));
  // ...at end of input the whole run matches...
  EXPECT_EQ(Split("x   ", 3), (Expected{"x", "   "}));
  // ...and a single space before a non-word falls through to \s+.
  EXPECT_EQ(Split("1 2", 1), (Expected{"1", " ", "2"}));
}

TEST(PretokenizeTest, NewlineClusters) {
  // \s*[\r\n]+ matches through the last CR/LF of a whitespace run.
  EXPECT_EQ(Split("hi\n\nthere", 3), (Expected{"hi", "\n\n", "there"}));
  EXPECT_EQ(Split("a \n \n x", 3), (Expected{"a", " \n \n", " x"}));
  EXPECT_EQ(Split("\r\n", 3), (Expected{"\r\n"}));
  // Punctuation swallows trailing newlines (alternative 4's [\r\n]*).
  EXPECT_EQ(Split("!!\n\n", 3), (Expected{"!!\n\n"}));
  // But newlines never glue to a following word (alternative 2 excludes
  // them from the optional prefix).
  EXPECT_EQ(Split("\na", 3), (Expected{"\n", "a"}));
}

TEST(PretokenizeTest, UnicodeCategoriesDriveTheClasses) {
  // CJK ideographs are letters; a single fullwidth comma is a valid
  // alternative-2 prefix, so it glues to the following run (HF does the
  // same — the cjk golden vector pins it).
  EXPECT_EQ(Split("你好，世界", 3), (Expected{"你好", "，世界"}));
  // NBSP (U+00A0) is not CR/LF/letter/number, so it prefixes a letter run
  // just like a space (it is \s, but alternative 2 is tried first).
  EXPECT_EQ(Split("\u00A0a", 3), (Expected{"\u00A0a"}));
  // Arabic-Indic digits (U+0660..U+0663) are \p{N}, obeying the run bound.
  EXPECT_EQ(Split("\u0660\u0661\u0662\u0663", 3),
            (Expected{"\u0660\u0661\u0662", "\u0663"}));
}

TEST(PretokenizeTest, EmptyAndTotalness) {
  EXPECT_TRUE(Split("", 3).empty());
  // Undecodable bytes become their own spans (encode normally routes them
  // around the matcher; the matcher stays total regardless).
  EXPECT_EQ(Split("a\xFF"
                  "b",
                  3),
            (Expected{"a", "\xFF", "b"}));
}

TEST(PretokenizeTest, SpansConcatenateToTheInput) {
  const std::string input =
      "The 2026 fixture isn't  small:\n\n\tCJK 世界, emoji "
      "\U0001F600, digits 123456789, trailing   ";
  for (const int max_digit_run : {1, 3}) {
    std::string joined;
    for (const std::string& span : Split(input, max_digit_run)) {
      joined += span;
    }
    EXPECT_EQ(joined, input);
  }
}

}  // namespace
