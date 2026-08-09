#include "common/paths.h"
#include "core/status.h"
#include "tokenizer/tokenizer.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// M4-T09 acceptance (design: docs/design/model-loading.md §6.4, §7): encode
// must be byte-identical to HF `tokenizers` for every committed test vector
// of both fixture families — with and without special tokens. The
// `encode_synthetic` vectors (invalid UTF-8, which HF rejects at its API
// boundary) pin this engine's own documented semantics instead. Synthetic
// micro-tokenizers then cover the pieces goldens cannot isolate: merge
// order, ignore_merges, added-token flag semantics, template suffixes.

namespace {

using engine::core::IsInvalidArgument;
using engine::tokenizer::Tokenizer;

// --- golden vectors ---------------------------------------------------------

const engine::core::StatusOr<Tokenizer>& FixtureTokenizer(
    std::string_view family) {
  static const auto* llama3 = new engine::core::StatusOr<Tokenizer>(
      Tokenizer::from_file(engine::testing::FixturesDir() / "tokenizers" /
                           "llama3" / "tokenizer.json"));
  static const auto* qwen2 = new engine::core::StatusOr<Tokenizer>(
      Tokenizer::from_file(engine::testing::FixturesDir() / "tokenizers" /
                           "qwen2" / "tokenizer.json"));
  return family == "llama3" ? *llama3 : *qwen2;
}

nlohmann::json LoadVectors(std::string_view family) {
  const std::filesystem::path path =
      engine::testing::FixturesDir() / "tokenizers" / family / "vectors.json";
  const std::ifstream file(path, std::ios::binary);
  EXPECT_TRUE(file) << path;
  std::ostringstream contents;
  contents << file.rdbuf();
  nlohmann::json root = nlohmann::json::parse(contents.str(), /*cb=*/nullptr,
                                              /*allow_exceptions=*/false);
  EXPECT_FALSE(root.is_discarded()) << path;
  return root;
}

// Strict base64 (the fixture generator emits standard padding).
std::string Base64Decode(std::string_view encoded) {
  constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::uint32_t accumulator = 0;
  int bits = 0;
  for (const char c : encoded) {
    if (c == '=') {
      break;
    }
    const auto value = kAlphabet.find(c);
    EXPECT_NE(value, std::string_view::npos) << "bad base64: " << encoded;
    accumulator = (accumulator << 6U) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((accumulator >> bits) & 0xFFU));
    }
  }
  return out;
}

std::vector<std::int32_t> IdList(const nlohmann::json& array) {
  std::vector<std::int32_t> ids;
  for (const auto& id : array) {
    ids.push_back(id.get<std::int32_t>());
  }
  return ids;
}

class GoldenEncodeTest : public ::testing::TestWithParam<std::string_view> {};

TEST_P(GoldenEncodeTest, MatchesEveryCommittedVector) {
  const auto& tok = FixtureTokenizer(GetParam());
  ASSERT_TRUE(tok.ok()) << tok.status();
  const nlohmann::json vectors = LoadVectors(GetParam());
  ASSERT_TRUE(vectors.contains("cases"));
  std::size_t cases = 0;
  for (const auto& c : vectors["cases"]) {
    const std::string name = c["name"].get<std::string>();
    const std::string text =
        c.contains("text_b64") ? Base64Decode(c["text_b64"].get<std::string>())
                               : c["text"].get<std::string>();
    const auto plain = tok->encode(text, /*add_special_tokens=*/false);
    ASSERT_TRUE(plain.ok()) << name << ": " << plain.status();
    EXPECT_EQ(*plain, IdList(c["ids"])) << name;
    const auto with_special = tok->encode(text, /*add_special_tokens=*/true);
    ASSERT_TRUE(with_special.ok()) << name << ": " << with_special.status();
    EXPECT_EQ(*with_special, IdList(c["ids_with_special"])) << name;
    ++cases;
  }
  // The committed suites are 24 cases each; a shrunken file must not pass
  // silently.
  EXPECT_GE(cases, 24U);
}

INSTANTIATE_TEST_SUITE_P(Families, GoldenEncodeTest,
                         ::testing::Values("llama3", "qwen2"),
                         // Not `info`: it shadows the generated gtest
                         // function's parameter under GCC's -Wshadow.
                         [](const auto& param_info) {
                           return std::string(param_info.param);
                         });

// --- synthetic micro-tokenizers ---------------------------------------------

// A minimal byte-level BPE tokenizer.json around the real Qwen-2 Split
// pattern (selection is enforced at parse since M4-T09). All inputs are
// raw JSON fragments.
struct MiniSpec {
  std::string vocab;
  std::string merges = "[]";
  std::string added = "[]";
  bool ignore_merges = false;
  bool nfc = false;
  std::string post_processor = "null";
};

std::string MiniJson(const MiniSpec& spec) {
  std::string json = R"({
    "version": "1.0", "truncation": null, "padding": null,
    "added_tokens": )";
  json += spec.added;
  json += R"(, "normalizer": )";
  json += spec.nfc ? R"({"type": "NFC"})" : "null";
  json += R"(, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
      {"type": "Split", "pattern": {"Regex":
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
       "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false,
       "trim_offsets": true, "use_regex": false}]},
    "post_processor": )";
  json += spec.post_processor;
  json += R"(, "decoder": null, "model": {"type": "BPE",
    "ignore_merges": )";
  json += spec.ignore_merges ? "true" : "false";
  json += R"(, "vocab": )";
  json += spec.vocab;
  json += R"(, "merges": )";
  json += spec.merges;
  json += "}}";
  return json;
}

using Ids = std::vector<std::int32_t>;

TEST(EncodeMergeLoopTest, MergesLowestRankLeftmostFirst) {
  // "aaa" with the single merge (a, a): the leftmost occurrence merges
  // first, so the result is [aa, a] — never [a, aa].
  const auto tok = Tokenizer::from_json(
      MiniJson({.vocab = R"({"a": 0, "aa": 1})", .merges = R"(["a a"])"}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  const auto ids = tok->encode("aaa", /*add_special_tokens=*/false);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, (Ids{1, 0}));
}

TEST(EncodeMergeLoopTest, RankOrderBeatsTextOrder) {
  // "abc": pair (b, c) has rank 0, (a, b) rank 1 — the lower rank merges
  // first even though (a, b) is leftmost; then (a, bc) completes.
  const auto tok = Tokenizer::from_json(MiniJson(
      {.vocab = R"({"a": 0, "b": 1, "c": 2, "bc": 3, "ab": 4, "abc": 5})",
       .merges = R"(["b c", "a b", "a bc"])"}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  const auto ids = tok->encode("abc", /*add_special_tokens=*/false);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, (Ids{5}));
}

TEST(EncodeMergeLoopTest, IgnoreMergesShortCircuitsVocabHits) {
  // With ignore_merges, a whole pre-token found in the vocab skips the
  // merge loop entirely (no merges exist here, so only that path works).
  const auto with_flag = Tokenizer::from_json(MiniJson(
      {.vocab = R"({"a": 0, "b": 1, "ab": 2})", .ignore_merges = true}));
  ASSERT_TRUE(with_flag.ok()) << with_flag.status();
  const auto hit = with_flag->encode("ab", /*add_special_tokens=*/false);
  ASSERT_TRUE(hit.ok()) << hit.status();
  EXPECT_EQ(*hit, (Ids{2}));
  // Without the flag (and no merges) the same word stays two symbols.
  const auto without_flag =
      Tokenizer::from_json(MiniJson({.vocab = R"({"a": 0, "b": 1, "ab": 2})"}));
  ASSERT_TRUE(without_flag.ok()) << without_flag.status();
  const auto split = without_flag->encode("ab", /*add_special_tokens=*/false);
  ASSERT_TRUE(split.ok()) << split.status();
  EXPECT_EQ(*split, (Ids{0, 1}));
}

TEST(EncodeMergeLoopTest, MissingSymbolIsInvalidArgument) {
  // A vocab without all 256 byte tokens is not complete byte-level BPE;
  // encoding a byte it cannot represent must fail loudly, naming it.
  const auto tok =
      Tokenizer::from_json(MiniJson({.vocab = R"({"a": 0, "b": 1})"}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  const auto ids = tok->encode("c", /*add_special_tokens=*/false);
  ASSERT_FALSE(ids.ok());
  EXPECT_TRUE(IsInvalidArgument(ids.status())) << ids.status();
  EXPECT_NE(ids.status().message().find("\"c\""), std::string_view::npos)
      << ids.status();
}

TEST(EncodeAddedTokenTest, LongestContentWinsAtOnePosition) {
  const auto tok = Tokenizer::from_json(
      MiniJson({.vocab = R"({"a": 0, "b": 1, "ab": 2})",
                .added = R"([{"id": 3, "content": "<x>", "special": true},
                    {"id": 4, "content": "<x>y", "special": true}])"}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  const auto longer = tok->encode("<x>y", /*add_special_tokens=*/false);
  ASSERT_TRUE(longer.ok()) << longer.status();
  EXPECT_EQ(*longer, (Ids{4}));
  const auto shorter = tok->encode("<x>a", /*add_special_tokens=*/false);
  ASSERT_TRUE(shorter.ok()) << shorter.status();
  EXPECT_EQ(*shorter, (Ids{3, 0}));
}

TEST(EncodeAddedTokenTest, LstripRstripConsumeAdjacentWhitespace) {
  // "Ġ" is the alphabet image of the space byte, so the no-flag variant
  // can encode the whitespace the flags would otherwise consume.
  const std::string vocab = R"({"a": 0, "b": 1, "ab": 2, "Ġ": 3})";
  const auto stripping = Tokenizer::from_json(MiniJson(
      {.vocab = vocab, .added = R"([{"id": 4, "content": "<s>", "special": true,
                     "lstrip": true, "rstrip": true}])"}));
  ASSERT_TRUE(stripping.ok()) << stripping.status();
  const auto stripped = stripping->encode("a <s> b",
                                          /*add_special_tokens=*/false);
  ASSERT_TRUE(stripped.ok()) << stripped.status();
  EXPECT_EQ(*stripped, (Ids{0, 4, 1}));

  const auto plain = Tokenizer::from_json(
      MiniJson({.vocab = vocab,
                .added = R"([{"id": 4, "content": "<s>", "special": true}])"}));
  ASSERT_TRUE(plain.ok()) << plain.status();
  const auto kept = plain->encode("a <s> b", /*add_special_tokens=*/false);
  ASSERT_TRUE(kept.ok()) << kept.status();
  // Segments "a " and " b": [a, Ġ] before the token, [Ġ, b] after.
  EXPECT_EQ(*kept, (Ids{0, 3, 4, 3, 1}));
}

TEST(EncodeAddedTokenTest, NormalizedTokensMatchAfterNfc) {
  // Content is composed é (U+00E9); the input is decomposed e + U+0301.
  // With normalized: true the token must match the text that only becomes
  // é *after* NFC. Alphabet symbols Ã/© are the images of composed é's
  // UTF-8 bytes 0xC3/0xA9, for the non-matching control below.
  const std::string vocab = R"({"a": 0, "Ã": 1, "©": 2})";
  const auto matching = Tokenizer::from_json(
      MiniJson({.vocab = vocab,
                .added = R"([{"id": 3, "content": "é", "special": true,
                     "normalized": true}])",
                .nfc = true}));
  ASSERT_TRUE(matching.ok()) << matching.status();
  const auto matched = matching->encode("é",
                                        /*add_special_tokens=*/false);
  ASSERT_TRUE(matched.ok()) << matched.status();
  EXPECT_EQ(*matched, (Ids{3}));

  // normalized: false only sees the raw (decomposed) text, so it must NOT
  // match — the composed bytes go through BPE instead.
  const auto raw_only = Tokenizer::from_json(
      MiniJson({.vocab = vocab,
                .added = R"([{"id": 3, "content": "é", "special": true,
                     "normalized": false}])",
                .nfc = true}));
  ASSERT_TRUE(raw_only.ok()) << raw_only.status();
  const auto unmatched = raw_only->encode("é",
                                          /*add_special_tokens=*/false);
  ASSERT_TRUE(unmatched.ok()) << unmatched.status();
  EXPECT_EQ(*unmatched, (Ids{1, 2}));  // the two bytes of composed é
}

TEST(EncodeTemplateTest, PrefixAndSuffixInsertOnlyWhenRequested) {
  const auto tok = Tokenizer::from_json(
      MiniJson({.vocab = R"({"a": 0, "b": 1, "ab": 2})",
                .added = R"([{"id": 3, "content": "<s>", "special": true},
                    {"id": 4, "content": "</s>", "special": true}])",
                .post_processor = R"({"type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                   {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        "special_tokens": {
          "<s>": {"id": "<s>", "ids": [3], "tokens": ["<s>"]},
          "</s>": {"id": "</s>", "ids": [4], "tokens": ["</s>"]}}})"}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  const auto plain = tok->encode("ab", /*add_special_tokens=*/false);
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_EQ(*plain, (Ids{0, 1}));
  const auto wrapped = tok->encode("ab", /*add_special_tokens=*/true);
  ASSERT_TRUE(wrapped.ok()) << wrapped.status();
  EXPECT_EQ(*wrapped, (Ids{3, 0, 1, 4}));
  const auto empty = tok->encode("", /*add_special_tokens=*/true);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(*empty, (Ids{3, 4}));
}

}  // namespace
