#include "common/paths.h"
#include "core/status.h"
#include "tokenizer/detokenize.h"
#include "tokenizer/tokenizer.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// M4-T10 acceptance (design: docs/design/model-loading.md §6.5, §8): decode
// must reproduce HF's golden output for every committed vector of both
// fixture families — decode(ids, skip=false) == `decoded` and
// decode(ids_with_special, skip=true) == `decoded_skip_special`, the exact
// pairing the generator recorded — and the streaming detokenizer must emit
// only valid UTF-8 (checked by a test-local validator, independent of the
// production classifier) whose concatenation is byte-identical to batch
// decode. Synthetic micro-tokenizers cover what the goldens cannot isolate:
// non-special added tokens under skip_special_tokens.

namespace {

using engine::core::IsInvalidArgument;
using engine::tokenizer::DetokenizerStream;
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

std::vector<std::int32_t> IdList(const nlohmann::json& array) {
  std::vector<std::int32_t> ids;
  for (const auto& id : array) {
    ids.push_back(id.get<std::int32_t>());
  }
  return ids;
}

// Test-local strict UTF-8 validator (deliberately not the production
// classifier — the acceptance criterion asks for an independent check).
// Table-free: decode each sequence and re-check shortest-form and scalar
// range by hand.
bool IsValidUtf8(std::string_view text) {
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto lead = static_cast<unsigned char>(text[pos]);
    std::size_t length = 0;
    std::uint32_t codepoint = 0;
    if (lead < 0x80) {
      length = 1;
      codepoint = lead;
    } else if ((lead & 0xE0U) == 0xC0U) {
      length = 2;
      codepoint = lead & 0x1FU;
    } else if ((lead & 0xF0U) == 0xE0U) {
      length = 3;
      codepoint = lead & 0x0FU;
    } else if ((lead & 0xF8U) == 0xF0U) {
      length = 4;
      codepoint = lead & 0x07U;
    } else {
      return false;
    }
    if (pos + length > text.size()) {
      return false;
    }
    for (std::size_t i = 1; i < length; ++i) {
      const auto cont = static_cast<unsigned char>(text[pos + i]);
      if ((cont & 0xC0U) != 0x80U) {
        return false;
      }
      codepoint = (codepoint << 6U) | (cont & 0x3FU);
    }
    constexpr std::uint32_t kShortestForm[] = {0x0, 0x80, 0x800, 0x10000};
    if (codepoint < kShortestForm[length - 1] ||  // overlong
        (codepoint >= 0xD800 && codepoint <= 0xDFFF) || codepoint > 0x10FFFF) {
      return false;
    }
    pos += length;
  }
  return true;
}

using Ids = std::vector<std::int32_t>;

class GoldenDecodeTest : public ::testing::TestWithParam<std::string_view> {};

TEST_P(GoldenDecodeTest, MatchesEveryCommittedVector) {
  const auto& tok = FixtureTokenizer(GetParam());
  ASSERT_TRUE(tok.ok()) << tok.status();
  const nlohmann::json vectors = LoadVectors(GetParam());
  ASSERT_TRUE(vectors.contains("cases"));
  std::size_t cases = 0;
  for (const auto& c : vectors["cases"]) {
    const std::string name = c["name"].get<std::string>();
    const auto plain =
        tok->decode(IdList(c["ids"]), /*skip_special_tokens=*/false);
    ASSERT_TRUE(plain.ok()) << name << ": " << plain.status();
    EXPECT_EQ(*plain, c["decoded"].get<std::string>()) << name;
    const auto skipped = tok->decode(IdList(c["ids_with_special"]),
                                     /*skip_special_tokens=*/true);
    ASSERT_TRUE(skipped.ok()) << name << ": " << skipped.status();
    EXPECT_EQ(*skipped, c["decoded_skip_special"].get<std::string>()) << name;
    ++cases;
  }
  // The committed suites are 24 cases each; a shrunken file must not pass
  // silently.
  EXPECT_GE(cases, 24U);
}

TEST_P(GoldenDecodeTest, RoundTripsEveryEncodableVector) {
  const auto& tok = FixtureTokenizer(GetParam());
  ASSERT_TRUE(tok.ok()) << tok.status();
  for (const auto& c : LoadVectors(GetParam())["cases"]) {
    if (c.contains("text_b64")) {
      continue;  // invalid UTF-8 round-trips only up to U+FFFD (goldens)
    }
    const std::string text = c["text"].get<std::string>();
    const auto ids = tok->encode(text, /*add_special_tokens=*/false);
    ASSERT_TRUE(ids.ok()) << c["name"] << ": " << ids.status();
    const auto decoded = tok->decode(*ids, /*skip_special_tokens=*/false);
    ASSERT_TRUE(decoded.ok()) << c["name"] << ": " << decoded.status();
    EXPECT_EQ(*decoded, text) << c["name"];
  }
}

// Token-by-token streaming over every vector, both golden pairings: each
// emission must be valid UTF-8 on its own, and the concatenation plus
// finish() must equal batch decode exactly (§6.5's invariant).
TEST_P(GoldenDecodeTest, StreamingMatchesBatchDecodeTokenByToken) {
  const auto& tok = FixtureTokenizer(GetParam());
  ASSERT_TRUE(tok.ok()) << tok.status();
  for (const auto& c : LoadVectors(GetParam())["cases"]) {
    const std::string name = c["name"].get<std::string>();
    for (const bool skip : {false, true}) {
      const Ids ids = IdList(c[skip ? "ids_with_special" : "ids"]);
      const auto batch = tok->decode(ids, skip);
      ASSERT_TRUE(batch.ok()) << name << ": " << batch.status();
      DetokenizerStream stream(*tok, skip);
      std::string streamed;
      for (const std::int32_t id : ids) {
        const auto emission = stream.push(id);
        ASSERT_TRUE(emission.ok()) << name << ": " << emission.status();
        EXPECT_TRUE(IsValidUtf8(*emission)) << name << " id " << id;
        streamed += *emission;
      }
      streamed += stream.finish();
      EXPECT_EQ(streamed, *batch) << name << " skip=" << skip;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Families, GoldenDecodeTest,
                         ::testing::Values("llama3", "qwen2"),
                         [](const auto& info) {
                           return std::string(info.param);
                         });

// --- the multi-token emoji, pinned explicitly -------------------------------

// U+1F469 WOMAN is F0 9F 91 A9 — three llama3 tokens covering [F0 9F],
// [91], [A9]. The stream must stay silent until the last byte arrives.
constexpr std::int32_t kWomanTokens[] = {9468, 239, 102};

TEST(DetokenizerStreamTest, BuffersAcrossAMultiTokenEmoji) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  ASSERT_TRUE(
      tok->decode(kWomanTokens, /*skip_special_tokens=*/false).value() ==
      "\U0001F469");
  DetokenizerStream stream(*tok, /*skip_special_tokens=*/false);
  EXPECT_EQ(stream.push(kWomanTokens[0]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[1]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[2]).value(), "\U0001F469");
  EXPECT_EQ(stream.finish(), "");
}

TEST(DetokenizerStreamTest, FinishReplacesATruncatedTailWithReplacement) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  DetokenizerStream stream(*tok, /*skip_special_tokens=*/false);
  // Two of the emoji's three tokens: bytes F0 9F 91, a valid prefix.
  EXPECT_EQ(stream.push(kWomanTokens[0]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[1]).value(), "");
  EXPECT_EQ(stream.finish(), "�");
  // Batch decode of the same truncated ids agrees (§6.5's invariant), and
  // the carry is reset: the stream is reusable afterwards.
  EXPECT_EQ(tok->decode(std::span(kWomanTokens).first(2),
                        /*skip_special_tokens=*/false)
                .value(),
            "�");
  EXPECT_EQ(stream.finish(), "");
  EXPECT_EQ(stream.push(kWomanTokens[0]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[1]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[2]).value(), "\U0001F469");
}

TEST(DetokenizerStreamTest, SkipsSpecialTokensSilently) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  const std::int32_t bos = tok->bos_id().value_or(-1);
  ASSERT_NE(bos, -1);
  DetokenizerStream stream(*tok, /*skip_special_tokens=*/true);
  EXPECT_EQ(stream.push(bos).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[0]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[1]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[2]).value(), "\U0001F469");
}

// --- errors -----------------------------------------------------------------

TEST(DecodeErrorTest, OutOfRangeIdsAreInvalidArgument) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  const auto vocab = tok->vocab_size();
  for (const std::int32_t bad :
       {std::int32_t{-1}, static_cast<std::int32_t>(vocab)}) {
    const auto decoded =
        tok->decode(Ids{40, bad}, /*skip_special_tokens=*/false);
    ASSERT_FALSE(decoded.ok()) << bad;
    EXPECT_TRUE(IsInvalidArgument(decoded.status())) << decoded.status();
    EXPECT_NE(decoded.status().message().find(std::to_string(bad)),
              std::string_view::npos)
        << decoded.status();
  }
}

TEST(DecodeErrorTest, StreamRejectsOutOfRangeIdsAndKeepsItsState) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  DetokenizerStream stream(*tok, /*skip_special_tokens=*/false);
  EXPECT_EQ(stream.push(kWomanTokens[0]).value(), "");
  const auto bad = stream.push(-1);
  ASSERT_FALSE(bad.ok());
  EXPECT_TRUE(IsInvalidArgument(bad.status())) << bad.status();
  // The rejected push left the carried prefix intact.
  EXPECT_EQ(stream.push(kWomanTokens[1]).value(), "");
  EXPECT_EQ(stream.push(kWomanTokens[2]).value(), "\U0001F469");
}

// --- synthetic: non-special added tokens under skip_special_tokens ----------

// A minimal byte-level BPE tokenizer.json around the real Qwen-2 Split
// pattern (as in tokenizer_encode_test): one special and one non-special
// added token — skip_special_tokens must drop only the former.
constexpr std::string_view kMiniJson = R"({
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 3, "content": "<s>", "special": true},
    {"id": 4, "content": "<tool>", "special": false}],
  "normalizer": null,
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex":
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
     "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false,
     "trim_offsets": true, "use_regex": false}]},
  "post_processor": null, "decoder": null,
  "model": {"type": "BPE", "ignore_merges": false,
    "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": []}})";

TEST(DecodeAddedTokenTest, SkipSpecialDropsOnlySpecialAddedTokens) {
  const auto tok = Tokenizer::from_json(kMiniJson);
  ASSERT_TRUE(tok.ok()) << tok.status();
  const Ids ids{3, 0, 4, 1};
  const auto kept = tok->decode(ids, /*skip_special_tokens=*/false);
  ASSERT_TRUE(kept.ok()) << kept.status();
  EXPECT_EQ(*kept, "<s>a<tool>b");
  const auto skipped = tok->decode(ids, /*skip_special_tokens=*/true);
  ASSERT_TRUE(skipped.ok()) << skipped.status();
  EXPECT_EQ(*skipped, "a<tool>b");

  DetokenizerStream stream(*tok, /*skip_special_tokens=*/true);
  std::string streamed;
  for (const std::int32_t id : ids) {
    streamed += stream.push(id).value();
  }
  streamed += stream.finish();
  EXPECT_EQ(streamed, "a<tool>b");
}

}  // namespace
