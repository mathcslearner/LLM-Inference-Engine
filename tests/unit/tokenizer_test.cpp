#include "tokenizer/tokenizer.h"

#include "common/paths.h"
#include "core/status.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// M4-T08 (design: docs/design/model-loading.md §6.2, §8): golden parses of
// the committed real Llama-3 and Qwen-2 tokenizer.json fixtures assert
// vocab size, token↔id pairs, and special-token metadata; the §6.2
// component whitelist is exercised by synthesized rejections whose message
// must name the offending component — the actionability criterion is
// asserted, not aspirational.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::core::IsUnimplemented;
using engine::tokenizer::AddedToken;
using engine::tokenizer::Tokenizer;

bool MessageContains(const engine::core::Status& status,
                     std::string_view substr) {
  return status.message().find(substr) != std::string_view::npos;
}

// --- synthetic tokenizer.json assembly -------------------------------------
//
// Mirrors config_test's ConfigJson: assemble a file from raw-JSON section
// values so each test can override or remove one section at a time. The
// default is a minimal valid byte-level BPE tokenizer: vocab {a, b, ab},
// one merge, one special added token.

std::string JsonObject(
    const std::map<std::string, std::string, std::less<>>& fields) {
  std::string json = "{";
  bool first = true;
  for (const auto& [key, value] : fields) {
    if (!first) {
      json += ", ";
    }
    first = false;
    json += "\"";
    json += key;
    json += "\": ";
    json += value;
  }
  json += "}";
  return json;
}

std::string ModelJson(
    const std::map<std::string, std::string, std::less<>>& overrides = {},
    const std::vector<std::string>& removed = {}) {
  std::map<std::string, std::string, std::less<>> fields = {
      {"type", R"("BPE")"},
      {"dropout", "null"},
      {"unk_token", "null"},
      {"continuing_subword_prefix", "null"},
      {"end_of_word_suffix", "null"},
      {"fuse_unk", "false"},
      {"byte_fallback", "false"},
      {"ignore_merges", "false"},
      {"vocab", R"({"a": 0, "b": 1, "ab": 2})"},
      {"merges", R"(["a b"])"},
  };
  for (const auto& [key, value] : overrides) {
    fields[key] = value;
  }
  for (const auto& key : removed) {
    fields.erase(key);
  }
  return JsonObject(fields);
}

std::string TokenizerJson(
    const std::map<std::string, std::string, std::less<>>& overrides = {},
    const std::vector<std::string>& removed = {}) {
  std::map<std::string, std::string, std::less<>> fields = {
      {"version", R"("1.0")"},
      {"truncation", "null"},
      {"padding", "null"},
      {"added_tokens",
       R"([{"id": 3, "content": "<s>", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}])"},
      {"normalizer", "null"},
      // The real Qwen-2 Split pattern: since M4-T09, from_json selects the
      // hand-written matcher against the known pattern strings, so the
      // builder must use one of them (JSON-escaped backslashes).
      {"pre_tokenizer",
       R"({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex":
              "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
             "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false,
             "trim_offsets": true, "use_regex": false}]})"},
      {"post_processor", "null"},
      {"decoder",
       R"({"type": "ByteLevel", "add_prefix_space": true,
           "trim_offsets": true, "use_regex": true})"},
      {"model", ModelJson()},
  };
  for (const auto& [key, value] : overrides) {
    fields[key] = value;
  }
  for (const auto& key : removed) {
    fields.erase(key);
  }
  return JsonObject(fields);
}

// --- fixture tokenizers, loaded once per process ---------------------------

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

// --- Llama-3 goldens -------------------------------------------------------

TEST(TokenizerLlama3Test, VocabSizeCountsBaseAndAddedIds) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  // 128000 base BPE ids + 256 added specials, contiguous.
  EXPECT_EQ(tok->vocab_size(), 128256);
}

TEST(TokenizerLlama3Test, TokenIdPairsRoundTrip) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->token_to_id("Hello"), 9906);
  EXPECT_EQ(tok->token_to_id("ĠHello"), 22691);
  EXPECT_EQ(tok->token_to_id("Ġworld"), 1917);
  EXPECT_EQ(tok->id_to_token(9906), "Hello");
  EXPECT_EQ(tok->id_to_token(1917), "Ġworld");
  EXPECT_EQ(tok->token_to_id("definitely not a token"), std::nullopt);
  EXPECT_EQ(tok->id_to_token(-1), std::nullopt);
  EXPECT_EQ(tok->id_to_token(128256), std::nullopt);
}

TEST(TokenizerLlama3Test, TokenBytesUnmapTheAlphabet) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  // Base-vocab bytes go through the inverse byte-level map (Ġ → space)…
  EXPECT_EQ(tok->token_bytes(1917), " world");
  EXPECT_EQ(tok->token_bytes(9906), "Hello");
  // …added tokens decode to their content verbatim.
  EXPECT_EQ(tok->token_bytes(128000), "<|begin_of_text|>");
  EXPECT_EQ(tok->token_bytes(128256), std::nullopt);
}

TEST(TokenizerLlama3Test, SpecialTokenMetadata) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->token_to_id("<|begin_of_text|>"), 128000);
  EXPECT_EQ(tok->id_to_token(128000), "<|begin_of_text|>");

  const AddedToken* bot = tok->added_token(128000);
  ASSERT_NE(bot, nullptr);
  EXPECT_EQ(bot->content, "<|begin_of_text|>");
  EXPECT_TRUE(bot->special);
  EXPECT_FALSE(bot->lstrip);
  EXPECT_FALSE(bot->rstrip);
  EXPECT_FALSE(bot->normalized);

  const AddedToken* eot = tok->added_token(128001);
  ASSERT_NE(eot, nullptr);
  EXPECT_EQ(eot->content, "<|end_of_text|>");
  EXPECT_TRUE(eot->special);

  EXPECT_EQ(tok->added_token(9906), nullptr);
  EXPECT_TRUE(tok->is_special(128000));
  EXPECT_TRUE(tok->is_special(128255));
  EXPECT_FALSE(tok->is_special(9906));
  EXPECT_FALSE(tok->is_special(-1));
  EXPECT_FALSE(tok->is_special(128256));
}

TEST(TokenizerLlama3Test, TemplateDeclaresBosOnly) {
  const auto& tok = FixtureTokenizer("llama3");
  ASSERT_TRUE(tok.ok()) << tok.status();
  // TemplateProcessing single = [<|begin_of_text|>, $A]: BOS, no EOS. The
  // fixture's `pair` template ($B) is ignored (§6.2 — never encodes pairs).
  EXPECT_EQ(tok->bos_id(), 128000);
  EXPECT_EQ(tok->eos_id(), std::nullopt);
}

// --- Qwen-2 goldens --------------------------------------------------------

TEST(TokenizerQwen2Test, VocabSizeCountsBaseAndAddedIds) {
  const auto& tok = FixtureTokenizer("qwen2");
  ASSERT_TRUE(tok.ok()) << tok.status();
  // 151643 base BPE ids + 3 added specials, contiguous.
  EXPECT_EQ(tok->vocab_size(), 151646);
}

TEST(TokenizerQwen2Test, TokenIdPairsRoundTrip) {
  const auto& tok = FixtureTokenizer("qwen2");
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->token_to_id("Hello"), 9707);
  EXPECT_EQ(tok->token_to_id("Ġworld"), 1879);
  EXPECT_EQ(tok->id_to_token(9707), "Hello");
  EXPECT_EQ(tok->token_bytes(1879), " world");
}

TEST(TokenizerQwen2Test, SpecialTokenMetadata) {
  const auto& tok = FixtureTokenizer("qwen2");
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->token_to_id("<|endoftext|>"), 151643);
  EXPECT_EQ(tok->token_to_id("<|im_start|>"), 151644);
  EXPECT_EQ(tok->token_to_id("<|im_end|>"), 151645);
  for (const std::int32_t id : {151643, 151644, 151645}) {
    const AddedToken* token = tok->added_token(id);
    ASSERT_NE(token, nullptr) << "id " << id;
    EXPECT_TRUE(token->special) << "id " << id;
    EXPECT_FALSE(token->lstrip) << "id " << id;
    EXPECT_FALSE(token->rstrip) << "id " << id;
    EXPECT_TRUE(tok->is_special(id)) << "id " << id;
  }
  EXPECT_FALSE(tok->is_special(9707));
}

TEST(TokenizerQwen2Test, NoTemplateMeansNoBosOrEos) {
  const auto& tok = FixtureTokenizer("qwen2");
  ASSERT_TRUE(tok.ok()) << tok.status();
  // Qwen's post_processor is a bare ByteLevel: no insertions.
  EXPECT_EQ(tok->bos_id(), std::nullopt);
  EXPECT_EQ(tok->eos_id(), std::nullopt);
}

// --- minimal synthetic acceptance ------------------------------------------

TEST(TokenizerParseTest, MinimalTokenizerParses) {
  const auto tok = Tokenizer::from_json(TokenizerJson());
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->vocab_size(), 4);  // {a, b, ab} + added <s>
  EXPECT_EQ(tok->token_to_id("ab"), 2);
  EXPECT_EQ(tok->token_bytes(2), "ab");
  EXPECT_EQ(tok->token_to_id("<s>"), 3);
  ASSERT_NE(tok->added_token(3), nullptr);
  EXPECT_TRUE(tok->added_token(3)->special);
  EXPECT_EQ(tok->bos_id(), std::nullopt);
  EXPECT_EQ(tok->eos_id(), std::nullopt);
}

TEST(TokenizerParseTest, ArrayFormMergesAccepted) {
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"model", ModelJson({{"merges", R"([["a", "b"]])"}})}}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->vocab_size(), 4);
}

TEST(TokenizerParseTest, NfcNormalizerAccepted) {
  const auto plain = Tokenizer::from_json(
      TokenizerJson({{"normalizer", R"({"type": "NFC"})"}}));
  EXPECT_TRUE(plain.ok()) << plain.status();
  const auto sequenced = Tokenizer::from_json(TokenizerJson(
      {{"normalizer",
        R"({"type": "Sequence", "normalizers": [{"type": "NFC"}]})"}}));
  EXPECT_TRUE(sequenced.ok()) << sequenced.status();
}

TEST(TokenizerParseTest, TemplateProcessingDeclaresBosAndEos) {
  // BOS <s> and EOS </s> around $A, with a ByteLevel post-processor in the
  // same Sequence (the Llama-3 shape; ByteLevel is an encoding no-op).
  const std::map<std::string, std::string, std::less<>> overrides = {
      {"added_tokens",
       R"([{"id": 3, "content": "<s>", "special": true},
           {"id": 4, "content": "</s>", "special": true}])"},
      {"post_processor",
       R"({"type": "Sequence", "processors": [
            {"type": "ByteLevel", "add_prefix_space": true,
             "trim_offsets": false, "use_regex": true},
            {"type": "TemplateProcessing",
             "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": "</s>", "type_id": 0}}],
             "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                      {"Sequence": {"id": "B", "type_id": 1}}],
             "special_tokens": {
               "<s>": {"id": "<s>", "ids": [3], "tokens": ["<s>"]},
               "</s>": {"id": "</s>", "ids": [4], "tokens": ["</s>"]}}}]})"},
  };
  const auto tok = Tokenizer::from_json(TokenizerJson(overrides));
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->bos_id(), 3);
  EXPECT_EQ(tok->eos_id(), 4);
}

TEST(TokenizerParseTest, AddedTokenMayShadowVocabEntry) {
  // HF allows an added token whose id/content already exist in the base
  // vocab (Llama-2-style BOS); metadata attaches, mappings are unchanged.
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"added_tokens",
                      R"([{"id": 0, "content": "a", "special": true},
            {"id": 3, "content": "<s>", "special": true}])"}}));
  ASSERT_TRUE(tok.ok()) << tok.status();
  EXPECT_EQ(tok->token_to_id("a"), 0);
  EXPECT_EQ(tok->token_bytes(0), "a");
  EXPECT_TRUE(tok->is_special(0));
}

TEST(TokenizerParseTest, DecodeIsUnimplementedUntilT10) {
  const auto tok = Tokenizer::from_json(TokenizerJson());
  ASSERT_TRUE(tok.ok()) << tok.status();
  const std::vector<std::int32_t> ids = {0, 1};
  const auto decoded = tok->decode(ids, /*skip_special_tokens=*/false);
  EXPECT_TRUE(IsUnimplemented(decoded.status()));
}

TEST(TokenizerRejectTest, UnknownSplitPatternIsUnimplemented) {
  // M4-T09: the Split regex must be one of the known GPT-2/cl100k-family
  // pattern strings the hand-written matcher was validated against.
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"pre_tokenizer",
                      R"({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "[a-z]+"},
             "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false,
             "trim_offsets": true, "use_regex": false}]})"}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsUnimplemented(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "[a-z]+")) << tok.status();
}

// --- unsupported tokenizer types (roadmap acceptance) ----------------------

TEST(TokenizerRejectTest, UnigramModelIsUnimplemented) {
  const auto tok = Tokenizer::from_json(TokenizerJson(
      {{"model", R"({"type": "Unigram", "unk_id": 0, "vocab": []})"}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsUnimplemented(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "Unigram"));
  EXPECT_TRUE(MessageContains(tok.status(), "BPE"));
}

TEST(TokenizerRejectTest, SentencepieceBinaryIsInvalidJson) {
  // A sentencepiece .model file is a protobuf, not JSON (§6.1).
  const std::string binary = {'\x0a', '\x12', '\x08', '\x00', '\xff', 's'};
  const auto tok = Tokenizer::from_json(binary);
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "not valid JSON"));
  EXPECT_TRUE(MessageContains(tok.status(), "sentencepiece"));
}

TEST(TokenizerRejectTest, SentencepieceBinaryFromFileNamesThePath) {
  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / "tokenizer_test_sp.model";
  {
    std::ofstream file(path, std::ios::binary);
    file << '\x0a' << '\x12' << "not json at all";
  }
  const auto tok = Tokenizer::from_file(path);
  std::filesystem::remove(path);
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "tokenizer_test_sp.model"));
  EXPECT_TRUE(MessageContains(tok.status(), "not valid JSON"));
}

TEST(TokenizerRejectTest, MissingFileIsNotFound) {
  const auto tok = Tokenizer::from_file("/no/such/dir/tokenizer.json");
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsNotFound(tok.status())) << tok.status();
}

TEST(TokenizerRejectTest, NonObjectRootIsInvalid) {
  const auto tok = Tokenizer::from_json("[1, 2]");
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
}

// --- component whitelist (§6.2) --------------------------------------------

struct RejectCase {
  const char* name;
  std::map<std::string, std::string, std::less<>> overrides;
  bool want_unimplemented;  // else InvalidArgument
  const char* want_substring;
};

class TokenizerWhitelistTest : public ::testing::TestWithParam<RejectCase> {};

TEST_P(TokenizerWhitelistTest, RejectsWithActionableMessage) {
  const RejectCase& test_case = GetParam();
  const auto tok = Tokenizer::from_json(TokenizerJson(test_case.overrides));
  ASSERT_FALSE(tok.ok());
  if (test_case.want_unimplemented) {
    EXPECT_TRUE(IsUnimplemented(tok.status())) << tok.status();
  } else {
    EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  }
  EXPECT_TRUE(MessageContains(tok.status(), test_case.want_substring))
      << tok.status();
}

INSTANTIATE_TEST_SUITE_P(
    Components, TokenizerWhitelistTest,
    ::testing::Values(
        RejectCase{.name = "TruncationConfigured",
                   .overrides = {{"truncation", R"({"max_length": 512})"}},
                   .want_unimplemented = true,
                   .want_substring = "truncation"},
        RejectCase{.name = "PaddingConfigured",
                   .overrides = {{"padding", R"({"strategy": "longest"})"}},
                   .want_unimplemented = true,
                   .want_substring = "padding"},
        RejectCase{.name = "NfkcNormalizer",
                   .overrides = {{"normalizer", R"({"type": "NFKC"})"}},
                   .want_unimplemented = true,
                   .want_substring = "NFKC"},
        RejectCase{
            .name = "UnknownPreTokenizer",
            .overrides = {{"pre_tokenizer", R"({"type": "Whitespace"})"}},
            .want_unimplemented = true,
            .want_substring = "Whitespace"},
        RejectCase{.name = "MissingPreTokenizer",
                   .overrides = {{"pre_tokenizer", "null"}},
                   .want_unimplemented = true,
                   .want_substring = "pre_tokenizer"},
        RejectCase{.name = "SplitInvert",
                   .overrides = {{"pre_tokenizer",
                                  R"({"type": "Sequence", "pretokenizers": [
                       {"type": "Split", "pattern": {"Regex": "pat"},
                        "behavior": "Isolated", "invert": true},
                       {"type": "ByteLevel", "add_prefix_space": false,
                        "use_regex": false}]})"}},
                   .want_unimplemented = true,
                   .want_substring = "invert"},
        RejectCase{.name = "SplitMergedBehavior",
                   .overrides = {{"pre_tokenizer",
                                  R"({"type": "Sequence", "pretokenizers": [
                       {"type": "Split", "pattern": {"Regex": "pat"},
                        "behavior": "MergedWithPrevious", "invert": false},
                       {"type": "ByteLevel", "add_prefix_space": false,
                        "use_regex": false}]})"}},
                   .want_unimplemented = true,
                   .want_substring = "MergedWithPrevious"},
        RejectCase{.name = "SplitStringPattern",
                   .overrides = {{"pre_tokenizer",
                                  R"({"type": "Sequence", "pretokenizers": [
                       {"type": "Split", "pattern": {"String": " "},
                        "behavior": "Isolated", "invert": false},
                       {"type": "ByteLevel", "add_prefix_space": false,
                        "use_regex": false}]})"}},
                   .want_unimplemented = true,
                   .want_substring = "Regex"},
        RejectCase{.name = "ByteLevelPrefixSpace",
                   .overrides = {{"pre_tokenizer",
                                  R"({"type": "Sequence", "pretokenizers": [
                       {"type": "Split", "pattern": {"Regex": "pat"},
                        "behavior": "Isolated", "invert": false},
                       {"type": "ByteLevel", "add_prefix_space": true,
                        "use_regex": false}]})"}},
                   .want_unimplemented = true,
                   .want_substring = "add_prefix_space"},
        RejectCase{.name = "UnknownPostProcessor",
                   .overrides = {{"post_processor",
                                  R"({"type": "RobertaProcessing"})"}},
                   .want_unimplemented = true,
                   .want_substring = "RobertaProcessing"},
        RejectCase{.name = "TemplateUnknownSpecialToken",
                   .overrides = {{"post_processor",
                                  R"({"type": "TemplateProcessing",
                      "single": [{"SpecialToken": {"id": "<missing>",
                                                   "type_id": 0}},
                                 {"Sequence": {"id": "A", "type_id": 0}}],
                      "special_tokens": {}})"}},
                   .want_unimplemented = false,
                   .want_substring = "<missing>"},
        RejectCase{.name = "UnknownDecoder",
                   .overrides = {{"decoder", R"({"type": "WordPiece"})"}},
                   .want_unimplemented = true,
                   .want_substring = "WordPiece"},
        RejectCase{
            .name = "ByteFallback",
            .overrides = {{"model", ModelJson({{"byte_fallback", "true"}})}},
            .want_unimplemented = true,
            .want_substring = "byte_fallback"},
        RejectCase{
            .name = "UnkToken",
            .overrides = {{"model", ModelJson({{"unk_token", R"("<unk>")"}})}},
            .want_unimplemented = true,
            .want_substring = "unk_token"},
        RejectCase{.name = "Dropout",
                   .overrides = {{"model", ModelJson({{"dropout", "0.1"}})}},
                   .want_unimplemented = true,
                   .want_substring = "dropout"},
        RejectCase{
            .name = "SubwordPrefix",
            .overrides = {{"model", ModelJson({{"continuing_subword_prefix",
                                                R"("##")"}})}},
            .want_unimplemented = true,
            .want_substring = "continuing_subword_prefix"},
        RejectCase{.name = "SingleWordAddedToken",
                   .overrides = {{"added_tokens",
                                  R"([{"id": 3, "content": "<s>",
                                "single_word": true, "special": true}])"}},
                   .want_unimplemented = true,
                   .want_substring = "single_word"}),
    [](const ::testing::TestParamInfo<RejectCase>& info) {
      return info.param.name;
    });

// --- malformed vocab / merges / added_tokens -------------------------------

TEST(TokenizerRejectTest, VocabIdGapIsInvalid) {
  const auto tok = Tokenizer::from_json(TokenizerJson(
      {{"model",
        ModelJson({{"vocab", R"({"a": 0, "ab": 2})"}, {"merges", "[]"}})}},
      {"added_tokens"}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "model.vocab"));
  EXPECT_TRUE(MessageContains(tok.status(), "contiguous"));
}

TEST(TokenizerRejectTest, VocabDuplicateIdIsInvalid) {
  const auto tok = Tokenizer::from_json(TokenizerJson(
      {{"model",
        ModelJson({{"vocab", R"({"a": 0, "b": 0})"}, {"merges", "[]"}})}},
      {"added_tokens"}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "more than one token"));
}

TEST(TokenizerRejectTest, NonAlphabetVocabTokenIsInvalid) {
  // A raw space (U+0020) is not in the byte-level alphabet — a vocab
  // containing one is not byte-level BPE output. (Merges emptied so the
  // alphabet check, not a merge-consistency error, is what fires.)
  const auto tok = Tokenizer::from_json(TokenizerJson(
      {{"model", ModelJson({{"vocab", R"({"a": 0, "b": 1, "a b": 2})"},
                            {"merges", "[]"}})}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "alphabet"));
}

TEST(TokenizerRejectTest, MergeWithoutSpaceIsInvalid) {
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"model", ModelJson({{"merges", R"(["ab"])"}})}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "model.merges[0]"));
}

TEST(TokenizerRejectTest, MergeReferencingUnknownTokenIsInvalid) {
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"model", ModelJson({{"merges", R"(["a c"])"}})}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "\"c\""));
}

TEST(TokenizerRejectTest, MergeProductMissingFromVocabIsInvalid) {
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"model", ModelJson({{"vocab", R"({"a": 0, "b": 1})"}})}},
                    {"added_tokens"}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "product"));
}

TEST(TokenizerRejectTest, AddedTokenIdGapIsInvalid) {
  const auto tok = Tokenizer::from_json(TokenizerJson(
      {{"added_tokens", R"([{"id": 5, "content": "<s>", "special": true}])"}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "gap"));
}

TEST(TokenizerRejectTest, AddedTokenDuplicateIdIsInvalid) {
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"added_tokens",
                      R"([{"id": 3, "content": "<s>", "special": true},
            {"id": 3, "content": "</s>", "special": true}])"}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "id 3"));
}

TEST(TokenizerRejectTest, AddedTokenContentCollisionIsInvalid) {
  // Id 0 exists in the base vocab as "a"; an added token claiming id 0
  // with different content contradicts the file.
  const auto tok = Tokenizer::from_json(
      TokenizerJson({{"added_tokens",
                      R"([{"id": 0, "content": "zzz", "special": true},
            {"id": 3, "content": "<s>", "special": true}])"}}));
  ASSERT_FALSE(tok.ok());
  EXPECT_TRUE(IsInvalidArgument(tok.status())) << tok.status();
  EXPECT_TRUE(MessageContains(tok.status(), "collides"));
}

}  // namespace
