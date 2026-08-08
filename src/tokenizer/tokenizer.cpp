#include "tokenizer/tokenizer.h"

#include "tokenizer/bpe.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::tokenizer {
namespace {

using nlohmann::json;

// --- field accessors -------------------------------------------------------
//
// External input, so every accessor is exception-free (design §2): presence
// via find(), type via is_*() before any get<>(). `field` is the dotted
// label used in messages ("model.vocab"), which is what makes the
// actionability criterion testable.

core::Status MissingField(std::string_view field) {
  return core::InvalidArgumentError("missing required field \"{}\"", field);
}

core::Status WrongType(std::string_view field, const json& value,
                       std::string_view want) {
  return core::InvalidArgumentError("field \"{}\" must be {}, got {}", field,
                                    want, value.type_name());
}

core::StatusOr<std::int64_t> IntValue(const json& value,
                                      std::string_view field) {
  if (!value.is_number_integer()) {
    return WrongType(field, value, "an integer");
  }
  if (value.is_number_unsigned() &&
      value.get<std::uint64_t>() >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max())) {
    return core::InvalidArgumentError("field \"{}\" is out of range", field);
  }
  return value.get<std::int64_t>();
}

core::StatusOr<std::string> StringValue(const json& value,
                                        std::string_view field) {
  if (!value.is_string()) {
    return WrongType(field, value, "a string");
  }
  return value.get<std::string>();
}

// Leaves `out` untouched when absent; errors on a present non-boolean.
core::Status OptionalBool(const json& obj, std::string_view key,
                          std::string_view label, bool& out) {
  const auto it = obj.find(key);
  if (it == obj.end()) {
    return core::OkStatus();
  }
  if (!it->is_boolean()) {
    return WrongType(label, *it, "a boolean");
  }
  out = it->get<bool>();
  return core::OkStatus();
}

// --- whitelist guards (§6.2) -----------------------------------------------
//
// Off-whitelist knobs are rejected by name — each of these changes encode
// semantics, and accepting-and-ignoring one is how byte-identity dies.

// `key` must be absent or null (e.g. model.dropout, model.unk_token).
core::Status RequireAbsentOrNull(const json& obj, std::string_view key,
                                 std::string_view label,
                                 std::string_view because) {
  const auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) {
    return core::OkStatus();
  }
  return core::UnimplementedError("\"{}\" is set; {}", label, because);
}

// `key` must be absent, null, or the empty string (the two serializations
// of "no affix" seen in the wild: Llama 3 writes null, Qwen 2 writes "").
core::Status RequireAbsentOrEmpty(const json& obj, std::string_view key,
                                  std::string_view label) {
  const auto it = obj.find(key);
  if (it == obj.end() || it->is_null() ||
      (it->is_string() && it->get<std::string_view>().empty())) {
    return core::OkStatus();
  }
  return core::UnimplementedError(
      "\"{}\" is set; subword affixes do not exist in byte-level BPE", label);
}

// `key` must be absent or false (e.g. model.byte_fallback).
core::Status RequireAbsentOrFalse(const json& obj, std::string_view key,
                                  std::string_view label) {
  const auto it = obj.find(key);
  if (it == obj.end()) {
    return core::OkStatus();
  }
  if (!it->is_boolean()) {
    return WrongType(label, *it, "a boolean");
  }
  if (it->get<bool>()) {
    return core::UnimplementedError("\"{}\": true is not supported", label);
  }
  return core::OkStatus();
}

// The `type` tag every pipeline component object carries.
core::StatusOr<std::string> ComponentType(const json& node,
                                          std::string_view label) {
  if (!node.is_object()) {
    return WrongType(label, node, "an object");
  }
  const auto it = node.find("type");
  if (it == node.end()) {
    return core::InvalidArgumentError("missing required field \"{}.type\"",
                                      label);
  }
  return StringValue(*it, label);
}

// --- model section (vocab, merges, flags) ----------------------------------

struct ModelData {
  StringToIdMap token_to_id;
  std::vector<std::string> id_to_token;  // dense, ids 0..n-1
  std::unordered_map<std::string, std::int32_t> merge_ranks;
  bool ignore_merges = false;
};

core::Status ParseModelFlags(const json& model, ModelData& data) {
  RETURN_IF_ERROR(RequireAbsentOrNull(model, "dropout", "model.dropout",
                                      "BPE dropout is not supported"));
  RETURN_IF_ERROR(RequireAbsentOrNull(
      model, "unk_token", "model.unk_token",
      "byte-level BPE has no unknown-token path (design §6.1)"));
  RETURN_IF_ERROR(RequireAbsentOrEmpty(model, "continuing_subword_prefix",
                                       "model.continuing_subword_prefix"));
  RETURN_IF_ERROR(RequireAbsentOrEmpty(model, "end_of_word_suffix",
                                       "model.end_of_word_suffix"));
  RETURN_IF_ERROR(RequireAbsentOrFalse(model, "fuse_unk", "model.fuse_unk"));
  RETURN_IF_ERROR(
      RequireAbsentOrFalse(model, "byte_fallback", "model.byte_fallback"));
  RETURN_IF_ERROR(OptionalBool(model, "ignore_merges", "model.ignore_merges",
                               data.ignore_merges));
  return core::OkStatus();
}

core::Status ParseVocab(const json& model, ModelData& data) {
  const auto vocab_it = model.find("vocab");
  if (vocab_it == model.end()) {
    return MissingField("model.vocab");
  }
  if (!vocab_it->is_object()) {
    return WrongType("model.vocab", *vocab_it, "an object");
  }
  const auto vocab_count = static_cast<std::int64_t>(vocab_it->size());
  if (vocab_count > std::numeric_limits<std::int32_t>::max()) {
    return core::InvalidArgumentError(
        "model.vocab has {} entries; ids must fit in int32", vocab_count);
  }
  // n entries whose ids all land in [0, n) with no duplicate cover the
  // range exactly — that contiguity is what makes the dense id-indexed
  // tables (and the decode vector built from them) valid.
  data.id_to_token.assign(static_cast<std::size_t>(vocab_count), {});
  std::vector<bool> seen(static_cast<std::size_t>(vocab_count), false);
  data.token_to_id.reserve(static_cast<std::size_t>(vocab_count));
  for (const auto& [token, id_value] : vocab_it->items()) {
    ASSIGN_OR_RETURN(const std::int64_t id, IntValue(id_value, "model.vocab"));
    if (id < 0 || id >= vocab_count) {
      return core::InvalidArgumentError(
          "model.vocab id {} (token \"{}\") is outside the contiguous range "
          "[0, {})",
          id, token, vocab_count);
    }
    const auto index = static_cast<std::size_t>(id);
    if (seen[index]) {
      return core::InvalidArgumentError(
          "model.vocab id {} is assigned to more than one token", id);
    }
    seen[index] = true;
    data.id_to_token[index] = token;
    data.token_to_id.emplace(token, static_cast<std::int32_t>(id));
  }
  return core::OkStatus();
}

core::Status ParseMerges(const json& model, ModelData& data) {
  const auto merges_it = model.find("merges");
  if (merges_it == model.end()) {
    return MissingField("model.merges");
  }
  if (!merges_it->is_array()) {
    return WrongType("model.merges", *merges_it, "an array");
  }
  data.merge_ranks.reserve(merges_it->size());
  std::size_t index = 0;
  for (const auto& entry : *merges_it) {
    // Either serialization: "left right" (tokenizers < 0.20 — unambiguous
    // because the byte-level alphabet contains no space) or
    // ["left", "right"].
    std::string left;
    std::string right;
    if (entry.is_string()) {
      const auto text = entry.get<std::string_view>();
      const auto space = text.find(' ');
      if (space == std::string_view::npos || space == 0 ||
          space + 1 == text.size() ||
          text.find(' ', space + 1) != std::string_view::npos) {
        return core::InvalidArgumentError(
            "model.merges[{}] (\"{}\") must be \"left right\" with exactly "
            "one space",
            index, text);
      }
      left = text.substr(0, space);
      right = text.substr(space + 1);
    } else if (entry.is_array() && entry.size() == 2) {
      ASSIGN_OR_RETURN(left, StringValue(entry[0], "model.merges pair"));
      ASSIGN_OR_RETURN(right, StringValue(entry[1], "model.merges pair"));
    } else {
      return WrongType(fmt::format("model.merges[{}]", index), entry,
                       "a \"left right\" string or a two-string array");
    }
    for (const auto* part : {&left, &right}) {
      if (!data.token_to_id.contains(*part)) {
        return core::InvalidArgumentError(
            "model.merges[{}] references \"{}\", which is not in the vocab",
            index, *part);
      }
    }
    if (!data.token_to_id.contains(left + right)) {
      return core::InvalidArgumentError(
          "model.merges[{}] product \"{}\" is not in the vocab", index,
          left + right);
    }
    std::string rank_key = left;
    rank_key += ' ';
    rank_key += right;
    const bool inserted =
        data.merge_ranks
            .emplace(std::move(rank_key), static_cast<std::int32_t>(index))
            .second;
    if (!inserted) {
      return core::InvalidArgumentError(
          "model.merges[{}] duplicates an earlier merge", index);
    }
    ++index;
  }
  return core::OkStatus();
}

core::StatusOr<ModelData> ParseModel(const json& root) {
  const auto model_it = root.find("model");
  if (model_it == root.end() || model_it->is_null()) {
    return MissingField("model");
  }
  if (!model_it->is_object()) {
    return WrongType("model", *model_it, "an object");
  }
  ASSIGN_OR_RETURN(const std::string type, ComponentType(*model_it, "model"));
  if (type != "BPE") {
    return core::UnimplementedError(
        "tokenizer model type \"{}\" is not supported; only byte-level "
        "\"BPE\" is (sentencepiece/unigram tokenizers are out of scope, "
        "design §6.1)",
        type);
  }
  ModelData data;
  RETURN_IF_ERROR(ParseModelFlags(*model_it, data));
  RETURN_IF_ERROR(ParseVocab(*model_it, data));
  RETURN_IF_ERROR(ParseMerges(*model_it, data));
  return data;
}

// --- added_tokens ----------------------------------------------------------

core::StatusOr<AddedToken> ParseAddedToken(const json& entry,
                                           std::size_t index) {
  if (!entry.is_object()) {
    return WrongType(fmt::format("added_tokens[{}]", index), entry,
                     "an object");
  }
  AddedToken token;
  const auto id_it = entry.find("id");
  if (id_it == entry.end()) {
    return core::InvalidArgumentError(
        "missing required field \"added_tokens[{}].id\"", index);
  }
  ASSIGN_OR_RETURN(const std::int64_t id,
                   IntValue(*id_it, fmt::format("added_tokens[{}].id", index)));
  if (id < 0 || id > std::numeric_limits<std::int32_t>::max()) {
    return core::InvalidArgumentError("added_tokens[{}].id {} is out of range",
                                      index, id);
  }
  token.id = static_cast<std::int32_t>(id);
  const auto content_it = entry.find("content");
  if (content_it == entry.end()) {
    return core::InvalidArgumentError(
        "missing required field \"added_tokens[{}].content\"", index);
  }
  ASSIGN_OR_RETURN(
      token.content,
      StringValue(*content_it, fmt::format("added_tokens[{}].content", index)));
  if (token.content.empty()) {
    return core::InvalidArgumentError("added_tokens[{}].content is empty",
                                      index);
  }
  const auto label = fmt::format("added_tokens[{}]", index);
  RETURN_IF_ERROR(OptionalBool(entry, "special", label, token.special));
  RETURN_IF_ERROR(OptionalBool(entry, "lstrip", label, token.lstrip));
  RETURN_IF_ERROR(OptionalBool(entry, "rstrip", label, token.rstrip));
  RETURN_IF_ERROR(OptionalBool(entry, "normalized", label, token.normalized));
  RETURN_IF_ERROR(OptionalBool(entry, "single_word", label, token.single_word));
  if (token.single_word) {
    return core::UnimplementedError(
        "added token \"{}\" sets single_word: true, which is not supported "
        "(design §6.2, §9)",
        token.content);
  }
  return token;
}

core::StatusOr<std::vector<AddedToken>> ParseAddedTokens(const json& root) {
  std::vector<AddedToken> tokens;
  const auto it = root.find("added_tokens");
  if (it == root.end() || it->is_null()) {
    return tokens;
  }
  if (!it->is_array()) {
    return WrongType("added_tokens", *it, "an array");
  }
  tokens.reserve(it->size());
  std::size_t index = 0;
  for (const auto& entry : *it) {
    ASSIGN_OR_RETURN(AddedToken token, ParseAddedToken(entry, index));
    tokens.push_back(std::move(token));
    ++index;
  }
  std::ranges::sort(tokens, {}, &AddedToken::id);
  std::unordered_set<std::string_view> contents;
  contents.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0 && tokens[i].id == tokens[i - 1].id) {
      return core::InvalidArgumentError(
          R"(added token id {} is assigned to both "{}" and "{}")",
          tokens[i].id, tokens[i - 1].content, tokens[i].content);
    }
    if (!contents.insert(tokens[i].content).second) {
      return core::InvalidArgumentError(
          "added token content \"{}\" appears more than once",
          tokens[i].content);
    }
  }
  return tokens;
}

// --- normalizer ------------------------------------------------------------

core::StatusOr<bool> ParseNormalizerNode(const json& node) {
  ASSIGN_OR_RETURN(const std::string type, ComponentType(node, "normalizer"));
  if (type == "NFC") {
    return true;
  }
  if (type == "Sequence") {
    const auto list_it = node.find("normalizers");
    if (list_it == node.end() || !list_it->is_array()) {
      return MissingField("normalizer.normalizers");
    }
    bool nfc = false;
    for (const auto& child : *list_it) {
      ASSIGN_OR_RETURN(const bool child_nfc, ParseNormalizerNode(child));
      nfc = nfc || child_nfc;
    }
    return nfc;
  }
  return core::UnimplementedError(
      "normalizer type \"{}\" is not supported; supported: NFC, Sequence "
      "(design §6.2)",
      type);
}

// false = no normalization (Llama 3), true = NFC (Qwen 2+).
core::StatusOr<bool> ParseNormalizer(const json& root) {
  const auto it = root.find("normalizer");
  if (it == root.end() || it->is_null()) {
    return false;
  }
  return ParseNormalizerNode(*it);
}

// --- pre_tokenizer / post_processor / decoder ------------------------------

// Flattens Sequence nodes (whose child list is named `list_key`) into their
// leaf components.
core::Status FlattenComponents(const json& node, std::string_view label,
                               std::string_view list_key,
                               std::vector<const json*>& parts) {
  ASSIGN_OR_RETURN(const std::string type, ComponentType(node, label));
  if (type != "Sequence") {
    parts.push_back(&node);
    return core::OkStatus();
  }
  const auto list_it = node.find(list_key);
  if (list_it == node.end() || !list_it->is_array()) {
    return core::InvalidArgumentError("missing required field \"{}.{}\"", label,
                                      list_key);
  }
  for (const auto& child : *list_it) {
    RETURN_IF_ERROR(FlattenComponents(child, label, list_key, parts));
  }
  return core::OkStatus();
}

core::StatusOr<std::string> ParseSplit(const json& node) {
  const auto pattern_it = node.find("pattern");
  if (pattern_it == node.end() || !pattern_it->is_object()) {
    return MissingField("pre_tokenizer Split.pattern");
  }
  const auto regex_it = pattern_it->find("Regex");
  if (regex_it == pattern_it->end()) {
    return core::UnimplementedError(
        "pre_tokenizer Split.pattern must be a Regex (String patterns are "
        "not supported)");
  }
  ASSIGN_OR_RETURN(std::string pattern,
                   StringValue(*regex_it, "pre_tokenizer Split.pattern.Regex"));
  const auto behavior_it = node.find("behavior");
  if (behavior_it == node.end()) {
    return MissingField("pre_tokenizer Split.behavior");
  }
  ASSIGN_OR_RETURN(const std::string behavior,
                   StringValue(*behavior_it, "pre_tokenizer Split.behavior"));
  if (behavior != "Isolated") {
    return core::UnimplementedError(
        "pre_tokenizer Split.behavior \"{}\" is not supported; supported: "
        "Isolated",
        behavior);
  }
  bool invert = false;
  RETURN_IF_ERROR(
      OptionalBool(node, "invert", "pre_tokenizer Split.invert", invert));
  if (invert) {
    return core::UnimplementedError(
        "pre_tokenizer Split.invert: true is not supported");
  }
  return pattern;
}

core::Status ParseByteLevelPre(const json& node) {
  // HF defaults both flags to true; the whitelist requires false (§6.2 —
  // the prefix space and the embedded GPT-2 regex would change every
  // pre-token), so absent is rejected too.
  bool add_prefix_space = true;
  RETURN_IF_ERROR(OptionalBool(node, "add_prefix_space",
                               "pre_tokenizer ByteLevel.add_prefix_space",
                               add_prefix_space));
  if (add_prefix_space) {
    return core::UnimplementedError(
        "pre_tokenizer ByteLevel.add_prefix_space: true is not supported");
  }
  bool use_regex = true;
  RETURN_IF_ERROR(OptionalBool(node, "use_regex",
                               "pre_tokenizer ByteLevel.use_regex", use_regex));
  if (use_regex) {
    return core::UnimplementedError(
        "pre_tokenizer ByteLevel.use_regex: true is not supported");
  }
  return core::OkStatus();  // trim_offsets only affects offsets; ignored
}

// Returns the Split regex pattern; from_json then selects the matcher
// against the known pattern family (design §6.4), so an off-whitelist
// pattern fails at load, not at first encode.
core::StatusOr<std::string> ParsePreTokenizer(const json& root) {
  const auto it = root.find("pre_tokenizer");
  if (it == root.end() || it->is_null()) {
    return core::UnimplementedError(
        "tokenizer has no pre_tokenizer; only Split + ByteLevel "
        "pre-tokenization is supported (design §6.2)");
  }
  std::vector<const json*> parts;
  RETURN_IF_ERROR(
      FlattenComponents(*it, "pre_tokenizer", "pretokenizers", parts));
  const auto part_type = [&parts](std::size_t i) -> std::string {
    const auto type_it = parts[i]->find("type");
    return type_it != parts[i]->end() && type_it->is_string()
               ? type_it->get<std::string>()
               : std::string("?");
  };
  if (parts.size() != 2 || part_type(0) != "Split" ||
      part_type(1) != "ByteLevel") {
    std::string found;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      found += (i == 0 ? "" : ", ") + part_type(i);
    }
    return core::UnimplementedError(
        "pre_tokenizer must be a Split followed by a ByteLevel; found [{}] "
        "(design §6.2)",
        found);
  }
  ASSIGN_OR_RETURN(std::string pattern, ParseSplit(*parts[0]));
  RETURN_IF_ERROR(ParseByteLevelPre(*parts[1]));
  return pattern;
}

// --- post_processor --------------------------------------------------------

struct TemplateInserts {
  std::vector<std::int32_t> prefix;
  std::vector<std::int32_t> suffix;
};

// One SpecialToken insertion: resolves its name through the processor's
// special_tokens map to a single id.
core::Status ResolveSpecialToken(const json& value, const json& tp,
                                 std::int64_t vocab_size,
                                 std::vector<std::int32_t>& out) {
  if (!value.is_object()) {
    return WrongType("TemplateProcessing SpecialToken", value, "an object");
  }
  const auto name_it = value.find("id");
  if (name_it == value.end()) {
    return MissingField("TemplateProcessing SpecialToken.id");
  }
  ASSIGN_OR_RETURN(const std::string name,
                   StringValue(*name_it, "TemplateProcessing SpecialToken.id"));
  const auto map_it = tp.find("special_tokens");
  if (map_it == tp.end() || !map_it->is_object()) {
    return core::InvalidArgumentError(
        "TemplateProcessing special token \"{}\" is missing from "
        "special_tokens",
        name);
  }
  const auto entry_it = map_it->find(name);
  if (entry_it == map_it->end()) {
    return core::InvalidArgumentError(
        "TemplateProcessing special token \"{}\" is missing from "
        "special_tokens",
        name);
  }
  const auto ids_it =
      entry_it->is_object() ? entry_it->find("ids") : entry_it->end();
  if (ids_it == entry_it->end() || !ids_it->is_array()) {
    return core::InvalidArgumentError(
        R"(TemplateProcessing special_tokens["{}"] has no "ids" array)", name);
  }
  if (ids_it->size() != 1) {
    return core::UnimplementedError(
        "TemplateProcessing special token \"{}\" expands to {} ids; only "
        "single-id insertions are supported",
        name, ids_it->size());
  }
  ASSIGN_OR_RETURN(
      const std::int64_t id,
      IntValue(ids_it->front(),
               fmt::format("TemplateProcessing special_tokens[\"{}\"].ids[0]",
                           name)));
  if (id < 0 || id >= vocab_size) {
    return core::InvalidArgumentError(
        "TemplateProcessing special token \"{}\" id {} is outside the vocab "
        "(size {})",
        name, id, vocab_size);
  }
  out.push_back(static_cast<std::int32_t>(id));
  return core::OkStatus();
}

// The `single` template: SpecialToken insertions around exactly one
// Sequence $A. The `pair` template is deliberately ignored — inference
// never encodes sequence pairs (§6.2).
core::StatusOr<TemplateInserts> ParseTemplate(const json& tp,
                                              std::int64_t vocab_size) {
  const auto single_it = tp.find("single");
  if (single_it == tp.end() || !single_it->is_array()) {
    return MissingField("TemplateProcessing.single");
  }
  TemplateInserts inserts;
  bool seen_sequence = false;
  for (const auto& entry : *single_it) {
    if (!entry.is_object() || entry.size() != 1) {
      return WrongType("TemplateProcessing.single entry", entry,
                       "a single-key object");
    }
    const auto item = entry.begin();  // object iterator: key() + value()
    const std::string& kind = item.key();
    const json& value = item.value();
    if (kind == "Sequence") {
      if (seen_sequence) {
        return core::InvalidArgumentError(
            "TemplateProcessing.single must contain exactly one Sequence");
      }
      const auto id_it = value.is_object() ? value.find("id") : value.end();
      if (id_it == value.end() || !id_it->is_string() ||
          id_it->get<std::string_view>() != "A") {
        return core::InvalidArgumentError(
            "TemplateProcessing.single Sequence must reference \"A\"");
      }
      seen_sequence = true;
    } else if (kind == "SpecialToken") {
      RETURN_IF_ERROR(
          ResolveSpecialToken(value, tp, vocab_size,
                              seen_sequence ? inserts.suffix : inserts.prefix));
    } else {
      return core::UnimplementedError(
          "TemplateProcessing.single entry type \"{}\" is not supported; "
          "supported: SpecialToken, Sequence",
          kind);
    }
  }
  if (!seen_sequence) {
    return core::InvalidArgumentError(
        "TemplateProcessing.single must contain exactly one Sequence");
  }
  return inserts;
}

core::StatusOr<TemplateInserts> ParsePostProcessor(const json& root,
                                                   std::int64_t vocab_size) {
  TemplateInserts inserts;
  const auto it = root.find("post_processor");
  if (it == root.end() || it->is_null()) {
    return inserts;
  }
  std::vector<const json*> parts;
  RETURN_IF_ERROR(
      FlattenComponents(*it, "post_processor", "processors", parts));
  const json* template_node = nullptr;
  for (const auto* part : parts) {
    ASSIGN_OR_RETURN(const std::string type,
                     ComponentType(*part, "post_processor"));
    if (type == "ByteLevel") {
      // A no-op for id sequences — it only rewrites offsets, which we
      // never produce (§6.2, amended by M4-T08).
      continue;
    }
    if (type != "TemplateProcessing") {
      return core::UnimplementedError(
          "post_processor type \"{}\" is not supported; supported: "
          "ByteLevel, TemplateProcessing, Sequence (design §6.2)",
          type);
    }
    if (template_node != nullptr) {
      return core::UnimplementedError(
          "multiple TemplateProcessing post-processors are not supported");
    }
    template_node = part;
  }
  if (template_node == nullptr) {
    return inserts;
  }
  return ParseTemplate(*template_node, vocab_size);
}

// --- decoder ---------------------------------------------------------------

core::Status ParseDecoder(const json& root) {
  const auto it = root.find("decoder");
  if (it == root.end() || it->is_null()) {
    return core::OkStatus();  // decode is the byte-level unmap regardless
  }
  std::vector<const json*> parts;
  RETURN_IF_ERROR(FlattenComponents(*it, "decoder", "decoders", parts));
  for (const auto* part : parts) {
    ASSIGN_OR_RETURN(const std::string type, ComponentType(*part, "decoder"));
    if (type != "ByteLevel") {
      return core::UnimplementedError(
          "decoder type \"{}\" is not supported; supported: ByteLevel "
          "(design §6.2)",
          type);
    }
  }
  return core::OkStatus();
}

// `truncation` / `padding` must be unconfigured — either would change the
// id sequence, and inference always encodes the full text.
core::Status RequireUnconfigured(const json& root, std::string_view key) {
  const auto it = root.find(key);
  if (it == root.end() || it->is_null()) {
    return core::OkStatus();
  }
  return core::UnimplementedError("\"{}\" is configured; not supported", key);
}

// --- encode pipeline (§6.4) -------------------------------------------------

// One piece of an added-token split: either a text span between matches
// (id < 0) or a matched added token's id.
struct AddedSplitPiece {
  std::string_view text;
  std::int32_t id = -1;
};

// Walks back from `from` over whitespace codepoints, not before `floor`
// (lstrip: the whitespace joins the added token's span and is dropped).
std::size_t StripWhitespaceBackward(std::string_view text, std::size_t from,
                                    std::size_t floor) {
  std::size_t begin = from;
  while (begin > floor) {
    std::size_t lead = begin - 1;
    while (lead > floor &&
           (static_cast<unsigned char>(text[lead]) & 0xC0) == 0x80) {
      --lead;
    }
    std::size_t next = lead;
    const auto codepoint = decode_utf8(text, next);
    if (!codepoint.has_value() || next != begin || !is_whitespace(*codepoint)) {
      break;
    }
    begin = lead;
  }
  return begin;
}

// Walks forward from `from` over whitespace codepoints (rstrip).
std::size_t StripWhitespaceForward(std::string_view text, std::size_t from) {
  std::size_t end = from;
  while (end < text.size()) {
    std::size_t next = end;
    const auto codepoint = decode_utf8(text, next);
    if (!codepoint.has_value() || !is_whitespace(*codepoint)) {
      break;
    }
    end = next;
  }
  return end;
}

// §6.4 step 1: longest-match scan for added-token contents whose
// `normalized` flag equals `match_normalized` (false → raw text, before
// normalization; true → after). Matching is unconditional — embedded
// specials always become their ids; add_special_tokens only controls the
// template insertions (the `special_mid_text` vectors pin this).
std::vector<AddedSplitPiece> SplitOnAddedTokens(
    std::string_view text, const std::vector<AddedToken>& tokens,
    const std::unordered_map<char, std::vector<std::size_t>>& buckets,
    bool match_normalized) {
  std::vector<AddedSplitPiece> pieces;
  std::size_t segment_start = 0;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto bucket = buckets.find(text[pos]);
    if (bucket == buckets.end()) {
      ++pos;
      continue;
    }
    const AddedToken* match = nullptr;
    for (const std::size_t index : bucket->second) {  // longest first
      const AddedToken& token = tokens[index];
      if (token.normalized == match_normalized &&
          text.substr(pos).starts_with(token.content)) {
        match = &token;
        break;
      }
    }
    if (match == nullptr) {
      ++pos;
      continue;
    }
    std::size_t begin = pos;
    std::size_t end = pos + match->content.size();
    if (match->lstrip) {
      begin = StripWhitespaceBackward(text, begin, segment_start);
    }
    if (match->rstrip) {
      end = StripWhitespaceForward(text, end);
    }
    if (begin > segment_start) {
      pieces.push_back(
          {.text = text.substr(segment_start, begin - segment_start)});
    }
    pieces.push_back({.text = {}, .id = match->id});
    segment_start = pos = end;
  }
  if (segment_start < text.size()) {
    pieces.push_back({.text = text.substr(segment_start)});
  }
  return pieces;
}

}  // namespace

core::StatusOr<Tokenizer> Tokenizer::from_json(std::string_view json_text) {
  const json root = json::parse(json_text.begin(), json_text.end(),
                                /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded()) {
    return core::InvalidArgumentError(
        "tokenizer file is not valid JSON — expected an HF tokenizer.json "
        "(a sentencepiece .model file is not supported, design §6.1)");
  }
  if (!root.is_object()) {
    return core::InvalidArgumentError(
        "tokenizer.json must be a JSON object, got {}", root.type_name());
  }
  RETURN_IF_ERROR(RequireUnconfigured(root, "truncation"));
  RETURN_IF_ERROR(RequireUnconfigured(root, "padding"));

  ASSIGN_OR_RETURN(ModelData model, ParseModel(root));
  ASSIGN_OR_RETURN(std::vector<AddedToken> added, ParseAddedTokens(root));

  Tokenizer tokenizer;
  tokenizer.ignore_merges_ = model.ignore_merges;
  tokenizer.token_to_id_ = std::move(model.token_to_id);
  tokenizer.id_to_token_ = std::move(model.id_to_token);
  tokenizer.merge_ranks_ = std::move(model.merge_ranks);

  // The decode side of the ticket: raw bytes per id, base vocab unmapped
  // through the byte-level alphabet.
  const std::size_t base_size = tokenizer.id_to_token_.size();
  tokenizer.id_to_bytes_.reserve(base_size + added.size());
  for (std::size_t id = 0; id < base_size; ++id) {
    auto bytes = unmap_alphabet_to_bytes(tokenizer.id_to_token_[id]);
    if (!bytes.ok()) {
      return core::InvalidArgumentError(
          "model.vocab token \"{}\" (id {}) is not a byte-level alphabet "
          "string: {}",
          tokenizer.id_to_token_[id], id, bytes.status().message());
    }
    tokenizer.id_to_bytes_.push_back(*std::move(bytes));
  }

  // Added tokens extend the id space contiguously (or annotate a base id
  // they textually match — HF added tokens may shadow vocab entries). An
  // added token's id decodes to its content verbatim, and its content wins
  // token→id lookups (added tokens are matched before BPE, §6.4).
  tokenizer.added_index_.reserve(added.size());
  for (std::size_t i = 0; i < added.size(); ++i) {
    const AddedToken& token = added[i];
    const auto id = static_cast<std::size_t>(token.id);
    if (id < base_size) {
      if (tokenizer.id_to_token_[id] != token.content) {
        return core::InvalidArgumentError(
            R"(added token "{}" (id {}) collides with vocab token "{}")",
            token.content, token.id, tokenizer.id_to_token_[id]);
      }
      tokenizer.id_to_bytes_[id] = token.content;
    } else {
      if (id != tokenizer.id_to_token_.size()) {
        return core::InvalidArgumentError(
            "added token \"{}\" (id {}) leaves a gap in the id space (next "
            "id is {})",
            token.content, token.id, tokenizer.id_to_token_.size());
      }
      tokenizer.id_to_token_.push_back(token.content);
      tokenizer.id_to_bytes_.push_back(token.content);
    }
    const auto [existing, inserted] =
        tokenizer.token_to_id_.emplace(token.content, token.id);
    if (!inserted && existing->second != token.id) {
      return core::InvalidArgumentError(
          "added token \"{}\" (id {}) duplicates vocab token with id {}",
          token.content, token.id, existing->second);
    }
    tokenizer.added_index_.emplace(token.id, i);
  }
  tokenizer.added_tokens_ = std::move(added);

  ASSIGN_OR_RETURN(tokenizer.normalize_nfc_, ParseNormalizer(root));
  ASSIGN_OR_RETURN(tokenizer.split_pattern_, ParsePreTokenizer(root));
  ASSIGN_OR_RETURN(tokenizer.split_spec_,
                   select_split_pattern(tokenizer.split_pattern_));
  // Encode's added-token match index: first content byte → token indices,
  // longest content first (distinct contents of equal length cannot both
  // prefix-match at one position, so their relative order is irrelevant).
  for (std::size_t i = 0; i < tokenizer.added_tokens_.size(); ++i) {
    tokenizer.added_buckets_[tokenizer.added_tokens_[i].content.front()]
        .push_back(i);
  }
  for (auto& [byte, indices] : tokenizer.added_buckets_) {
    std::ranges::sort(indices, std::ranges::greater{}, [&](std::size_t i) {
      return tokenizer.added_tokens_[i].content.size();
    });
  }
  ASSIGN_OR_RETURN(
      TemplateInserts inserts,
      ParsePostProcessor(
          root, static_cast<std::int64_t>(tokenizer.id_to_token_.size())));
  tokenizer.template_prefix_ = std::move(inserts.prefix);
  tokenizer.template_suffix_ = std::move(inserts.suffix);
  RETURN_IF_ERROR(ParseDecoder(root));
  return tokenizer;
}

core::StatusOr<Tokenizer> Tokenizer::from_file(
    const std::filesystem::path& tokenizer_json) {
  const std::ifstream file(tokenizer_json, std::ios::binary);
  if (!file) {
    return core::NotFoundError("cannot read tokenizer file {}",
                               tokenizer_json.string());
  }
  std::ostringstream contents;
  contents << file.rdbuf();
  if (file.bad()) {
    return core::NotFoundError("cannot read tokenizer file {}",
                               tokenizer_json.string());
  }

  auto tokenizer = from_json(contents.view());
  if (!tokenizer.ok()) {
    return core::Status(tokenizer.status().code(),
                        fmt::format("{}: {}", tokenizer_json.string(),
                                    tokenizer.status().message()));
  }
  return tokenizer;
}

// §6.4 steps 4–5 for one pre-token: map its raw bytes into the merge
// alphabet, run the merge loop (short-circuited by ignore_merges on a
// whole-word vocab hit), and append the resulting ids.
core::Status Tokenizer::append_pretoken_ids(
    std::string_view pretoken_bytes, std::vector<std::int32_t>& ids) const {
  const std::string word = map_bytes_to_alphabet(pretoken_bytes);
  if (ignore_merges_) {
    const auto whole = token_to_id_.find(word);
    if (whole != token_to_id_.end()) {
      ids.push_back(whole->second);
      return core::OkStatus();
    }
  }
  for (const std::string_view symbol : bpe_split(word, merge_ranks_)) {
    const auto it = token_to_id_.find(symbol);
    if (it == token_to_id_.end()) {
      return core::InvalidArgumentError(
          "BPE symbol \"{}\" is not in the vocab — the tokenizer is not a "
          "complete byte-level BPE model (every single-byte token must be "
          "present)",
          symbol);
    }
    ids.push_back(it->second);
  }
  return core::OkStatus();
}

// §6.4 steps 2–5 for one added-token-free segment of valid UTF-8:
// normalize, match normalized:true added tokens, pre-tokenize, BPE.
core::Status Tokenizer::encode_text_segment(
    std::string_view segment, std::vector<std::int32_t>& ids) const {
  std::string normalized_storage;
  std::string_view normalized = segment;
  if (normalize_nfc_) {
    normalized_storage = nfc_normalize(segment);
    normalized = normalized_storage;
  }
  for (const AddedSplitPiece& piece :
       SplitOnAddedTokens(normalized, added_tokens_, added_buckets_,
                          /*match_normalized=*/true)) {
    if (piece.id >= 0) {
      ids.push_back(piece.id);
      continue;
    }
    for (const std::string_view pretoken :
         pretokenize(piece.text, split_spec_)) {
      RETURN_IF_ERROR(append_pretoken_ids(pretoken, ids));
    }
  }
  return core::OkStatus();
}

core::StatusOr<std::vector<std::int32_t>> Tokenizer::encode(
    std::string_view text, bool add_special_tokens) const {
  std::vector<std::int32_t> ids;
  if (add_special_tokens) {
    ids.insert(ids.end(), template_prefix_.begin(), template_prefix_.end());
  }
  // Step 1: split on added tokens with normalized: false against the raw,
  // un-normalized text (the typical special token).
  for (const AddedSplitPiece& piece :
       SplitOnAddedTokens(text, added_tokens_, added_buckets_,
                          /*match_normalized=*/false)) {
    if (piece.id >= 0) {
      ids.push_back(piece.id);
      continue;
    }
    // Byte-level BPE is total over bytes: each maximal run of bytes that
    // does not decode as UTF-8 becomes its own pre-token and goes through
    // the normal merge loop; the valid spans around it encode normally
    // (engine-defined semantics pinned by the encode_synthetic vectors —
    // HF rejects non-UTF-8 input at its API boundary, §6.4).
    const std::string_view segment = piece.text;
    std::size_t valid_start = 0;
    std::size_t pos = 0;
    while (pos < segment.size()) {
      std::size_t next = pos;
      if (decode_utf8(segment, next).has_value()) {
        pos = next;
        continue;
      }
      if (valid_start < pos) {
        RETURN_IF_ERROR(encode_text_segment(
            segment.substr(valid_start, pos - valid_start), ids));
      }
      const std::size_t run_start = pos;
      ++pos;
      while (pos < segment.size()) {
        std::size_t probe = pos;
        if (decode_utf8(segment, probe).has_value()) {
          break;
        }
        ++pos;
      }
      RETURN_IF_ERROR(
          append_pretoken_ids(segment.substr(run_start, pos - run_start), ids));
      valid_start = pos;
    }
    if (valid_start < segment.size()) {
      RETURN_IF_ERROR(encode_text_segment(segment.substr(valid_start), ids));
    }
  }
  if (add_special_tokens) {
    ids.insert(ids.end(), template_suffix_.begin(), template_suffix_.end());
  }
  return ids;
}

// Stub until M4-T10, whose implementation reads the tokenizer's state.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
core::StatusOr<std::string> Tokenizer::decode(
    std::span<const std::int32_t> /*ids*/, bool /*skip_special_tokens*/) const {
  return core::UnimplementedError(
      "Tokenizer::decode is not implemented yet (M4-T10)");
}

std::int64_t Tokenizer::vocab_size() const {
  return static_cast<std::int64_t>(id_to_token_.size());
}

std::optional<std::int32_t> Tokenizer::token_to_id(
    std::string_view token) const {
  const auto it = token_to_id_.find(token);
  if (it == token_to_id_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string_view> Tokenizer::id_to_token(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
    return std::nullopt;
  }
  return std::string_view(id_to_token_[static_cast<std::size_t>(id)]);
}

std::optional<std::string_view> Tokenizer::token_bytes(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= id_to_bytes_.size()) {
    return std::nullopt;
  }
  return std::string_view(id_to_bytes_[static_cast<std::size_t>(id)]);
}

std::optional<std::int32_t> Tokenizer::bos_id() const {
  if (template_prefix_.empty()) {
    return std::nullopt;
  }
  return template_prefix_.front();
}

std::optional<std::int32_t> Tokenizer::eos_id() const {
  if (template_suffix_.empty()) {
    return std::nullopt;
  }
  return template_suffix_.back();
}

bool Tokenizer::is_special(std::int32_t id) const {
  const AddedToken* token = added_token(id);
  return token != nullptr && token->special;
}

const AddedToken* Tokenizer::added_token(std::int32_t id) const {
  const auto it = added_index_.find(id);
  if (it == added_index_.end()) {
    return nullptr;
  }
  return &added_tokens_[it->second];
}

}  // namespace engine::tokenizer
