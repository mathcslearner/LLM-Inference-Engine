#pragma once

#include "core/status.h"
#include "tokenizer/pretokenize.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Byte-level BPE tokenizer over the HF `tokenizer.json` format (M4-T08/T09;
// design: docs/design/model-loading.md §6). Parsing builds the in-memory
// structures — token↔id maps, merge ranks, the added-token registry,
// per-id raw bytes via the byte-level alphabet (bpe.h) — and encode
// (M4-T09) runs the §6.4 pipeline over them: added-token split →
// normalize → pre-tokenize (pretokenize.h) → byte-level map → merge loop
// → template insertions. decode lands in M4-T10.
//
// The format is a pipeline description (`normalizer` → `pre_tokenizer` →
// `model` → `post_processor`, plus `decoder` and `added_tokens`); we
// implement the closure of what the Llama-3 and Qwen-2+ families use and
// reject everything else by name (§6.2's whitelist) — a non-BPE model
// (sentencepiece/unigram) or an off-whitelist component/flag is
// `Unimplemented` naming what was found, silent partial support being how
// byte-identity dies. Malformed input (bad JSON, wrong types, id gaps or
// duplicates) is `InvalidArgument` naming the offending field. Never a
// CHECK, never an exception (ADR-003).

namespace engine::tokenizer {

// One `added_tokens` entry, exactly as parsed (§6.2: all flags stored;
// encode consumes `special`, `lstrip`/`rstrip`, `normalized` — §6.4
// step 1; `single_word: true` is rejected at parse — stored here only so
// the struct mirrors the format).
struct AddedToken {
  std::int32_t id = 0;
  std::string content;
  bool special = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = false;
  bool single_word = false;
};

// Heterogeneous string lookup so token→id queries take a string_view
// without allocating.
struct StringHash {
  using is_transparent = void;
  [[nodiscard]] std::size_t operator()(std::string_view text) const {
    return std::hash<std::string_view>{}(text);
  }
};
using StringToIdMap =
    std::unordered_map<std::string, std::int32_t, StringHash, std::equal_to<>>;

class Tokenizer {
 public:
  // Parses `tokenizer.json`. Errors: `NotFound` for an unreadable path;
  // otherwise `from_json` semantics with the path prefixed to the message.
  [[nodiscard]] static core::StatusOr<Tokenizer> from_file(
      const std::filesystem::path& tokenizer_json);

  // Parses tokenizer.json text (the testable seam, mirroring
  // `ParseModelConfig`). `Unimplemented` for non-BPE models and
  // off-whitelist components; `InvalidArgument` for malformed input.
  [[nodiscard]] static core::StatusOr<Tokenizer> from_json(
      std::string_view json_text);

  // Encodes `text` per the §6.4 pipeline. Added tokens embedded in the
  // text always match (that is HF's behavior and the fixtures pin it);
  // `add_special_tokens` controls only the post-processor template
  // insertions (Llama 3 prepends <|begin_of_text|>). Total over arbitrary
  // bytes — invalid UTF-8 is not an error (each maximal invalid-byte run
  // forms its own pre-token; the `encode_synthetic` vectors pin this
  // engine-defined semantic, §6.4). `InvalidArgument` only if a final
  // merge symbol is missing from the vocab, which a real byte-level vocab
  // (all 256 byte tokens present) cannot produce.
  [[nodiscard]] core::StatusOr<std::vector<std::int32_t>> encode(
      std::string_view text, bool add_special_tokens) const;

  // decode lands in M4-T10; until then it returns `Unimplemented`.
  [[nodiscard]] core::StatusOr<std::string> decode(
      std::span<const std::int32_t> ids, bool skip_special_tokens) const;

  // Number of distinct ids: base vocab plus added tokens (ids are
  // validated contiguous, 0 .. vocab_size()-1). Note this is the
  // *tokenizer's* id space; config.json's `vocab_size` may be larger
  // (padded embedding tables).
  [[nodiscard]] std::int64_t vocab_size() const;

  // Token string ↔ id. Base-vocab tokens are alphabet-space strings
  // ("Ġworld"); added tokens match their raw content ("<|im_start|>").
  // Returned views are valid while this tokenizer (or a moved-to
  // successor) is alive.
  [[nodiscard]] std::optional<std::int32_t> token_to_id(
      std::string_view token) const;
  [[nodiscard]] std::optional<std::string_view> id_to_token(
      std::int32_t id) const;

  // The raw bytes a token id decodes to: base-vocab tokens unmapped
  // through the byte-level alphabet (id_to_token "Ġworld" → " world"),
  // added tokens their content verbatim. May be an incomplete UTF-8
  // fragment — concatenation, not per-token validity, is the decode
  // contract (§6.5). nullopt for out-of-range ids.
  [[nodiscard]] std::optional<std::string_view> token_bytes(
      std::int32_t id) const;

  // BOS/EOS as declared by the post-processor template's insertions
  // (Llama 3 prepends <|begin_of_text|>; Qwen inserts nothing → nullopt).
  [[nodiscard]] std::optional<std::int32_t> bos_id() const;
  [[nodiscard]] std::optional<std::int32_t> eos_id() const;

  // True iff `id` is an added token with `special: true`. False for
  // ordinary and out-of-range ids.
  [[nodiscard]] bool is_special(std::int32_t id) const;

  // The added-token entry for `id`, or nullptr if `id` is not an added
  // token. Valid while this tokenizer (or a moved-to successor) is alive.
  [[nodiscard]] const AddedToken* added_token(std::int32_t id) const;

 private:
  Tokenizer() = default;

  // §6.4 steps 4–5 for one pre-token (raw bytes): byte-level map, merge
  // loop (ignore_merges short-circuit), symbol→id lookups.
  [[nodiscard]] core::Status append_pretoken_ids(
      std::string_view pretoken_bytes, std::vector<std::int32_t>& ids) const;

  // §6.4 steps 2–5 for one added-token-free segment of valid UTF-8:
  // normalize, normalized:true added-token pass, pre-tokenize, BPE.
  [[nodiscard]] core::Status encode_text_segment(
      std::string_view segment, std::vector<std::int32_t>& ids) const;

  // --- pipeline configuration (consumed by encode) ---
  bool normalize_nfc_ = false;  // §6.2: absent (Llama 3) or NFC (Qwen)
  std::string split_pattern_;   // the Split regex, verbatim (provenance)
  SplitSpec split_spec_;        // matcher parameters selected at parse
  bool ignore_merges_ = false;  // §6.2: vocab hit skips the merge loop
  std::vector<std::int32_t> template_prefix_;  // TemplateProcessing inserts
  std::vector<std::int32_t> template_suffix_;

  // --- vocab / merges / added tokens ---
  StringToIdMap token_to_id_;
  std::vector<std::string> id_to_token_;  // dense; alphabet space / content
  std::vector<std::string> id_to_bytes_;  // dense; raw bytes per id
  std::unordered_map<std::string, std::int32_t> merge_ranks_;  // "left right"
  std::vector<AddedToken> added_tokens_;                       // sorted by id
  std::unordered_map<std::int32_t, std::size_t> added_index_;  // id → index

  // Added-token match index for encode's longest-match scan (§6.4 step 1):
  // first content byte → indices into added_tokens_, longest content
  // first. A few hundred entries at most, so a trie is not worth it.
  std::unordered_map<char, std::vector<std::size_t>> added_buckets_;
};

}  // namespace engine::tokenizer
