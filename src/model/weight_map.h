#pragma once

#include "core/status.h"
#include "model/checkpoint.h"
#include "model/config.h"
#include "tensor/shape.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

// Weight-name mapping (M4-T06; design: docs/design/model-loading.md §4).
//
// Internal code (M5's modules, M6's optimized backend) speaks the canonical
// namespace exclusively; HF checkpoint spellings appear nowhere outside this
// TU's per-architecture tables. Canonical names are flat strings with the
// layer index inline:
//
//   embed_tokens.weight
//   layers.{i}.attn_norm.weight               (HF: input_layernorm)
//   layers.{i}.attn.{q,k,v,o}_proj.weight     [+ .bias when attention_bias]
//   layers.{i}.mlp_norm.weight                (HF: post_attention_layernorm)
//   layers.{i}.mlp.{gate,up,down}_proj.weight
//   final_norm.weight
//   lm_head.weight
//
// `attn_norm`/`mlp_norm` name the *position* (pre-attention / pre-MLP)
// rather than HF's spellings, which read as if the second norm followed
// attention's output — it precedes the MLP.
//
// Shape validation happens here (design §4): every mapped weight's shape is
// checked against its expectation from config, so M5 never sees a malformed
// registry. Canonical linear weights keep HF's [out_features, in_features]
// storage orientation — that convention is fixed here and consumed by M5.
// Dtype policy is deliberately *not* applied here; it belongs to load_model
// (M4-T07, design §5).

namespace engine::model {

// The load report (design §4). `missing` makes the load fail (via
// CheckNoMissingWeights below); `unexpected` and `ignored` never do — a
// fine-tune with extra adapter tensors should load, loudly.
struct WeightMapReport {
  std::vector<std::string> missing;     // canonical names, in layer order
  std::vector<std::string> unexpected;  // checkpoint names — warned
  std::vector<std::string> ignored;     // checkpoint names — known artifacts
};

// Metadata view of a checkpoint: tensor name → shape. All that mapping and
// shape validation consume — testable without tensor bytes (the
// qwen2-weight-names.json fixture is such an inventory).
using WeightInventory = std::map<std::string, tensor::Shape, std::less<>>;

// The mapping result. With tied embeddings (`config.tie_word_embeddings`),
// "lm_head.weight" and "embed_tokens.weight" map to the *same* source name,
// so resolving both through a Checkpoint yields one shared Tensor (tensor.md
// shallow-copy semantics — the design's tied-alias rule for free); a
// checkpoint that redundantly serializes lm_head.weight anyway has that copy
// `ignored` (the alias wins, matching HF load behavior).
struct WeightMap {
  // canonical name → checkpoint tensor name.
  std::map<std::string, std::string, std::less<>> canonical_to_source;
  WeightMapReport report;
};

// Builds the mapping for `config`'s architecture against a checkpoint
// inventory. Absent required weights are recorded in `report.missing`, not
// an error here — the load-error policy is CheckNoMissingWeights, so one
// call reports *every* missing name (design §3.6). A shape mismatch is
// `InvalidArgument` naming the weight, expected, and actual shapes.
// Unexpected names are logged as one warning line; ignored names at debug.
[[nodiscard]] core::StatusOr<WeightMap> BuildWeightMap(
    const ModelConfig& config, const WeightInventory& inventory);

// Convenience over a real checkpoint: reads every tensor's shape (zero-copy
// views; this maps all shards, and a defective shard's error propagates)
// and delegates to the inventory overload. Non-const Checkpoint& because
// shard mapping is lazy. Note an F8_E4M3 tensor fails materialization
// (`Unimplemented`, reserved dtype) even where the inventory overload would
// merely report it unexpected — in-design, the loader rejects F8 anyway
// (design §5).
[[nodiscard]] core::StatusOr<WeightMap> BuildWeightMap(
    const ModelConfig& config, Checkpoint& checkpoint);

// The load-error policy step (design §4): OK iff `report.missing` is empty,
// else `InvalidArgument` listing every missing canonical name — not just the
// first, so one round-trip fixes a broken checkpoint.
[[nodiscard]] core::Status CheckNoMissingWeights(const WeightMapReport& report);

}  // namespace engine::model
