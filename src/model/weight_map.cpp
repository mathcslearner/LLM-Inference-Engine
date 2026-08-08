#include "model/weight_map.h"

#include "core/logging.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::model {
namespace {

// One expected weight, expanded from the per-architecture table (design §4)
// with layer indices and config dimensions substituted. Expected dims live in
// a plain vector rather than a Shape: config values are external input, and
// Shape's literal constructor CHECKs on (theoretical) numel overflow.
struct ExpectedWeight {
  std::string canonical;
  std::string source;              // HF checkpoint name
  std::vector<std::int64_t> dims;  // [out_features, in_features] for linears
};

// The per-layer rows. Bias presence is per-config (`attention_bias`): q/k/v
// biases for both architectures, o_proj bias only for Llama — HF's
// LlamaAttention gives all four projections a bias, Qwen2Attention leaves
// o_proj bias-free.
void AddLayerWeights(const ModelConfig& config, int layer,
                     std::vector<ExpectedWeight>& out) {
  const std::int64_t hidden = config.hidden_size;
  const std::int64_t q_rows =
      static_cast<std::int64_t>(config.num_heads) * config.head_dim;
  const std::int64_t kv_rows =
      static_cast<std::int64_t>(config.num_kv_heads) * config.head_dim;

  const auto add = [&out, layer](std::string_view canonical,
                                 std::string_view source,
                                 std::vector<std::int64_t> dims) {
    out.push_back({.canonical = fmt::format("layers.{}.{}", layer, canonical),
                   .source = fmt::format("model.layers.{}.{}", layer, source),
                   .dims = std::move(dims)});
  };

  add("attn_norm.weight", "input_layernorm.weight", {hidden});
  add("attn.q_proj.weight", "self_attn.q_proj.weight", {q_rows, hidden});
  add("attn.k_proj.weight", "self_attn.k_proj.weight", {kv_rows, hidden});
  add("attn.v_proj.weight", "self_attn.v_proj.weight", {kv_rows, hidden});
  add("attn.o_proj.weight", "self_attn.o_proj.weight", {hidden, q_rows});
  if (config.attention_bias) {
    add("attn.q_proj.bias", "self_attn.q_proj.bias", {q_rows});
    add("attn.k_proj.bias", "self_attn.k_proj.bias", {kv_rows});
    add("attn.v_proj.bias", "self_attn.v_proj.bias", {kv_rows});
    if (config.architecture == Architecture::kLlama) {
      add("attn.o_proj.bias", "self_attn.o_proj.bias", {hidden});
    }
  }
  add("mlp_norm.weight", "post_attention_layernorm.weight", {hidden});
  add("mlp.gate_proj.weight", "mlp.gate_proj.weight",
      {config.intermediate_size, hidden});
  add("mlp.up_proj.weight", "mlp.up_proj.weight",
      {config.intermediate_size, hidden});
  add("mlp.down_proj.weight", "mlp.down_proj.weight",
      {hidden, config.intermediate_size});
}

// Expands the full expected-weight list, in layer order (which is the order
// `report.missing` reads back in).
[[nodiscard]] std::vector<ExpectedWeight> ExpectedWeights(
    const ModelConfig& config) {
  std::vector<ExpectedWeight> expected;
  expected.push_back({.canonical = "embed_tokens.weight",
                      .source = "model.embed_tokens.weight",
                      .dims = {config.vocab_size, config.hidden_size}});
  for (int layer = 0; layer < config.num_layers; ++layer) {
    AddLayerWeights(config, layer, expected);
  }
  expected.push_back({.canonical = "final_norm.weight",
                      .source = "model.norm.weight",
                      .dims = {config.hidden_size}});
  // Tied embeddings: lm_head is an alias of embed_tokens — the same source
  // name, so both canonical names resolve to one Tensor (design §4).
  expected.push_back({.canonical = "lm_head.weight",
                      .source = config.tie_word_embeddings
                                    ? "model.embed_tokens.weight"
                                    : "lm_head.weight",
                      .dims = {config.vocab_size, config.hidden_size}});
  return expected;
}

// Known non-weight artifacts (design §4's ignore list): per-layer RoPE
// frequency tables serialized by older HF exports, and — for tied models —
// a redundantly-serialized lm_head, which the alias supersedes.
[[nodiscard]] bool IsIgnoredName(const ModelConfig& config,
                                 std::string_view name) {
  if (name.starts_with("model.layers.") &&
      name.ends_with(".self_attn.rotary_emb.inv_freq")) {
    return true;
  }
  return config.tie_word_embeddings && name == "lm_head.weight";
}

}  // namespace

core::StatusOr<WeightMap> BuildWeightMap(const ModelConfig& config,
                                         const WeightInventory& inventory) {
  WeightMap map;
  // Views into `inventory`'s keys; outlives every use below.
  std::set<std::string_view> claimed;
  std::vector<ExpectedWeight> expected = ExpectedWeights(config);
  for (ExpectedWeight& weight : expected) {
    const auto it = inventory.find(weight.source);
    if (it == inventory.end()) {
      map.report.missing.push_back(std::move(weight.canonical));
      continue;
    }
    if (!std::ranges::equal(it->second.dims(), weight.dims)) {
      return core::InvalidArgumentError(
          "weight \"{}\" (checkpoint tensor \"{}\") has shape {}, expected "
          "[{}]",
          weight.canonical, weight.source, it->second,
          fmt::join(weight.dims, ", "));
    }
    claimed.insert(it->first);
    map.canonical_to_source.emplace(std::move(weight.canonical),
                                    std::move(weight.source));
  }
  for (const auto& [name, shape] : inventory) {
    if (claimed.contains(name)) {
      continue;
    }
    auto& bucket = IsIgnoredName(config, name) ? map.report.ignored
                                               : map.report.unexpected;
    bucket.push_back(name);
  }
  if (!map.report.unexpected.empty()) {
    LOG_WARN("model", "checkpoint has {} unexpected tensor(s): {}",
             map.report.unexpected.size(),
             fmt::join(map.report.unexpected, ", "));
  }
  if (!map.report.ignored.empty()) {
    LOG_DEBUG("model", "ignoring {} known non-weight tensor(s): {}",
              map.report.ignored.size(), fmt::join(map.report.ignored, ", "));
  }
  return map;
}

core::StatusOr<WeightMap> BuildWeightMap(const ModelConfig& config,
                                         Checkpoint& checkpoint) {
  WeightInventory inventory;
  for (const std::string_view name : checkpoint.names()) {
    ASSIGN_OR_RETURN(const tensor::Tensor tensor, checkpoint.tensor(name));
    inventory.emplace(std::string(name), tensor.shape());
  }
  return BuildWeightMap(config, inventory);
}

core::Status CheckNoMissingWeights(const WeightMapReport& report) {
  if (report.missing.empty()) {
    return core::OkStatus();
  }
  return core::InvalidArgumentError(
      "checkpoint is missing {} required weight(s): {}", report.missing.size(),
      fmt::join(report.missing, ", "));
}

}  // namespace engine::model
