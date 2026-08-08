#include "model/weight_map.h"

#include "common/paths.h"
#include "core/status.h"
#include "model/checkpoint.h"
#include "model/config.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// Weight-name mapping (M4-T06; design: docs/design/model-loading.md §4).
// Positive coverage runs the committed fixtures: tiny-llama (a real
// checkpoint through the Checkpoint overload, plus a tied-embeddings variant
// of its config) and qwen2-weight-names.json (the full name/shape inventory
// of a real Qwen2-0.5B checkpoint — biases present, lm_head tied and absent).
// Negative cases perturb a synthetic Llama inventory one defect at a time
// and assert the report — or the status message — names the offending
// weights (the actionability criterion).

namespace {

using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::Architecture;
using engine::model::BuildWeightMap;
using engine::model::CheckNoMissingWeights;
using engine::model::Checkpoint;
using engine::model::LoadModelConfig;
using engine::model::ModelConfig;
using engine::model::WeightInventory;
using engine::model::WeightMap;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// --- helpers ---------------------------------------------------------------

// A small synthetic Llama config for inventory-level tests. q rows = 8,
// kv rows = 4.
[[nodiscard]] ModelConfig LlamaTestConfig() {
  ModelConfig config;
  config.architecture = Architecture::kLlama;
  config.architecture_name = "LlamaForCausalLM";
  config.hidden_size = 8;
  config.intermediate_size = 16;
  config.num_layers = 2;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.vocab_size = 32;
  config.max_position_embeddings = 64;
  return config;
}

// The complete HF-spelling inventory matching LlamaTestConfig() (untied, no
// biases) — spelled out with the checkpoint names so the test cannot share a
// table with the implementation.
[[nodiscard]] WeightInventory LlamaInventory(const ModelConfig& config) {
  const std::int64_t hidden = config.hidden_size;
  const std::int64_t inter = config.intermediate_size;
  const std::int64_t q_rows =
      static_cast<std::int64_t>(config.num_heads) * config.head_dim;
  const std::int64_t kv_rows =
      static_cast<std::int64_t>(config.num_kv_heads) * config.head_dim;

  WeightInventory inventory;
  inventory.emplace("model.embed_tokens.weight",
                    Shape{config.vocab_size, hidden});
  for (int i = 0; i < config.num_layers; ++i) {
    const auto add = [&inventory, i](std::string_view suffix, Shape shape) {
      inventory.emplace(fmt::format("model.layers.{}.{}", i, suffix), shape);
    };
    add("input_layernorm.weight", Shape{hidden});
    add("self_attn.q_proj.weight", Shape{q_rows, hidden});
    add("self_attn.k_proj.weight", Shape{kv_rows, hidden});
    add("self_attn.v_proj.weight", Shape{kv_rows, hidden});
    add("self_attn.o_proj.weight", Shape{hidden, q_rows});
    add("post_attention_layernorm.weight", Shape{hidden});
    add("mlp.gate_proj.weight", Shape{inter, hidden});
    add("mlp.up_proj.weight", Shape{inter, hidden});
    add("mlp.down_proj.weight", Shape{hidden, inter});
  }
  inventory.emplace("model.norm.weight", Shape{hidden});
  inventory.emplace("lm_head.weight", Shape{config.vocab_size, hidden});
  return inventory;
}

// The canonical names LlamaTestConfig()'s mapping must produce.
[[nodiscard]] std::vector<std::string> LlamaCanonicalNames(
    const ModelConfig& config) {
  std::vector<std::string> names = {"embed_tokens.weight"};
  for (int i = 0; i < config.num_layers; ++i) {
    for (const std::string_view suffix :
         {"attn_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
          "attn.v_proj.weight", "attn.o_proj.weight", "mlp_norm.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight",
          "mlp.down_proj.weight"}) {
      names.push_back(fmt::format("layers.{}.{}", i, suffix));
    }
  }
  names.emplace_back("final_norm.weight");
  names.emplace_back("lm_head.weight");
  return names;
}

void ExpectReportEmpty(const WeightMap& map) {
  EXPECT_TRUE(map.report.missing.empty())
      << fmt::format("missing: {}", fmt::join(map.report.missing, ", "));
  EXPECT_TRUE(map.report.unexpected.empty())
      << fmt::format("unexpected: {}", fmt::join(map.report.unexpected, ", "));
  EXPECT_TRUE(map.report.ignored.empty())
      << fmt::format("ignored: {}", fmt::join(map.report.ignored, ", "));
}

// --- synthetic Llama inventory ---------------------------------------------

TEST(WeightMapTest, CompleteLlamaInventoryMapsEveryWeight) {
  const ModelConfig config = LlamaTestConfig();
  const StatusOr<WeightMap> map =
      BuildWeightMap(config, LlamaInventory(config));
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  ExpectReportEmpty(*map);
  const std::vector<std::string> canonical = LlamaCanonicalNames(config);
  EXPECT_EQ(map->canonical_to_source.size(), canonical.size());
  for (const std::string& name : canonical) {
    EXPECT_TRUE(map->canonical_to_source.contains(name)) << name;
  }
  EXPECT_EQ(map->canonical_to_source.at("layers.0.attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight");
  EXPECT_EQ(map->canonical_to_source.at("layers.1.attn_norm.weight"),
            "model.layers.1.input_layernorm.weight");
  EXPECT_EQ(map->canonical_to_source.at("layers.1.mlp_norm.weight"),
            "model.layers.1.post_attention_layernorm.weight");
  EXPECT_EQ(map->canonical_to_source.at("final_norm.weight"),
            "model.norm.weight");
  EXPECT_EQ(map->canonical_to_source.at("lm_head.weight"), "lm_head.weight");
}

TEST(WeightMapTest, MissingWeightIsReportedAndFailsThePolicyCheck) {
  const ModelConfig config = LlamaTestConfig();
  WeightInventory inventory = LlamaInventory(config);
  inventory.erase("model.layers.1.mlp.down_proj.weight");
  const StatusOr<WeightMap> map = BuildWeightMap(config, inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(map->report.missing,
            std::vector<std::string>{"layers.1.mlp.down_proj.weight"});
  EXPECT_FALSE(
      map->canonical_to_source.contains("layers.1.mlp.down_proj.weight"));
  const Status status = CheckNoMissingWeights(map->report);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
  EXPECT_NE(status.message().find("layers.1.mlp.down_proj.weight"),
            std::string::npos)
      << status.ToString();
}

TEST(WeightMapTest, EveryMissingWeightIsListedNotJustTheFirst) {
  const ModelConfig config = LlamaTestConfig();
  WeightInventory inventory = LlamaInventory(config);
  inventory.erase("model.embed_tokens.weight");
  inventory.erase("model.norm.weight");
  const StatusOr<WeightMap> map = BuildWeightMap(config, inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(
      map->report.missing,
      (std::vector<std::string>{"embed_tokens.weight", "final_norm.weight"}));
  const Status status = CheckNoMissingWeights(map->report);
  ASSERT_FALSE(status.ok());
  EXPECT_NE(status.message().find("embed_tokens.weight"), std::string::npos)
      << status.ToString();
  EXPECT_NE(status.message().find("final_norm.weight"), std::string::npos)
      << status.ToString();
}

TEST(WeightMapTest, StrayTensorIsUnexpectedNeverAnError) {
  const ModelConfig config = LlamaTestConfig();
  WeightInventory inventory = LlamaInventory(config);
  inventory.emplace("model.layers.0.self_attn.lora_A.weight", Shape{4, 8});
  const StatusOr<WeightMap> map = BuildWeightMap(config, inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(map->report.unexpected,
            std::vector<std::string>{"model.layers.0.self_attn.lora_A.weight"});
  EXPECT_TRUE(map->report.missing.empty());
  EXPECT_TRUE(CheckNoMissingWeights(map->report).ok());
}

TEST(WeightMapTest, RotaryInvFreqIsIgnoredNotUnexpected) {
  const ModelConfig config = LlamaTestConfig();
  WeightInventory inventory = LlamaInventory(config);
  inventory.emplace("model.layers.1.self_attn.rotary_emb.inv_freq", Shape{2});
  const StatusOr<WeightMap> map = BuildWeightMap(config, inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(
      map->report.ignored,
      std::vector<std::string>{"model.layers.1.self_attn.rotary_emb.inv_freq"});
  EXPECT_TRUE(map->report.unexpected.empty());
}

TEST(WeightMapTest, ShapeMismatchIsRejectedNamingTheWeight) {
  const ModelConfig config = LlamaTestConfig();
  WeightInventory inventory = LlamaInventory(config);
  inventory.at("model.layers.0.self_attn.q_proj.weight") = Shape{8, 9};
  const StatusOr<WeightMap> map = BuildWeightMap(config, inventory);
  ASSERT_FALSE(map.ok());
  // status() returns by value; keep it alive while message() is inspected.
  const Status status = map.status();
  EXPECT_TRUE(IsInvalidArgument(status)) << status.ToString();
  const std::string_view message = status.message();
  EXPECT_NE(message.find("layers.0.attn.q_proj.weight"), std::string::npos)
      << status.ToString();
  EXPECT_NE(message.find("[8, 9]"), std::string::npos) << status.ToString();
  EXPECT_NE(message.find("[8, 8]"), std::string::npos) << status.ToString();
}

TEST(WeightMapTest, LlamaAttentionBiasGatesTheBiasRows) {
  ModelConfig config = LlamaTestConfig();
  config.attention_bias = true;
  // Inventory without biases: all four projections' biases are missing, per
  // layer (HF LlamaAttention biases o_proj too, unlike Qwen2).
  const StatusOr<WeightMap> without =
      BuildWeightMap(config, LlamaInventory(config));
  ASSERT_TRUE(without.ok()) << without.status().ToString();
  EXPECT_EQ(without->report.missing.size(), 8U);
  for (const std::string_view name :
       {"layers.0.attn.q_proj.bias", "layers.0.attn.o_proj.bias",
        "layers.1.attn.k_proj.bias", "layers.1.attn.v_proj.bias"}) {
    EXPECT_NE(std::ranges::find(without->report.missing, name),
              without->report.missing.end())
        << name;
  }

  WeightInventory inventory = LlamaInventory(config);
  for (int i = 0; i < config.num_layers; ++i) {
    inventory.emplace(fmt::format("model.layers.{}.self_attn.q_proj.bias", i),
                      Shape{8});
    inventory.emplace(fmt::format("model.layers.{}.self_attn.k_proj.bias", i),
                      Shape{4});
    inventory.emplace(fmt::format("model.layers.{}.self_attn.v_proj.bias", i),
                      Shape{4});
    inventory.emplace(fmt::format("model.layers.{}.self_attn.o_proj.bias", i),
                      Shape{8});
  }
  const StatusOr<WeightMap> with = BuildWeightMap(config, inventory);
  ASSERT_TRUE(with.ok()) << with.status().ToString();
  ExpectReportEmpty(*with);
  EXPECT_EQ(with->canonical_to_source.at("layers.0.attn.o_proj.bias"),
            "model.layers.0.self_attn.o_proj.bias");
}

TEST(WeightMapTest, BiasTensorWithoutAttentionBiasIsUnexpected) {
  const ModelConfig config = LlamaTestConfig();  // attention_bias = false
  WeightInventory inventory = LlamaInventory(config);
  inventory.emplace("model.layers.0.self_attn.q_proj.bias", Shape{8});
  const StatusOr<WeightMap> map = BuildWeightMap(config, inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(map->report.unexpected,
            std::vector<std::string>{"model.layers.0.self_attn.q_proj.bias"});
}

// --- committed fixtures: tiny-llama ----------------------------------------

TEST(WeightMapTest, TinyLlamaFixtureResolvesEveryWeight) {
  const std::filesystem::path dir =
      engine::testing::FixturesDir() / "models/tiny-llama";
  const StatusOr<ModelConfig> config = LoadModelConfig(dir / "config.json");
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  StatusOr<Checkpoint> checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  const StatusOr<WeightMap> map = BuildWeightMap(*config, *checkpoint);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  ExpectReportEmpty(*map);
  const std::vector<std::string> canonical = LlamaCanonicalNames(*config);
  EXPECT_EQ(map->canonical_to_source.size(), canonical.size());
  for (const std::string& name : canonical) {
    EXPECT_TRUE(map->canonical_to_source.contains(name)) << name;
  }
  // Every checkpoint tensor is claimed by the mapping (21 names, and the
  // report above is empty — nothing unaccounted for on either side).
  EXPECT_EQ(map->canonical_to_source.size(), checkpoint->names().size());
}

TEST(WeightMapTest, TiedEmbeddingsAliasShareOneTensorHandle) {
  const std::filesystem::path dir =
      engine::testing::FixturesDir() / "models/tiny-llama";
  StatusOr<ModelConfig> config = LoadModelConfig(dir / "config.json");
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  ASSERT_FALSE(config->tie_word_embeddings);  // fixture is untied
  config->tie_word_embeddings = true;
  StatusOr<Checkpoint> checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  const StatusOr<WeightMap> map = BuildWeightMap(*config, *checkpoint);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  // Both canonical names resolve to the embedding; the checkpoint's own
  // lm_head.weight is superseded by the alias (design §4: alias wins,
  // matching HF) and reported as ignored, not unexpected.
  EXPECT_EQ(map->canonical_to_source.at("lm_head.weight"),
            "model.embed_tokens.weight");
  EXPECT_EQ(map->canonical_to_source.at("embed_tokens.weight"),
            "model.embed_tokens.weight");
  EXPECT_EQ(map->report.ignored, std::vector<std::string>{"lm_head.weight"});
  EXPECT_TRUE(map->report.missing.empty());
  EXPECT_TRUE(map->report.unexpected.empty());

  const StatusOr<Tensor> embed =
      checkpoint->tensor(map->canonical_to_source.at("embed_tokens.weight"));
  ASSERT_TRUE(embed.ok()) << embed.status().ToString();
  const StatusOr<Tensor> lm_head =
      checkpoint->tensor(map->canonical_to_source.at("lm_head.weight"));
  ASSERT_TRUE(lm_head.ok()) << lm_head.status().ToString();
  EXPECT_EQ(embed->data(), lm_head->data());
}

// --- committed fixtures: qwen2-weight-names.json ----------------------------

struct Qwen2Fixture {
  ModelConfig config;
  WeightInventory inventory;
};

// Parses the committed name/shape inventory of Qwen/Qwen2-0.5B-Instruct into
// a config and a WeightInventory. Uses ASSERT_ internally, so callers take
// the result by out-parameter inside ASSERT_NO_FATAL_FAILURE.
void LoadQwen2Fixture(Qwen2Fixture& out) {
  const std::filesystem::path path =
      engine::testing::FixturesDir() / "models/qwen2-weight-names.json";
  const std::ifstream file(path, std::ios::binary);
  ASSERT_TRUE(file) << path;
  std::ostringstream contents;
  contents << file.rdbuf();
  const nlohmann::json root = nlohmann::json::parse(
      contents.str(), /*cb=*/nullptr, /*allow_exceptions=*/false);
  ASSERT_FALSE(root.is_discarded()) << path;

  const nlohmann::json& config_json = root.at("config");
  ModelConfig& config = out.config;
  config.architecture = Architecture::kQwen2;
  config.architecture_name = "Qwen2ForCausalLM";
  config.hidden_size = config_json.at("hidden_size").get<std::int64_t>();
  config.intermediate_size =
      config_json.at("intermediate_size").get<std::int64_t>();
  config.num_layers = config_json.at("num_hidden_layers").get<int>();
  config.num_heads = config_json.at("num_attention_heads").get<int>();
  config.num_kv_heads = config_json.at("num_key_value_heads").get<int>();
  ASSERT_EQ(config.hidden_size % config.num_heads, 0);
  config.head_dim = static_cast<int>(config.hidden_size / config.num_heads);
  config.vocab_size = config_json.at("vocab_size").get<std::int64_t>();
  config.tie_word_embeddings =
      config_json.at("tie_word_embeddings").get<bool>();
  config.attention_bias = true;            // Qwen2 per-architecture default
  config.max_position_embeddings = 32768;  // unused by mapping

  for (const auto& [name, entry] : root.at("tensors").items()) {
    const auto dims = entry.at("shape").get<std::vector<std::int64_t>>();
    const StatusOr<Shape> shape = Shape::FromDims(dims);
    ASSERT_TRUE(shape.ok()) << name << ": " << shape.status().ToString();
    out.inventory.emplace(name, *shape);
  }
}

TEST(WeightMapTest, Qwen2InventoryMapsBiasesAndTiedLmHead) {
  Qwen2Fixture fixture;
  ASSERT_NO_FATAL_FAILURE(LoadQwen2Fixture(fixture));
  ASSERT_TRUE(fixture.config.tie_word_embeddings);
  ASSERT_FALSE(fixture.inventory.contains("lm_head.weight"));

  const StatusOr<WeightMap> map =
      BuildWeightMap(fixture.config, fixture.inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  ExpectReportEmpty(*map);
  // 24 layers × (9 weights + 3 biases) + embed_tokens + final_norm +
  // lm_head-as-alias. Qwen2 has no o_proj bias — a clean report proves the
  // table doesn't demand one.
  EXPECT_EQ(map->canonical_to_source.size(), (24U * 12U) + 3U);
  EXPECT_EQ(map->canonical_to_source.at("layers.0.attn.q_proj.bias"),
            "model.layers.0.self_attn.q_proj.bias");
  EXPECT_EQ(map->canonical_to_source.at("layers.23.attn.v_proj.bias"),
            "model.layers.23.self_attn.v_proj.bias");
  EXPECT_FALSE(map->canonical_to_source.contains("layers.0.attn.o_proj.bias"));
  EXPECT_EQ(map->canonical_to_source.at("lm_head.weight"),
            "model.embed_tokens.weight");
}

TEST(WeightMapTest, Qwen2MissingBiasIsReported) {
  Qwen2Fixture fixture;
  ASSERT_NO_FATAL_FAILURE(LoadQwen2Fixture(fixture));
  fixture.inventory.erase("model.layers.3.self_attn.v_proj.bias");
  const StatusOr<WeightMap> map =
      BuildWeightMap(fixture.config, fixture.inventory);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  EXPECT_EQ(map->report.missing,
            std::vector<std::string>{"layers.3.attn.v_proj.bias"});
}

}  // namespace
