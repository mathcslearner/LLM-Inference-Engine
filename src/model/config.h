#pragma once

#include "core/status.h"
#include "tensor/dtype.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

// HF config.json → ModelConfig (M4-T03; design: docs/design/model-loading.md
// §3.2).
//
// Parsing is tolerant of unknown fields (HF configs carry many keys the
// engine never consumes) and strict about the ones it does consume: missing
// required fields, wrong JSON types, and out-of-range values are recoverable
// `InvalidArgument` errors naming the offending field — never a CHECK and
// never an exception (ADR-003). The JSON library is an implementation
// detail of the .cpp; this header stays nlohmann-free.

namespace engine::model {

// The architectures the engine executes. `architectures[0]` in config.json
// selects one; anything else is `Unimplemented`. Adding an architecture =
// one registry entry here + one weight-mapping table (M4-T06) — the
// Llama-3/Qwen-2 focus is a floor, not a ceiling (design §3.2).
enum class Architecture : std::uint8_t { kLlama, kQwen2 };

// The config.json `rope_scaling` block, parsed here and interpreted by the
// executor (M5/M7). Unknown `rope_type` values are parsed, not rejected —
// rejection would couple config parsing to execution capability; the
// executor rejects what it can't honor (design §3.2).
struct RopeScaling {
  std::string rope_type;  // "llama3", "linear", … (legacy key: "type")
  float factor = 1.0F;
  float low_freq_factor = 1.0F;   // llama3-type only
  float high_freq_factor = 4.0F;  // llama3-type only
  std::int64_t original_max_position_embeddings = 0;
};

// The engine's view of config.json. Defaults on the members are the values
// used when the checkpoint omits the field, matching HF `transformers`
// semantics — except `attention_bias`, whose default is per-architecture
// (false for Llama, true for Qwen2; an explicit JSON value wins), and
// `architecture`/`max_position_embeddings`, which are always required.
struct ModelConfig {
  Architecture architecture = Architecture::kLlama;
  std::string architecture_name;  // raw HF string, for messages
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  int num_layers = 0;    // HF: num_hidden_layers
  int num_heads = 0;     // HF: num_attention_heads
  int num_kv_heads = 0;  // HF: num_key_value_heads (GQA)
  int head_dim = 0;      // explicit or hidden_size / num_heads
  std::int64_t vocab_size = 0;
  float rope_theta = 10000.0F;
  std::optional<RopeScaling> rope_scaling;
  float rms_norm_eps = 1e-6F;
  std::int64_t max_position_embeddings = 0;
  bool tie_word_embeddings = false;
  bool attention_bias = false;  // per-arch default (design §3.2)
  tensor::DataType torch_dtype = tensor::DataType::kFloat32;
};

// Parses the text of a config.json. Errors: `InvalidArgument` naming the
// malformed/missing field (or the JSON syntax problem); `Unimplemented`
// listing the supported strings for an unknown architecture.
[[nodiscard]] core::StatusOr<ModelConfig> ParseModelConfig(
    std::string_view json_text);

// Reads `path` and parses it. `NotFound` when the file cannot be read;
// parse errors are prefixed with the path for context.
[[nodiscard]] core::StatusOr<ModelConfig> LoadModelConfig(
    const std::filesystem::path& path);

}  // namespace engine::model
