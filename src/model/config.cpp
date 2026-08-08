#include "model/config.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace engine::model {
namespace {

using nlohmann::json;

// --- field accessors -------------------------------------------------------
//
// External input, so every accessor is exception-free (design §2): presence
// via find(), type via is_*() before any get<>(). `field` is the dotted
// label used in messages ("rope_scaling.factor"), which is what makes the
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

core::StatusOr<float> FloatValue(const json& value, std::string_view field) {
  if (!value.is_number()) {
    return WrongType(field, value, "a number");
  }
  return static_cast<float>(value.get<double>());
}

core::StatusOr<bool> BoolValue(const json& value, std::string_view field) {
  if (!value.is_boolean()) {
    return WrongType(field, value, "a boolean");
  }
  return value.get<bool>();
}

core::StatusOr<std::string> StringValue(const json& value,
                                        std::string_view field) {
  if (!value.is_string()) {
    return WrongType(field, value, "a string");
  }
  return value.get<std::string>();
}

core::StatusOr<std::int64_t> RequiredInt(const json& obj,
                                         std::string_view field) {
  const auto it = obj.find(field);
  if (it == obj.end()) {
    return MissingField(field);
  }
  return IntValue(*it, field);
}

// Optional accessors leave `out` untouched when the field is absent (the
// struct member already carries the default) and error on a present field
// of the wrong type — silently ignoring a malformed value the user wrote
// would be the opposite of unknown-field tolerance.
core::Status OptionalInt(const json& obj, std::string_view field,
                         std::int64_t& out) {
  const auto it = obj.find(field);
  if (it == obj.end()) {
    return core::OkStatus();
  }
  auto value = IntValue(*it, field);
  if (!value.ok()) {
    return value.status();
  }
  out = *value;
  return core::OkStatus();
}

core::Status OptionalFloat(const json& obj, std::string_view field,
                           float& out) {
  const auto it = obj.find(field);
  if (it == obj.end()) {
    return core::OkStatus();
  }
  auto value = FloatValue(*it, field);
  if (!value.ok()) {
    return value.status();
  }
  out = *value;
  return core::OkStatus();
}

core::Status OptionalBool(const json& obj, std::string_view field, bool& out) {
  const auto it = obj.find(field);
  if (it == obj.end()) {
    return core::OkStatus();
  }
  auto value = BoolValue(*it, field);
  if (!value.ok()) {
    return value.status();
  }
  out = *value;
  return core::OkStatus();
}

// Narrows a validated int64 config value into an `int` member (layer/head
// counts). Positivity is checked later for every count in one place; this
// only guards the narrowing itself.
core::StatusOr<int> AsInt(std::int64_t value, std::string_view field) {
  if (value > std::numeric_limits<int>::max() ||
      value < std::numeric_limits<int>::min()) {
    return core::InvalidArgumentError("field \"{}\" is out of range: {}", field,
                                      value);
  }
  return static_cast<int>(value);
}

core::Status CheckPositive(std::int64_t value, std::string_view field) {
  if (value <= 0) {
    return core::InvalidArgumentError(
        "field \"{}\" must be strictly positive, got {}", field, value);
  }
  return core::OkStatus();
}

core::Status CheckPositiveFloat(float value, std::string_view field) {
  if (!(value > 0.0F)) {
    return core::InvalidArgumentError(
        "field \"{}\" must be strictly positive, got {}", field, value);
  }
  return core::OkStatus();
}

// --- sub-parsers ------------------------------------------------------------

core::StatusOr<Architecture> ParseArchitecture(const std::string& name) {
  if (name == "LlamaForCausalLM") {
    return Architecture::kLlama;
  }
  if (name == "Qwen2ForCausalLM") {
    return Architecture::kQwen2;
  }
  return core::UnimplementedError(
      "unsupported architecture \"{}\"; supported: LlamaForCausalLM, "
      "Qwen2ForCausalLM",
      name);
}

core::StatusOr<RopeScaling> ParseRopeScaling(const json& obj) {
  if (!obj.is_object()) {
    return WrongType("rope_scaling", obj, "an object or null");
  }
  RopeScaling scaling;
  // "rope_type" is the current HF spelling; configs written before
  // transformers 4.43 say "type" ("linear", "dynamic"). transformers
  // itself accepts both, normalizing to "rope_type" — so do we.
  auto it = obj.find("rope_type");
  if (it == obj.end()) {
    it = obj.find("type");
  }
  if (it == obj.end()) {
    return MissingField("rope_scaling.rope_type");
  }
  auto rope_type = StringValue(*it, "rope_scaling.rope_type");
  if (!rope_type.ok()) {
    return rope_type.status();
  }
  scaling.rope_type = *std::move(rope_type);

  // Numeric subfields keep their struct defaults when absent; the dotted
  // labels keep wrong-type messages actionable ("rope_scaling.factor").
  struct FloatField {
    const char* key;
    const char* label;
    float& out;
  };
  const FloatField float_fields[] = {
      {.key = "factor", .label = "rope_scaling.factor", .out = scaling.factor},
      {.key = "low_freq_factor",
       .label = "rope_scaling.low_freq_factor",
       .out = scaling.low_freq_factor},
      {.key = "high_freq_factor",
       .label = "rope_scaling.high_freq_factor",
       .out = scaling.high_freq_factor},
  };
  for (const auto& field : float_fields) {
    const auto field_it = obj.find(field.key);
    if (field_it == obj.end()) {
      continue;
    }
    auto value = FloatValue(*field_it, field.label);
    if (!value.ok()) {
      return value.status();
    }
    field.out = *value;
  }
  const auto orig_it = obj.find("original_max_position_embeddings");
  if (orig_it != obj.end()) {
    auto value =
        IntValue(*orig_it, "rope_scaling.original_max_position_embeddings");
    if (!value.ok()) {
      return value.status();
    }
    scaling.original_max_position_embeddings = *value;
  }
  return scaling;
}

// --- parse stages -----------------------------------------------------------
//
// ParseModelConfig runs these in order, each filling part of `config`. The
// split mirrors the design §3.2 rule groups (architecture registry →
// required → defaulted → derived → validation).

// Architecture first: the attention_bias default depends on it.
core::Status ParseArchitectureField(const json& root, ModelConfig& config) {
  const auto arch_it = root.find("architectures");
  if (arch_it == root.end()) {
    return MissingField("architectures");
  }
  if (!arch_it->is_array() || arch_it->empty()) {
    return WrongType("architectures", *arch_it, "a non-empty array");
  }
  auto arch_name = StringValue(arch_it->front(), "architectures[0]");
  if (!arch_name.ok()) {
    return arch_name.status();
  }
  auto architecture = ParseArchitecture(*arch_name);
  if (!architecture.ok()) {
    return architecture.status();
  }
  config.architecture = *architecture;
  config.architecture_name = *std::move(arch_name);
  return core::OkStatus();
}

// Required dimensions (design §3.2; max_position_embeddings is required
// by implementation note — both target families always serialize it and
// HF's class-level defaults differ per architecture).
core::Status ParseRequiredDims(const json& root, ModelConfig& config) {
  auto hidden_size = RequiredInt(root, "hidden_size");
  if (!hidden_size.ok()) {
    return hidden_size.status();
  }
  config.hidden_size = *hidden_size;

  auto intermediate_size = RequiredInt(root, "intermediate_size");
  if (!intermediate_size.ok()) {
    return intermediate_size.status();
  }
  config.intermediate_size = *intermediate_size;

  auto num_layers = RequiredInt(root, "num_hidden_layers");
  if (!num_layers.ok()) {
    return num_layers.status();
  }
  auto num_layers_int = AsInt(*num_layers, "num_hidden_layers");
  if (!num_layers_int.ok()) {
    return num_layers_int.status();
  }
  config.num_layers = *num_layers_int;

  auto num_heads = RequiredInt(root, "num_attention_heads");
  if (!num_heads.ok()) {
    return num_heads.status();
  }
  auto num_heads_int = AsInt(*num_heads, "num_attention_heads");
  if (!num_heads_int.ok()) {
    return num_heads_int.status();
  }
  config.num_heads = *num_heads_int;

  auto vocab_size = RequiredInt(root, "vocab_size");
  if (!vocab_size.ok()) {
    return vocab_size.status();
  }
  config.vocab_size = *vocab_size;

  auto max_position = RequiredInt(root, "max_position_embeddings");
  if (!max_position.ok()) {
    return max_position.status();
  }
  config.max_position_embeddings = *max_position;
  return core::OkStatus();
}

// Defaulted fields (HF transformers semantics).
core::Status ParseDefaultedFields(const json& root, ModelConfig& config) {
  std::int64_t num_kv_heads = config.num_heads;  // MHA when absent
  auto status = OptionalInt(root, "num_key_value_heads", num_kv_heads);
  if (!status.ok()) {
    return status;
  }
  auto num_kv_heads_int = AsInt(num_kv_heads, "num_key_value_heads");
  if (!num_kv_heads_int.ok()) {
    return num_kv_heads_int.status();
  }
  config.num_kv_heads = *num_kv_heads_int;

  status = OptionalFloat(root, "rope_theta", config.rope_theta);
  if (!status.ok()) {
    return status;
  }
  status = OptionalFloat(root, "rms_norm_eps", config.rms_norm_eps);
  if (!status.ok()) {
    return status;
  }
  status =
      OptionalBool(root, "tie_word_embeddings", config.tie_word_embeddings);
  if (!status.ok()) {
    return status;
  }

  // attention_bias defaults per architecture: Qwen2's modeling code
  // hardcodes QKV bias and its configs typically omit the field; an
  // explicit value wins over the default (design §3.2).
  config.attention_bias = config.architecture == Architecture::kQwen2;
  status = OptionalBool(root, "attention_bias", config.attention_bias);
  if (!status.ok()) {
    return status;
  }
  return core::OkStatus();
}

core::Status ParseTorchDtype(const json& root, ModelConfig& config) {
  const auto dtype_it = root.find("torch_dtype");
  if (dtype_it != root.end()) {
    auto dtype_name = StringValue(*dtype_it, "torch_dtype");
    if (!dtype_name.ok()) {
      return dtype_name.status();
    }
    auto dtype = tensor::from_string(*dtype_name);
    if (!dtype.ok()) {
      return core::InvalidArgumentError("field \"torch_dtype\": {}",
                                        dtype.status().message());
    }
    config.torch_dtype = *dtype;
  }
  return core::OkStatus();
}

// head_dim: explicit, or derived as hidden_size / num_heads. When
// explicit, `hidden_size == num_heads * head_dim` is deliberately not
// enforced (real checkpoints legitimately decouple them); when derived,
// it is the definition, so non-divisibility is an error.
core::Status ParseHeadDim(const json& root, ModelConfig& config) {
  const auto head_dim_it = root.find("head_dim");
  if (head_dim_it != root.end()) {
    auto head_dim = IntValue(*head_dim_it, "head_dim");
    if (!head_dim.ok()) {
      return head_dim.status();
    }
    auto head_dim_int = AsInt(*head_dim, "head_dim");
    if (!head_dim_int.ok()) {
      return head_dim_int.status();
    }
    config.head_dim = *head_dim_int;
  } else {
    if (config.num_heads <= 0) {
      return core::InvalidArgumentError(
          "field \"num_attention_heads\" must be strictly positive, got {}",
          config.num_heads);
    }
    if (config.hidden_size % config.num_heads != 0) {
      return core::InvalidArgumentError(
          "cannot derive \"head_dim\": \"hidden_size\" ({}) is not divisible "
          "by \"num_attention_heads\" ({})",
          config.hidden_size, config.num_heads);
    }
    auto head_dim_int =
        AsInt(config.hidden_size / config.num_heads, "head_dim");
    if (!head_dim_int.ok()) {
      return head_dim_int.status();
    }
    config.head_dim = *head_dim_int;
  }
  return core::OkStatus();
}

core::Status ParseRopeScalingField(const json& root, ModelConfig& config) {
  const auto rope_it = root.find("rope_scaling");
  if (rope_it != root.end() && !rope_it->is_null()) {
    auto scaling = ParseRopeScaling(*rope_it);
    if (!scaling.ok()) {
      return scaling.status();
    }
    config.rope_scaling = *std::move(scaling);
  }
  return core::OkStatus();
}

// Validation (design §3.2): every count/size strictly positive, and the
// GQA head grouping must divide evenly.
core::Status ValidateConfig(const ModelConfig& config) {
  auto status = CheckPositive(config.hidden_size, "hidden_size");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositive(config.intermediate_size, "intermediate_size");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositive(config.num_layers, "num_hidden_layers");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositive(config.num_heads, "num_attention_heads");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositive(config.num_kv_heads, "num_key_value_heads");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositive(config.head_dim, "head_dim");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositive(config.vocab_size, "vocab_size");
  if (!status.ok()) {
    return status;
  }
  status =
      CheckPositive(config.max_position_embeddings, "max_position_embeddings");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositiveFloat(config.rope_theta, "rope_theta");
  if (!status.ok()) {
    return status;
  }
  status = CheckPositiveFloat(config.rms_norm_eps, "rms_norm_eps");
  if (!status.ok()) {
    return status;
  }
  if (config.num_heads % config.num_kv_heads != 0) {
    return core::InvalidArgumentError(
        "field \"num_key_value_heads\" ({}) must divide "
        "\"num_attention_heads\" ({})",
        config.num_kv_heads, config.num_heads);
  }
  return core::OkStatus();
}

}  // namespace

core::StatusOr<ModelConfig> ParseModelConfig(std::string_view json_text) {
  const json root = json::parse(json_text.begin(), json_text.end(),
                                /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded()) {
    return core::InvalidArgumentError("config is not valid JSON");
  }
  if (!root.is_object()) {
    return core::InvalidArgumentError("config must be a JSON object, got {}",
                                      root.type_name());
  }

  ModelConfig config;
  using ParseStage = core::Status (*)(const json&, ModelConfig&);
  constexpr ParseStage kStages[] = {
      ParseArchitectureField, ParseRequiredDims, ParseDefaultedFields,
      ParseTorchDtype,        ParseHeadDim,      ParseRopeScalingField,
  };
  for (const ParseStage stage : kStages) {
    auto status = stage(root, config);
    if (!status.ok()) {
      return status;
    }
  }
  auto status = ValidateConfig(config);
  if (!status.ok()) {
    return status;
  }
  return config;
}

core::StatusOr<ModelConfig> LoadModelConfig(const std::filesystem::path& path) {
  const std::ifstream file(path, std::ios::binary);
  if (!file) {
    return core::NotFoundError("cannot read config file {}", path.string());
  }
  std::ostringstream contents;
  contents << file.rdbuf();
  if (file.bad()) {
    return core::NotFoundError("cannot read config file {}", path.string());
  }

  auto config = ParseModelConfig(contents.str());
  if (!config.ok()) {
    return core::Status(
        config.status().code(),
        fmt::format("{}: {}", path.string(), config.status().message()));
  }
  return config;
}

}  // namespace engine::model
