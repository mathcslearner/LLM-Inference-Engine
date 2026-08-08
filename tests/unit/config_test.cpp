#include "model/config.h"

#include "common/paths.h"
#include "core/status.h"
#include "tensor/dtype.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// M4-T03 (design: docs/design/model-loading.md §3.2, §8): golden parses of
// committed real config.json files assert every ModelConfig field; malformed
// inputs are recoverable errors whose message names the offending field —
// the actionability criterion is asserted, not aspirational.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::core::IsUnimplemented;
using engine::model::Architecture;
using engine::model::LoadModelConfig;
using engine::model::ParseModelConfig;
using engine::model::RopeScaling;
using engine::tensor::DataType;

bool MessageContains(const engine::core::Status& status,
                     std::string_view substr) {
  return status.message().find(substr) != std::string_view::npos;
}

// Assembles a config.json string from raw-JSON field values, so tests can
// override or remove one field at a time without hand-writing each variant.
std::string ConfigJson(
    const std::map<std::string, std::string, std::less<>>& overrides = {},
    const std::vector<std::string>& removed = {}) {
  std::map<std::string, std::string, std::less<>> fields = {
      {"architectures", R"(["LlamaForCausalLM"])"},
      {"hidden_size", "64"},
      {"intermediate_size", "128"},
      {"num_hidden_layers", "2"},
      {"num_attention_heads", "4"},
      {"vocab_size", "512"},
      {"max_position_embeddings", "256"},
  };
  for (const auto& [key, value] : overrides) {
    fields[key] = value;
  }
  for (const auto& key : removed) {
    fields.erase(key);
  }
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

// ------------------------------------------------- committed real configs --

TEST(ModelConfigTest, ParsesRealLlama31Config) {
  const auto config = LoadModelConfig(engine::testing::FixturesDir() /
                                      "models/configs/llama3/config.json");
  ASSERT_TRUE(config.ok()) << config.status().ToString();

  EXPECT_EQ(config->architecture, Architecture::kLlama);
  EXPECT_EQ(config->architecture_name, "LlamaForCausalLM");
  EXPECT_EQ(config->hidden_size, 4096);
  EXPECT_EQ(config->intermediate_size, 14336);
  EXPECT_EQ(config->num_layers, 32);
  EXPECT_EQ(config->num_heads, 32);
  EXPECT_EQ(config->num_kv_heads, 8);
  EXPECT_EQ(config->head_dim, 128);  // derived: 4096 / 32
  EXPECT_EQ(config->vocab_size, 128256);
  EXPECT_FLOAT_EQ(config->rope_theta, 500000.0F);
  EXPECT_FLOAT_EQ(config->rms_norm_eps, 1e-5F);
  EXPECT_EQ(config->max_position_embeddings, 131072);
  EXPECT_FALSE(config->tie_word_embeddings);
  EXPECT_FALSE(config->attention_bias);  // explicit in this config
  EXPECT_EQ(config->torch_dtype, DataType::kBFloat16);

  // The Llama-3.1 fixture is the rope_scaling *presence* golden.
  // (value_or after the ASSERT: bugprone-unchecked-optional-access cannot
  // follow gtest's ASSERT control flow, and value_or is checked access.)
  ASSERT_TRUE(config->rope_scaling.has_value());
  const RopeScaling scaling = config->rope_scaling.value_or(RopeScaling{});
  EXPECT_EQ(scaling.rope_type, "llama3");
  EXPECT_FLOAT_EQ(scaling.factor, 8.0F);
  EXPECT_FLOAT_EQ(scaling.low_freq_factor, 1.0F);
  EXPECT_FLOAT_EQ(scaling.high_freq_factor, 4.0F);
  EXPECT_EQ(scaling.original_max_position_embeddings, 8192);
}

TEST(ModelConfigTest, ParsesRealQwen2Config) {
  const auto config = LoadModelConfig(engine::testing::FixturesDir() /
                                      "models/configs/qwen2/config.json");
  ASSERT_TRUE(config.ok()) << config.status().ToString();

  EXPECT_EQ(config->architecture, Architecture::kQwen2);
  EXPECT_EQ(config->architecture_name, "Qwen2ForCausalLM");
  EXPECT_EQ(config->hidden_size, 896);
  EXPECT_EQ(config->intermediate_size, 4864);
  EXPECT_EQ(config->num_layers, 24);
  EXPECT_EQ(config->num_heads, 14);
  EXPECT_EQ(config->num_kv_heads, 2);
  EXPECT_EQ(config->head_dim, 64);  // derived: 896 / 14
  EXPECT_EQ(config->vocab_size, 151936);
  EXPECT_FLOAT_EQ(config->rope_theta, 1000000.0F);
  EXPECT_FLOAT_EQ(config->rms_norm_eps, 1e-6F);
  EXPECT_EQ(config->max_position_embeddings, 32768);
  EXPECT_TRUE(config->tie_word_embeddings);
  // The config omits attention_bias: the Qwen2 per-arch default applies.
  EXPECT_TRUE(config->attention_bias);
  EXPECT_EQ(config->torch_dtype, DataType::kBFloat16);
  // …and this fixture is the rope_scaling *absence* golden.
  EXPECT_FALSE(config->rope_scaling.has_value());
}

TEST(ModelConfigTest, ParsesTinyLlamaFixtureConfig) {
  const auto config = LoadModelConfig(engine::testing::FixturesDir() /
                                      "models/tiny-llama/config.json");
  ASSERT_TRUE(config.ok()) << config.status().ToString();

  EXPECT_EQ(config->architecture, Architecture::kLlama);
  EXPECT_EQ(config->hidden_size, 64);
  EXPECT_EQ(config->intermediate_size, 176);
  EXPECT_EQ(config->num_layers, 2);
  EXPECT_EQ(config->num_heads, 4);
  EXPECT_EQ(config->num_kv_heads, 2);
  EXPECT_EQ(config->head_dim, 16);
  EXPECT_EQ(config->vocab_size, 512);
  EXPECT_FLOAT_EQ(config->rope_theta, 10000.0F);
  EXPECT_FLOAT_EQ(config->rms_norm_eps, 1e-5F);
  EXPECT_EQ(config->max_position_embeddings, 128);
  EXPECT_FALSE(config->tie_word_embeddings);
  EXPECT_FALSE(config->attention_bias);
  EXPECT_EQ(config->torch_dtype, DataType::kBFloat16);
  EXPECT_FALSE(config->rope_scaling.has_value());
}

// ------------------------------------------------------- default semantics --

TEST(ModelConfigTest, MinimalConfigAppliesHfDefaults) {
  const auto config = ParseModelConfig(ConfigJson());
  ASSERT_TRUE(config.ok()) << config.status().ToString();

  EXPECT_EQ(config->num_kv_heads, config->num_heads);  // MHA when absent
  EXPECT_EQ(config->head_dim, 16);                     // 64 / 4
  EXPECT_FLOAT_EQ(config->rope_theta, 10000.0F);
  EXPECT_FLOAT_EQ(config->rms_norm_eps, 1e-6F);
  EXPECT_FALSE(config->tie_word_embeddings);
  EXPECT_FALSE(config->attention_bias);  // Llama default
  EXPECT_EQ(config->torch_dtype, DataType::kFloat32);
  EXPECT_FALSE(config->rope_scaling.has_value());
}

TEST(ModelConfigTest, AttentionBiasDefaultsTrueForQwen2) {
  const auto config = ParseModelConfig(
      ConfigJson({{"architectures", R"(["Qwen2ForCausalLM"])"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_EQ(config->architecture, Architecture::kQwen2);
  EXPECT_TRUE(config->attention_bias);
}

TEST(ModelConfigTest, ExplicitAttentionBiasBeatsPerArchDefault) {
  const auto config =
      ParseModelConfig(ConfigJson({{"architectures", R"(["Qwen2ForCausalLM"])"},
                                   {"attention_bias", "false"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_FALSE(config->attention_bias);
}

TEST(ModelConfigTest, ExplicitHeadDimIsNotCoupledToHiddenSize) {
  // Real checkpoints legitimately decouple head_dim from hidden_size /
  // num_heads; explicit values are taken as-is (design §3.2).
  const auto config = ParseModelConfig(ConfigJson({{"head_dim", "32"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_EQ(config->head_dim, 32);
}

TEST(ModelConfigTest, ExplicitNumKvHeadsParsed) {
  const auto config =
      ParseModelConfig(ConfigJson({{"num_key_value_heads", "2"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_EQ(config->num_kv_heads, 2);
}

TEST(ModelConfigTest, UnknownFieldsAreIgnored) {
  const auto config =
      ParseModelConfig(ConfigJson({{"sliding_window", "4096"},
                                   {"use_cache", "true"},
                                   {"bos_token_id", "1"},
                                   {"quantization_config", R"({"bits": 4})"}}));
  EXPECT_TRUE(config.ok()) << config.status().ToString();
}

TEST(ModelConfigTest, NonFloatTorchDtypeIsAcceptedAtParse) {
  // Weight-acceptance policy belongs to the loader (design §5); the parser
  // accepts any dtype the tensor library can name.
  const auto config =
      ParseModelConfig(ConfigJson({{"torch_dtype", "\"int8\""}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_EQ(config->torch_dtype, DataType::kInt8);
}

// ----------------------------------------------------------- rope_scaling --

TEST(ModelConfigTest, NullRopeScalingIsAbsent) {
  const auto config = ParseModelConfig(ConfigJson({{"rope_scaling", "null"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  EXPECT_FALSE(config->rope_scaling.has_value());
}

TEST(ModelConfigTest, RopeScalingSubfieldsDefaultWhenAbsent) {
  const auto config = ParseModelConfig(
      ConfigJson({{"rope_scaling", R"({"rope_type": "linear"})"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  ASSERT_TRUE(config->rope_scaling.has_value());
  const RopeScaling scaling = config->rope_scaling.value_or(RopeScaling{});
  EXPECT_EQ(scaling.rope_type, "linear");
  EXPECT_FLOAT_EQ(scaling.factor, 1.0F);
  EXPECT_FLOAT_EQ(scaling.low_freq_factor, 1.0F);
  EXPECT_FLOAT_EQ(scaling.high_freq_factor, 4.0F);
  EXPECT_EQ(scaling.original_max_position_embeddings, 0);
}

TEST(ModelConfigTest, LegacyRopeScalingTypeKeyIsAccepted) {
  // Configs written before transformers 4.43 spell it "type".
  const auto config = ParseModelConfig(
      ConfigJson({{"rope_scaling", R"({"type": "linear", "factor": 2.0})"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  ASSERT_TRUE(config->rope_scaling.has_value());
  const RopeScaling scaling = config->rope_scaling.value_or(RopeScaling{});
  EXPECT_EQ(scaling.rope_type, "linear");
  EXPECT_FLOAT_EQ(scaling.factor, 2.0F);
}

TEST(ModelConfigTest, UnknownRopeTypeIsParsedNotRejected) {
  // Rejection would couple config parsing to execution capability; the
  // executor rejects what it can't honor (design §3.2).
  const auto config = ParseModelConfig(
      ConfigJson({{"rope_scaling", R"({"rope_type": "hyperbolic"})"}}));
  ASSERT_TRUE(config.ok()) << config.status().ToString();
  ASSERT_TRUE(config->rope_scaling.has_value());
  EXPECT_EQ(config->rope_scaling.value_or(RopeScaling{}).rope_type,
            "hyperbolic");
}

TEST(ModelConfigTest, RopeScalingMissingTypeNamesField) {
  const auto config =
      ParseModelConfig(ConfigJson({{"rope_scaling", R"({"factor": 8.0})"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "rope_scaling.rope_type"))
      << config.status().ToString();
}

TEST(ModelConfigTest, RopeScalingWrongSubfieldTypeNamesDottedField) {
  const auto config = ParseModelConfig(ConfigJson(
      {{"rope_scaling", R"({"rope_type": "llama3", "factor": "big"})"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "rope_scaling.factor"))
      << config.status().ToString();
}

TEST(ModelConfigTest, RopeScalingWrongTypeIsError) {
  const auto config = ParseModelConfig(ConfigJson({{"rope_scaling", "42"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "rope_scaling"))
      << config.status().ToString();
}

// -------------------------------------------------------- malformed input --

TEST(ModelConfigTest, NonJsonInputIsInvalidArgument) {
  const auto config = ParseModelConfig("this is { not json");
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "not valid JSON"))
      << config.status().ToString();
}

TEST(ModelConfigTest, NonObjectRootIsInvalidArgument) {
  const auto config = ParseModelConfig("[1, 2, 3]");
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "JSON object"))
      << config.status().ToString();
}

TEST(ModelConfigTest, EachMissingRequiredFieldIsNamed) {
  const std::vector<std::string> required = {
      "architectures",          "hidden_size",         "intermediate_size",
      "num_hidden_layers",      "num_attention_heads", "vocab_size",
      "max_position_embeddings"};
  for (const auto& field : required) {
    const auto config = ParseModelConfig(ConfigJson({}, {field}));
    ASSERT_FALSE(config.ok()) << "parsing succeeded without " << field;
    EXPECT_TRUE(IsInvalidArgument(config.status())) << field;
    EXPECT_TRUE(MessageContains(config.status(), field))
        << field << ": " << config.status().ToString();
  }
}

TEST(ModelConfigTest, WrongTypesAreNamedErrors) {
  // (field, raw JSON value) — each must fail naming the field.
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"hidden_size", R"("4096")"},                // string, not integer
      {"hidden_size", "64.5"},                     // float, not integer
      {"architectures", R"("LlamaForCausalLM")"},  // string, not array
      {"architectures", "[]"},                     // empty array
      {"tie_word_embeddings", "1"},                // integer, not boolean
      {"rope_theta", R"("high")"},                 // string, not number
      {"torch_dtype", "16"},                       // number, not string
      {"num_hidden_layers", "true"},               // boolean, not integer
  };
  for (const auto& [field, value] : cases) {
    const auto config = ParseModelConfig(ConfigJson({{field, value}}));
    ASSERT_FALSE(config.ok()) << field << " = " << value;
    EXPECT_TRUE(IsInvalidArgument(config.status())) << field;
    EXPECT_TRUE(MessageContains(config.status(), field))
        << field << ": " << config.status().ToString();
  }
}

TEST(ModelConfigTest, NonStringArchitectureEntryIsNamed) {
  const auto config = ParseModelConfig(ConfigJson({{"architectures", "[42]"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "architectures[0]"))
      << config.status().ToString();
}

TEST(ModelConfigTest, NonPositiveDimsAreNamedErrors) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"hidden_size", "0"},         {"intermediate_size", "-176"},
      {"num_hidden_layers", "0"},   {"num_attention_heads", "-4"},
      {"vocab_size", "-512"},       {"max_position_embeddings", "0"},
      {"num_key_value_heads", "0"}, {"head_dim", "-16"},
      {"rope_theta", "0.0"},        {"rms_norm_eps", "-1e-5"},
  };
  for (const auto& [field, value] : cases) {
    const auto config = ParseModelConfig(ConfigJson({{field, value}}));
    ASSERT_FALSE(config.ok()) << field << " = " << value;
    EXPECT_TRUE(IsInvalidArgument(config.status())) << field;
    EXPECT_TRUE(MessageContains(config.status(), field))
        << field << ": " << config.status().ToString();
  }
}

TEST(ModelConfigTest, KvHeadsMustDivideHeads) {
  const auto config =
      ParseModelConfig(ConfigJson({{"num_key_value_heads", "3"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "num_key_value_heads"))
      << config.status().ToString();
  EXPECT_TRUE(MessageContains(config.status(), "num_attention_heads"))
      << config.status().ToString();
}

TEST(ModelConfigTest, DerivedHeadDimRequiresDivisibility) {
  const auto config = ParseModelConfig(ConfigJson({{"hidden_size", "65"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "head_dim"))
      << config.status().ToString();
}

TEST(ModelConfigTest, CountFieldOverflowIsNamedError) {
  const auto config =
      ParseModelConfig(ConfigJson({{"num_hidden_layers", "3000000000"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "num_hidden_layers"))
      << config.status().ToString();
}

TEST(ModelConfigTest, UnknownArchitectureIsUnimplementedListingSupported) {
  const auto config = ParseModelConfig(
      ConfigJson({{"architectures", R"(["MambaForCausalLM"])"}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsUnimplemented(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "MambaForCausalLM"))
      << config.status().ToString();
  EXPECT_TRUE(MessageContains(config.status(), "LlamaForCausalLM"))
      << config.status().ToString();
  EXPECT_TRUE(MessageContains(config.status(), "Qwen2ForCausalLM"))
      << config.status().ToString();
}

TEST(ModelConfigTest, UnknownTorchDtypeIsNamedError) {
  const auto config =
      ParseModelConfig(ConfigJson({{"torch_dtype", "\"float99\""}}));
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "torch_dtype"))
      << config.status().ToString();
}

// ----------------------------------------------------------- file loading --

TEST(ModelConfigTest, LoadFromMissingPathIsNotFoundNamingPath) {
  const auto config = LoadModelConfig("/no/such/dir/config.json");
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsNotFound(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "/no/such/dir/config.json"))
      << config.status().ToString();
}

TEST(ModelConfigTest, LoadParseErrorIsPrefixedWithPath) {
  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / "bad_config.json";
  {
    std::ofstream out(path);
    out << "{ definitely not json";
  }
  const auto config = LoadModelConfig(path);
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(IsInvalidArgument(config.status()));
  EXPECT_TRUE(MessageContains(config.status(), "bad_config.json"))
      << config.status().ToString();
  std::filesystem::remove(path);
}

}  // namespace
