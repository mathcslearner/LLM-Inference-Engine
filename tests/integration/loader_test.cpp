#include "model/loader.h"

#include "common/paths.h"
#include "core/logging.h"
#include "core/status.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// load_model end-to-end (M4-T07; design: docs/design/model-loading.md §3.7).
// The happy path runs the committed tiny-llama fixture in both layouts and
// spot-checks loaded values against the fixture's recorded bytes; the tied
// and error cases stage variations of it (and one synthetic micro
// checkpoint) in scratch directories, asserting the status code and that
// the message names the offending file/weight (the actionability
// criterion, ticket acceptance).

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::model::load_model;
using engine::model::LoadedModel;
using engine::model::SafetensorsFile;
using engine::tensor::bfloat16;
using engine::tensor::DataType;
using engine::tensor::itemsize;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// --- helpers ---------------------------------------------------------------

[[nodiscard]] std::filesystem::path TinyLlamaDir() {
  return engine::testing::FixturesDir() / "models/tiny-llama";
}

[[nodiscard]] std::filesystem::path ScratchDir() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / "loader_test" /
      info->name();
  std::filesystem::create_directories(dir);
  return dir;
}

void WriteFile(const std::filesystem::path& path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(out.good()) << path;
}

[[nodiscard]] std::string ReadFile(const std::filesystem::path& path) {
  const std::ifstream in(path, std::ios::binary);
  std::ostringstream contents;
  contents << in.rdbuf();
  return std::move(contents).str();
}

// Copies the fixture's config.json into `dir`, with `from` → `to` applied
// once (e.g. flipping tie_word_embeddings, renaming the architecture).
void StagePatchedConfig(const std::filesystem::path& dir, std::string_view from,
                        std::string_view to) {
  std::string config = ReadFile(TinyLlamaDir() / "config.json");
  const std::size_t pos = config.find(from);
  ASSERT_NE(pos, std::string::npos) << from;
  config.replace(pos, from.size(), to);
  WriteFile(dir / "config.json", config);
}

// Raw bytes of a tensor (contiguous fixture views), for byte-identity
// checks across independently mapped copies of the same weights.
[[nodiscard]] std::string_view TensorBytes(const Tensor& tensor) {
  return {static_cast<const char*>(tensor.data()),
          static_cast<std::size_t>(tensor.numel()) *
              static_cast<std::size_t>(itemsize(tensor.dtype()))};
}

// The 21 canonical names of a 2-layer, bias-free, untied Llama.
[[nodiscard]] std::vector<std::string> TinyLlamaCanonicalNames() {
  std::vector<std::string> names = {"embed_tokens.weight", "final_norm.weight",
                                    "lm_head.weight"};
  for (int layer = 0; layer < 2; ++layer) {
    for (const std::string_view suffix :
         {"attn_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
          "attn.v_proj.weight", "attn.o_proj.weight", "mlp_norm.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight",
          "mlp.down_proj.weight"}) {
      names.push_back(fmt::format("layers.{}.{}", layer, suffix));
    }
  }
  return names;
}

// --- the committed fixture, single-file layout -----------------------------

TEST(LoaderTest, LoadsTinyLlamaEndToEnd) {
  const StatusOr<LoadedModel> model = load_model(TinyLlamaDir());
  ASSERT_TRUE(model.ok()) << model.status().ToString();

  EXPECT_EQ(model->config.architecture_name, "LlamaForCausalLM");
  EXPECT_EQ(model->config.num_layers, 2);
  EXPECT_EQ(model->config.hidden_size, 64);
  EXPECT_EQ(model->config.vocab_size, 512);
  EXPECT_FALSE(model->config.tie_word_embeddings);

  EXPECT_TRUE(model->report.missing.empty());
  EXPECT_TRUE(model->report.unexpected.empty());
  EXPECT_TRUE(model->report.ignored.empty());

  const std::vector<std::string> expected = TinyLlamaCanonicalNames();
  EXPECT_EQ(model->weights.size(), expected.size());
  for (const std::string& name : expected) {
    EXPECT_TRUE(model->weights.contains(name)) << name;
  }

  // Checkpoint dtype preserved (design §5): bf16 stays bf16.
  const Tensor& embed = model->weights.at("embed_tokens.weight");
  EXPECT_EQ(embed.dtype(), DataType::kBFloat16);
  EXPECT_EQ(embed.shape(), (Shape{512, 64}));
}

TEST(LoaderTest, WeightValuesMatchFixture) {
  const StatusOr<LoadedModel> model = load_model(TinyLlamaDir());
  ASSERT_TRUE(model.ok()) << model.status().ToString();

  // Absolute spot-checks: bf16 bit patterns read from the committed
  // model.safetensors (stable until the fixture is regenerated, which is a
  // reviewed change by the fixture rules).
  const Tensor& embed = model->weights.at("embed_tokens.weight");
  EXPECT_EQ((embed.item<bfloat16>({0, 0}).to_bits()), 0xBCA8);
  EXPECT_EQ((embed.item<bfloat16>({0, 1}).to_bits()), 0xBB0B);
  EXPECT_EQ((embed.item<bfloat16>({0, 2}).to_bits()), 0x3CEE);
  EXPECT_EQ((embed.item<bfloat16>({0, 63}).to_bits()), 0x3BBF);
  const Tensor& q1 = model->weights.at("layers.1.attn.q_proj.weight");
  EXPECT_EQ((q1.item<bfloat16>({0, 0}).to_bits()), 0xBCC4);
  EXPECT_EQ((q1.item<bfloat16>({0, 1}).to_bits()), 0x3C9D);

  // Cross-read: every spot-checked canonical weight is byte-identical to
  // its HF-named source read directly from the file.
  const auto file = SafetensorsFile::Open(TinyLlamaDir() / "model.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  const struct {
    std::string_view canonical;
    std::string_view source;
  } pairs[] = {
      {.canonical = "embed_tokens.weight",
       .source = "model.embed_tokens.weight"},
      {.canonical = "layers.1.attn.q_proj.weight",
       .source = "model.layers.1.self_attn.q_proj.weight"},
      {.canonical = "final_norm.weight", .source = "model.norm.weight"},
      {.canonical = "lm_head.weight", .source = "lm_head.weight"},
  };
  for (const auto& [canonical, source] : pairs) {
    const StatusOr<Tensor> direct = file->tensor(source);
    ASSERT_TRUE(direct.ok()) << direct.status().ToString();
    const Tensor& loaded = model->weights.at(std::string(canonical));
    EXPECT_EQ(loaded.dtype(), direct->dtype()) << canonical;
    EXPECT_EQ(loaded.shape(), direct->shape()) << canonical;
    EXPECT_EQ(TensorBytes(loaded), TensorBytes(*direct)) << canonical;
  }
}

// --- the committed fixture, sharded layout ---------------------------------

TEST(LoaderTest, ShardedLayoutMatchesSingleFile) {
  // The sharded fixture ships without a config.json (it shares tiny-llama's),
  // so stage a complete model directory: config + index + both shards.
  const std::filesystem::path dir = ScratchDir();
  std::filesystem::copy_file(TinyLlamaDir() / "config.json",
                             dir / "config.json",
                             std::filesystem::copy_options::overwrite_existing);
  for (const std::string_view name :
       {"model.safetensors.index.json", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"}) {
    std::filesystem::copy_file(
        TinyLlamaDir() / "sharded" / name, dir / name,
        std::filesystem::copy_options::overwrite_existing);
  }

  // Capture the load's log lines: §3.7's per-shard progress is part of the
  // contract, and this is the one test that maps shards.
  std::ostringstream log;
  engine::core::SetLogStreamForTesting(&log);
  const StatusOr<LoadedModel> sharded = load_model(dir);
  engine::core::SetLogStreamForTesting(nullptr);
  ASSERT_TRUE(sharded.ok()) << sharded.status().ToString();

  const StatusOr<LoadedModel> single = load_model(TinyLlamaDir());
  ASSERT_TRUE(single.ok()) << single.status().ToString();

  ASSERT_EQ(sharded->weights.size(), single->weights.size());
  for (const auto& [name, weight] : single->weights) {
    ASSERT_TRUE(sharded->weights.contains(name)) << name;
    EXPECT_EQ(TensorBytes(sharded->weights.at(name)), TensorBytes(weight))
        << name;
  }

  const std::string lines = log.str();
  EXPECT_NE(lines.find("mapped model-00001-of-00002.safetensors"),
            std::string::npos)
      << lines;
  EXPECT_NE(lines.find("mapped model-00002-of-00002.safetensors"),
            std::string::npos)
      << lines;
}

// --- tied embeddings -------------------------------------------------------

TEST(LoaderTest, TiedEmbeddingsShareOneTensor) {
  const std::filesystem::path dir = ScratchDir();
  StagePatchedConfig(dir, R"("tie_word_embeddings": false)",
                     R"("tie_word_embeddings": true)");
  std::filesystem::copy_file(TinyLlamaDir() / "model.safetensors",
                             dir / "model.safetensors",
                             std::filesystem::copy_options::overwrite_existing);

  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_TRUE(model.ok()) << model.status().ToString();

  // The alias is the *same* handle — shared storage, not equal bytes.
  EXPECT_EQ(model->weights.at("lm_head.weight").data(),
            model->weights.at("embed_tokens.weight").data());
  // The checkpoint's redundant lm_head.weight is superseded by the alias
  // and reported ignored (design §4).
  EXPECT_EQ(model->report.ignored,
            (std::vector<std::string>{"lm_head.weight"}));
  EXPECT_TRUE(model->report.unexpected.empty());
}

// --- dtype policy (design §5) ----------------------------------------------

// A complete, valid 1-layer Llama micro checkpoint (hidden 4, 2 heads, 2 KV
// heads, intermediate 8, vocab 8) whose tensors are all F32 zeros except
// `except_name`, which is written with `except_dtype` (same 4-byte
// itemsize, so offsets are unaffected). With no exception it loads clean;
// the dtype-policy test makes one weight I32, and the missing-weight test
// drops `omit_name` from the checkpoint entirely.
void StageMicroCheckpoint(const std::filesystem::path& dir,
                          std::string_view except_name,
                          std::string_view except_dtype,
                          std::string_view omit_name = "") {
  WriteFile(dir / "config.json", R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "tie_word_embeddings": false,
    "torch_dtype": "float32",
    "vocab_size": 8
  })");

  const struct {
    std::string_view name;
    std::vector<std::int64_t> dims;
  } tensors[] = {
      {.name = "model.embed_tokens.weight", .dims = {8, 4}},
      {.name = "model.layers.0.input_layernorm.weight", .dims = {4}},
      {.name = "model.layers.0.self_attn.q_proj.weight", .dims = {4, 4}},
      {.name = "model.layers.0.self_attn.k_proj.weight", .dims = {4, 4}},
      {.name = "model.layers.0.self_attn.v_proj.weight", .dims = {4, 4}},
      {.name = "model.layers.0.self_attn.o_proj.weight", .dims = {4, 4}},
      {.name = "model.layers.0.post_attention_layernorm.weight", .dims = {4}},
      {.name = "model.layers.0.mlp.gate_proj.weight", .dims = {8, 4}},
      {.name = "model.layers.0.mlp.up_proj.weight", .dims = {8, 4}},
      {.name = "model.layers.0.mlp.down_proj.weight", .dims = {4, 8}},
      {.name = "model.norm.weight", .dims = {4}},
      {.name = "lm_head.weight", .dims = {8, 4}},
  };

  std::string header = "{";
  std::uint64_t offset = 0;
  for (const auto& [name, dims] : tensors) {
    if (name == omit_name) {
      continue;
    }
    std::uint64_t numel = 1;
    for (const std::int64_t dim : dims) {
      numel *= static_cast<std::uint64_t>(dim);
    }
    const std::uint64_t end = offset + (numel * 4);
    const std::string_view dtype =
        name == except_name ? except_dtype : std::string_view("F32");
    header += fmt::format(
        R"("{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}},)", name,
        dtype, fmt::join(dims, ","), offset, end);
    offset = end;
  }
  header.back() = '}';
  // Space-pad the header (legal trailing JSON whitespace) so the data
  // section starts 8-aligned — what real HF serializers do, and what the
  // parser's absolute-alignment check expects.
  header.append((8 - (8 + header.size()) % 8) % 8, ' ');

  std::string bytes(8, '\0');
  const std::uint64_t header_len = header.size();
  std::memcpy(bytes.data(), &header_len, sizeof(header_len));
  bytes.append(header);
  bytes.append(offset, '\0');  // all-zeros data section
  WriteFile(dir / "model.safetensors", bytes);
}

TEST(LoaderTest, MicroCheckpointLoadsClean) {
  // Baseline: the synthetic checkpoint itself is valid, so the rejection
  // test below fails on the dtype and nothing else.
  const std::filesystem::path dir = ScratchDir();
  StageMicroCheckpoint(dir, "", "F32");
  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_TRUE(model.ok()) << model.status().ToString();
  EXPECT_EQ(model->weights.size(), 12U);
}

TEST(LoaderTest, MissingWeightFailsNamingTheCanonicalName) {
  // End-to-end wiring of CheckNoMissingWeights through load_model — the
  // per-list mechanics live in weight_map_test; this pins that a deleted
  // weight actually fails the *load* with the canonical name.
  const std::filesystem::path dir = ScratchDir();
  StageMicroCheckpoint(dir, "", "F32",
                       /*omit_name=*/"model.layers.0.self_attn.k_proj.weight");
  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsInvalidArgument(model.status())) << model.status().ToString();
  EXPECT_NE(model.status().message().find("layers.0.attn.k_proj.weight"),
            std::string::npos)
      << model.status().ToString();
}

TEST(LoaderTest, RejectsIntegerWeightDtype) {
  const std::filesystem::path dir = ScratchDir();
  StageMicroCheckpoint(dir, "model.layers.0.self_attn.q_proj.weight", "I32");
  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsUnimplemented(model.status())) << model.status().ToString();
  const std::string message(model.status().message());
  EXPECT_NE(message.find("layers.0.attn.q_proj.weight"), std::string::npos)
      << message;
  EXPECT_NE(message.find("M12"), std::string::npos) << message;
}

// --- error actionability ---------------------------------------------------

TEST(LoaderTest, MissingDirectoryIsNotFound) {
  const std::filesystem::path dir =
      engine::testing::FixturesDir() / "models/does-not-exist";
  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsNotFound(model.status())) << model.status().ToString();
  const std::string message(model.status().message());
  EXPECT_NE(message.find("config.json"), std::string::npos) << message;
  EXPECT_NE(message.find("does-not-exist"), std::string::npos) << message;
}

TEST(LoaderTest, ConfigLessDirectoryIsNotFound) {
  const std::filesystem::path dir = ScratchDir();  // exists, but empty
  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsNotFound(model.status())) << model.status().ToString();
  EXPECT_NE(std::string(model.status().message()).find("config.json"),
            std::string::npos)
      << model.status().ToString();
}

TEST(LoaderTest, PickleOnlyDirectorySaysConvert) {
  const std::filesystem::path dir = ScratchDir();
  std::filesystem::copy_file(TinyLlamaDir() / "config.json",
                             dir / "config.json",
                             std::filesystem::copy_options::overwrite_existing);
  WriteFile(dir / "pytorch_model.bin", "not really a pickle");

  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsNotFound(model.status())) << model.status().ToString();
  const std::string message(model.status().message());
  EXPECT_NE(message.find("pytorch_model.bin"), std::string::npos) << message;
  EXPECT_NE(message.find("convert"), std::string::npos) << message;
  EXPECT_NE(message.find("safetensors"), std::string::npos) << message;
}

TEST(LoaderTest, UnsupportedArchitectureIsUnimplemented) {
  const std::filesystem::path dir = ScratchDir();
  StagePatchedConfig(dir, "LlamaForCausalLM", "GPT2LMHeadModel");

  const StatusOr<LoadedModel> model = load_model(dir);
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsUnimplemented(model.status())) << model.status().ToString();
  const std::string message(model.status().message());
  EXPECT_NE(message.find("GPT2LMHeadModel"), std::string::npos) << message;
  EXPECT_NE(message.find("LlamaForCausalLM"), std::string::npos) << message;
}

}  // namespace
