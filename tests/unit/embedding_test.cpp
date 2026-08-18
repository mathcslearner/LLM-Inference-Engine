#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "model/loader.h"
#include "model/modules.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// cpu::embedding_lookup goldens & properties, and the Embedding module (M5-T04;
// design: docs/design/model-execution.md §4, §7, §12, §13). The lookup is a
// pure gather + bf16/f16->f32 widening, so agreement with the HF goldens is
// bit-exact (widening a bf16 checkpoint value is lossless and unique) — the
// fixture tests assert atol=rtol=0.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::Embedding;
using engine::model::load_model;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] SafetensorsFile OpsFile() {
  auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

[[nodiscard]] std::span<const std::int32_t> IdsSpan(const Tensor& ids) {
  return {ids.data_ptr<std::int32_t>(), static_cast<std::size_t>(ids.numel())};
}

// --- Fixture golden: embedding_edge (real bf16 embed table, edge ids) ---

TEST(CpuEmbeddingTest, MatchesOpsFixtureBitExact) {
  const SafetensorsFile file = OpsFile();
  const Tensor weight = Unwrap(file.tensor("embedding_edge.weight"));  // bf16
  const Tensor ids = Unwrap(file.tensor("embedding_edge.ids"));        // i32
  const Tensor expected = Unwrap(file.tensor("embedding_edge.y"));     // f32

  Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
  const Status s = engine::cpu::embedding_lookup(weight, IdsSpan(ids), out);
  ASSERT_TRUE(s.ok()) << s.ToString();

  // Widening a bf16 value to fp32 is lossless and matches torch's cast exactly,
  // so the gather agrees to the last bit (atol=rtol=0).
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, expected, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- End-to-end golden: the model's embedding row for the fixed prompt ---

TEST(CpuEmbeddingTest, MatchesActivationsGolden) {
  const auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const auto weight_it = loaded->weights.find("embed_tokens.weight");
  ASSERT_NE(weight_it, loaded->weights.end());
  const Tensor& weight = weight_it->second;  // [512, 64] bf16

  auto acts = SafetensorsFile::Open(
      engine::testing::FixturesDir() /
      "models/tiny-llama/expected/activations.safetensors");
  ASSERT_TRUE(acts.ok()) << acts.status().ToString();
  const Tensor input_ids = Unwrap(acts->tensor("input_ids"));  // [1, T] i32
  const Tensor embeddings =
      Unwrap(acts->tensor("embeddings"));  // [1, T, E] f32

  const std::int64_t seq = input_ids.shape().dim(1);
  const std::int64_t e = embeddings.shape().dim(2);
  const Tensor ids_flat = Unwrap(input_ids.reshape(Shape{seq}));
  const Tensor expected = Unwrap(embeddings.reshape(Shape{seq, e}));

  Tensor out = Unwrap(ops::zeros(Shape{seq, e}, DataType::kFloat32));
  ASSERT_TRUE(
      engine::cpu::embedding_lookup(weight, IdsSpan(ids_flat), out).ok());

  // The HF golden is model.float()[ids] — the same lossless bf16->f32 widening,
  // gathered. Bit-exact.
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, expected, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Widening path equivalence: bf16 gather == cast-to-f32 then gather ---

TEST(CpuEmbeddingTest, OnTheFlyWideningIsBitExact) {
  const SafetensorsFile file = OpsFile();
  const Tensor w_bf16 = Unwrap(file.tensor("embedding_edge.weight"));
  const Tensor ids = Unwrap(file.tensor("embedding_edge.ids"));
  const Tensor w_f32 = Unwrap(ops::cast(w_bf16, DataType::kFloat32));

  const std::int64_t t = ids.numel();
  const std::int64_t e = w_bf16.shape().dim(1);
  Tensor from_half = Unwrap(ops::zeros(Shape{t, e}, DataType::kFloat32));
  Tensor from_f32 = Unwrap(ops::zeros(Shape{t, e}, DataType::kFloat32));
  ASSERT_TRUE(
      engine::cpu::embedding_lookup(w_bf16, IdsSpan(ids), from_half).ok());
  ASSERT_TRUE(
      engine::cpu::embedding_lookup(w_f32, IdsSpan(ids), from_f32).ok());

  const ops::AllCloseResult r =
      Unwrap(ops::allclose(from_half, from_f32, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Threading independence: gather is bit-identical across thread counts ---

TEST(CpuEmbeddingTest, ThreadingIsBitExactVsSerial) {
  // A table and many ids (spread across threads); the serial gather is the
  // oracle. f32 table so the comparison is purely about traversal order.
  const std::int64_t vocab = 300;
  const std::int64_t e = 40;
  const Tensor table = Unwrap(ops::zeros(Shape{vocab, e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(table, 0.0, 1.0, 7).ok());

  std::vector<std::int32_t> ids;
  ids.reserve(257);
  for (int i = 0; i < 257; ++i) {
    ids.push_back(
        static_cast<std::int32_t>((static_cast<std::int64_t>(i) * 53) % vocab));
  }
  const std::span<const std::int32_t> ids_span(ids);

  Tensor out = Unwrap(ops::zeros(
      Shape{static_cast<std::int64_t>(ids.size()), e}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::embedding_lookup(table, ids_span, out).ok());

  // Serial reference gather.
  const Tensor naive = Unwrap(ops::zeros(
      Shape{static_cast<std::int64_t>(ids.size()), e}, DataType::kFloat32));
  const auto* td = table.data_ptr<float>();
  auto* nd = naive.data_ptr<float>();
  for (std::size_t t = 0; t < ids.size(); ++t) {
    for (std::int64_t j = 0; j < e; ++j) {
      nd[(static_cast<std::int64_t>(t) * e) + j] = td[(ids[t] * e) + j];
    }
  }
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, naive, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- The Embedding module wraps the op over the loaded weight ---

TEST(EmbeddingModuleTest, ForwardMatchesActivationsGolden) {
  const auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const auto weight_it = loaded->weights.find("embed_tokens.weight");
  ASSERT_NE(weight_it, loaded->weights.end());

  const Embedding embed = Unwrap(Embedding::Create(weight_it->second));
  EXPECT_EQ(embed.vocab_size(), 512);
  EXPECT_EQ(embed.hidden_size(), 64);

  auto acts = SafetensorsFile::Open(
      engine::testing::FixturesDir() /
      "models/tiny-llama/expected/activations.safetensors");
  ASSERT_TRUE(acts.ok()) << acts.status().ToString();
  const Tensor input_ids = Unwrap(acts->tensor("input_ids"));
  const Tensor embeddings = Unwrap(acts->tensor("embeddings"));
  const std::int64_t seq = input_ids.shape().dim(1);
  const std::int64_t e = embeddings.shape().dim(2);
  const Tensor ids_flat = Unwrap(input_ids.reshape(Shape{seq}));
  const Tensor expected = Unwrap(embeddings.reshape(Shape{seq, e}));

  Tensor out = Unwrap(ops::zeros(Shape{seq, e}, DataType::kFloat32));
  ASSERT_TRUE(embed.forward(IdsSpan(ids_flat), out).ok());

  const ops::AllCloseResult r = Unwrap(ops::allclose(out, expected, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Error paths (actionable messages naming the offending input) ---

TEST(CpuEmbeddingTest, OutOfRangeIdNamesIndexAndValue) {
  const Tensor table = Unwrap(ops::zeros(Shape{4, 3}, DataType::kFloat32));
  const std::vector<std::int32_t> ids = {0, 4};  // 4 == vocab, out of range
  Tensor out = Unwrap(ops::zeros(Shape{2, 3}, DataType::kFloat32));
  const Status s = engine::cpu::embedding_lookup(
      table, std::span<const std::int32_t>(ids), out);
  ASSERT_TRUE(IsInvalidArgument(s)) << s.ToString();
  EXPECT_NE(s.message().find("ids[1]"), std::string::npos) << s.message();
  EXPECT_NE(s.message().find('4'), std::string_view::npos) << s.message();
}

TEST(CpuEmbeddingTest, NegativeIdRejected) {
  const Tensor table = Unwrap(ops::zeros(Shape{4, 3}, DataType::kFloat32));
  const std::vector<std::int32_t> ids = {-1};
  Tensor out = Unwrap(ops::zeros(Shape{1, 3}, DataType::kFloat32));
  const Status s = engine::cpu::embedding_lookup(
      table, std::span<const std::int32_t>(ids), out);
  EXPECT_TRUE(IsInvalidArgument(s)) << s.ToString();
}

TEST(CpuEmbeddingTest, WrongYShapeRejected) {
  const Tensor table = Unwrap(ops::zeros(Shape{4, 3}, DataType::kFloat32));
  const std::vector<std::int32_t> ids = {0, 1};
  Tensor out = Unwrap(ops::zeros(Shape{2, 5}, DataType::kFloat32));  // E wrong
  EXPECT_TRUE(IsInvalidArgument(engine::cpu::embedding_lookup(
      table, std::span<const std::int32_t>(ids), out)));
}

TEST(CpuEmbeddingTest, WrongYDtypeRejected) {
  const Tensor table = Unwrap(ops::zeros(Shape{4, 3}, DataType::kFloat32));
  const std::vector<std::int32_t> ids = {0, 1};
  Tensor out = Unwrap(ops::zeros(Shape{2, 3}, DataType::kInt32));  // not f32
  EXPECT_TRUE(IsInvalidArgument(engine::cpu::embedding_lookup(
      table, std::span<const std::int32_t>(ids), out)));
}

TEST(CpuEmbeddingTest, EmptyIdsRejected) {
  const Tensor table = Unwrap(ops::zeros(Shape{4, 3}, DataType::kFloat32));
  Tensor out = Unwrap(ops::zeros(Shape{0, 3}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(engine::cpu::embedding_lookup(table, {}, out)));
}

TEST(CpuEmbeddingTest, NonRank2TableRejected) {
  const Tensor table = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  const std::vector<std::int32_t> ids = {0};
  Tensor out = Unwrap(ops::zeros(Shape{1, 4}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(engine::cpu::embedding_lookup(
      table, std::span<const std::int32_t>(ids), out)));
}

}  // namespace
