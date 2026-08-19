#include "common/paths.h"
#include "core/status.h"
#include "model/loader.h"
#include "model/modules.h"
#include "model/optimized_embedding.h"
#include "model/packed_linear.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// OptimizedEmbedding + the optimized logits path (M6-T06; design:
// optimized-cpu-execution.md §7, §10). Two ends of the model: the
// embedding-lookup gather (bit-exact vs the reference `Embedding`, both source
// layouts) and the lm_head projection (`PackedLinear` GEMM/GEMV vs
// `ReferenceLinear` within the §10 GEMM tolerance, and vs the HF logits
// golden). The tied case asserts the "one physical copy" property — the
// embedding shares the packed lm_head's storage. Registered SCALAR_PASS (§9).
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::Embedding;
using engine::model::Linear;
using engine::model::load_model;
using engine::model::LoadedModel;
using engine::model::OptimizedEmbedding;
using engine::model::PackedLinear;
using engine::model::ReferenceLinear;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] LoadedModel Load(const std::string& name) {
  auto loaded = load_model(engine::testing::FixturesDir() / "models" / name);
  EXPECT_TRUE(loaded.ok()) << loaded.status().ToString();
  return *std::move(loaded);
}

[[nodiscard]] Tensor Weight(const LoadedModel& m, const std::string& name) {
  const auto it = m.weights.find(name);
  EXPECT_NE(it, m.weights.end()) << "missing weight " << name;
  return it->second;
}

[[nodiscard]] std::span<const std::int32_t> IdsSpan(
    const std::vector<std::int32_t>& ids) {
  return {ids.data(), ids.size()};
}

// Random ids in [0, V) with repeats, plus the edge ids {0, V-1}.
[[nodiscard]] std::vector<std::int32_t> MakeIds(std::int64_t v,
                                                std::uint64_t seed) {
  std::vector<std::int32_t> ids = {0, static_cast<std::int32_t>(v - 1), 0,
                                   static_cast<std::int32_t>(v - 1)};
  std::uint64_t r = seed;
  for (int i = 0; i < 24; ++i) {
    r = (r * 6364136223846793005ULL) + 1442695040888963407ULL;
    ids.push_back(static_cast<std::int32_t>(r % static_cast<std::uint64_t>(v)));
  }
  return ids;
}

void ExpectBitExact(const Tensor& got, const Tensor& want) {
  const ops::AllCloseResult r = Unwrap(ops::allclose(got, want, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Create/forward validation mirrors the reference Embedding ---

TEST(OptimizedEmbeddingTest, FromTableRejectsMalformedWeight) {
  const Tensor r1 = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(OptimizedEmbedding::FromTable(r1).status()));
  const Tensor i32 = Unwrap(ops::zeros(Shape{4, 3}, DataType::kInt32));
  EXPECT_TRUE(IsInvalidArgument(OptimizedEmbedding::FromTable(i32).status()));
}

TEST(OptimizedEmbeddingTest, ForwardValidatesIdsAndOutput) {
  const Tensor table = Unwrap(ops::zeros(Shape{8, 4}, DataType::kFloat32));
  const OptimizedEmbedding emb = Unwrap(OptimizedEmbedding::FromTable(table));

  // Out-of-range id names the index and value.
  const std::vector<std::int32_t> bad = {0, 1, 8};  // 8 == V is out of range
  Tensor y = Unwrap(ops::zeros(Shape{3, 4}, DataType::kFloat32));
  const Status s = emb.forward(IdsSpan(bad), y);
  EXPECT_TRUE(IsInvalidArgument(s));

  // Wrong-shape y.
  const std::vector<std::int32_t> ok = {0, 1, 2};
  Tensor wrong = Unwrap(ops::zeros(Shape{3, 5}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(emb.forward(IdsSpan(ok), wrong)));
  // Non-fp32 y.
  Tensor bf = Unwrap(ops::zeros(Shape{3, 4}, DataType::kBFloat16));
  EXPECT_TRUE(IsInvalidArgument(emb.forward(IdsSpan(ok), bf)));
}

// --- Untied (tiny-llama): matches the reference Embedding bit-exactly ---

TEST(OptimizedEmbeddingTest, UntiedMatchesReferenceEmbedding) {
  const LoadedModel m = Load("tiny-llama");
  const Tensor table = Weight(m, "embed_tokens.weight");  // bf16 [512, 64]
  const std::int64_t v = table.shape().dim(0);
  const std::int64_t e = table.shape().dim(1);

  const Embedding ref = Unwrap(Embedding::Create(table));
  const OptimizedEmbedding opt = Unwrap(OptimizedEmbedding::FromTable(table));
  EXPECT_FALSE(opt.shares_packed_storage());
  EXPECT_EQ(opt.vocab_size(), v);
  EXPECT_EQ(opt.hidden_size(), e);

  const std::vector<std::int32_t> ids = MakeIds(v, 11);
  const auto rows = static_cast<std::int64_t>(ids.size());
  Tensor y_ref = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  Tensor y_opt = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  ASSERT_TRUE(ref.forward(IdsSpan(ids), y_ref).ok());
  ASSERT_TRUE(opt.forward(IdsSpan(ids), y_opt).ok());
  ExpectBitExact(y_opt, y_ref);
}

// --- Tied (tiny-qwen2): one physical copy shared with the packed lm_head ---

TEST(OptimizedEmbeddingTest, TiedSharesPackedLmHeadStorage) {
  const LoadedModel m = Load("tiny-qwen2");
  // Tied: lm_head.weight is aliased to embed_tokens.weight (same handle).
  const Tensor lm_weight = Weight(m, "lm_head.weight");  // bf16 [512, 64]
  const Tensor embed_weight = Weight(m, "embed_tokens.weight");
  const std::int64_t v = lm_weight.shape().dim(0);
  const std::int64_t e = lm_weight.shape().dim(1);

  const PackedLinear lm_head = Unwrap(PackedLinear::Create(lm_weight));
  const OptimizedEmbedding opt = OptimizedEmbedding::FromPackedLinear(lm_head);
  EXPECT_TRUE(opt.shares_packed_storage());
  EXPECT_EQ(opt.vocab_size(), v);
  EXPECT_EQ(opt.hidden_size(), e);
  // One physical copy: the embedding source is the lm_head's packed bytes.
  EXPECT_EQ(opt.source().data(), lm_head.packed_weight().data());

  // The gathered rows equal the reference lookup over the tied table.
  const Embedding ref = Unwrap(Embedding::Create(embed_weight));
  const std::vector<std::int32_t> ids = MakeIds(v, 23);
  const auto rows = static_cast<std::int64_t>(ids.size());
  Tensor y_ref = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  Tensor y_opt = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  ASSERT_TRUE(ref.forward(IdsSpan(ids), y_ref).ok());
  ASSERT_TRUE(opt.forward(IdsSpan(ids), y_opt).ok());
  ExpectBitExact(y_opt, y_ref);
}

// --- Storage outlives the PackedLinear it was built from (handle stability)
// ---

TEST(OptimizedEmbeddingTest, TiedStorageOutlivesMovedLmHead) {
  const LoadedModel m = Load("tiny-qwen2");
  const Tensor lm_weight = Weight(m, "lm_head.weight");
  const Tensor embed_weight = Weight(m, "embed_tokens.weight");
  const std::int64_t v = lm_weight.shape().dim(0);
  const std::int64_t e = lm_weight.shape().dim(1);

  PackedLinear lm_head = Unwrap(PackedLinear::Create(lm_weight));
  const OptimizedEmbedding opt = OptimizedEmbedding::FromPackedLinear(lm_head);
  // Move the linear behind a unique_ptr<Linear> (what OptimizedModel does):
  // the shared storage must survive, so the embedding still gathers correctly.
  const std::unique_ptr<Linear> owned =
      std::make_unique<PackedLinear>(std::move(lm_head));

  const Embedding ref = Unwrap(Embedding::Create(embed_weight));
  const std::vector<std::int32_t> ids = MakeIds(v, 5);
  const auto rows = static_cast<std::int64_t>(ids.size());
  Tensor y_ref = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  Tensor y_opt = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  ASSERT_TRUE(ref.forward(IdsSpan(ids), y_ref).ok());
  ASSERT_TRUE(opt.forward(IdsSpan(ids), y_opt).ok());
  ExpectBitExact(y_opt, y_ref);
  EXPECT_TRUE(owned != nullptr);
}

// --- Design §7 dedicated: gathered row v is the lm_head's logical row v ---

TEST(OptimizedEmbeddingTest, GatheredRowEqualsLmHeadLogicalRow) {
  const LoadedModel m = Load("tiny-qwen2");
  const Tensor lm_weight = Weight(m, "lm_head.weight");
  const std::int64_t v = lm_weight.shape().dim(0);
  const std::int64_t e = lm_weight.shape().dim(1);

  const PackedLinear lm_head = Unwrap(PackedLinear::Create(lm_weight));
  const OptimizedEmbedding opt = OptimizedEmbedding::FromPackedLinear(lm_head);

  // The shared-storage-across-layouts property (§7): the embedding's gathered
  // row v (out of the packed [P, E, kNr] lm_head) equals the lm_head's logical
  // row v — i.e. checkpoint row v widened to fp32. Sampled over ids incl. the
  // padded-panel edge V-1.
  const std::vector<std::int32_t> sample = {0, 7, 42,
                                            static_cast<std::int32_t>(v - 1)};
  for (const std::int32_t id : sample) {
    const std::vector<std::int32_t> one = {id};
    Tensor y_opt = Unwrap(ops::zeros(Shape{1, e}, DataType::kFloat32));
    ASSERT_TRUE(opt.forward(IdsSpan(one), y_opt).ok());
    // Reference: widen checkpoint row `id` to fp32.
    const Tensor row_bf =
        Unwrap(lm_weight.slice(0, id, id + 1));  // [1, e] bf16
    const Tensor row_ref = Unwrap(ops::cast(row_bf, DataType::kFloat32));
    ExpectBitExact(y_opt, row_ref);
  }
}

// --- Logits path: PackedLinear (GEMM + GEMV) vs ReferenceLinear & HF golden
// ---

void CheckLogitsPath(const std::string& model, double golden_atol) {
  const LoadedModel m = Load(model);
  const Tensor lm_weight = Weight(m, "lm_head.weight");
  const std::int64_t vocab = lm_weight.shape().dim(0);
  const std::int64_t e = lm_weight.shape().dim(1);

  // The HF activations golden: final_norm output [1, T, E] and logits
  // [1, T, vocab] (logits == lm_head(final_norm)).
  const auto acts =
      SafetensorsFile::Open(engine::testing::FixturesDir() / "models" / model /
                            "expected/activations.safetensors");
  ASSERT_TRUE(acts.ok()) << acts.status().ToString();
  const Tensor fn3 = Unwrap(acts->tensor("final_norm"));  // [1, T, E]
  const Tensor golden = Unwrap(acts->tensor("logits"));   // [1, T, vocab]
  const std::int64_t t = fn3.shape().dim(1);
  const Tensor x = Unwrap(fn3.reshape(Shape{t, e}));
  const Tensor golden2 = Unwrap(golden.reshape(Shape{t, vocab}));

  const ReferenceLinear ref = Unwrap(ReferenceLinear::Create(lm_weight));
  const PackedLinear packed = Unwrap(PackedLinear::Create(lm_weight));

  // kAll (GEMM shape): all T rows.
  Tensor l_ref = Unwrap(ops::zeros(Shape{t, vocab}, DataType::kFloat32));
  Tensor l_packed = Unwrap(ops::zeros(Shape{t, vocab}, DataType::kFloat32));
  ASSERT_TRUE(static_cast<const Linear&>(ref).forward(x, l_ref).ok());
  ASSERT_TRUE(static_cast<const Linear&>(packed).forward(x, l_packed).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(l_packed, l_ref, 1e-4, 1e-4)).allclose);
  // vs the HF logits golden.
  EXPECT_TRUE(
      Unwrap(ops::allclose(l_packed, golden2, 1e-4, golden_atol)).allclose);

  // kLast (GEMV shape): only the last row's hidden state. Must equal the last
  // row of the GEMM result bit-identically (PackedGemm routes M==1 to GEMV,
  // asserted bit-exact against GEMM per row in packed_gemm_test).
  const Tensor last = Unwrap(x.slice(0, t - 1, t));  // [1, E]
  Tensor l_last = Unwrap(ops::zeros(Shape{1, vocab}, DataType::kFloat32));
  ASSERT_TRUE(static_cast<const Linear&>(packed).forward(last, l_last).ok());
  const auto* gemm_last = l_packed.data_ptr<float>() + ((t - 1) * vocab);
  const auto* gemv = l_last.data_ptr<float>();
  for (std::int64_t j = 0; j < vocab; ++j) {
    EXPECT_EQ(gemm_last[j], gemv[j]) << "j=" << j;
  }
}

TEST(OptimizedLogitsTest, TinyLlamaLogitsMatchReferenceAndGolden) {
  CheckLogitsPath("tiny-llama", /*golden_atol=*/2e-4);
}

TEST(OptimizedLogitsTest, TinyQwen2LogitsMatchReferenceAndGolden) {
  CheckLogitsPath("tiny-qwen2", /*golden_atol=*/2e-4);
}

}  // namespace
