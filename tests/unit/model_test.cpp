#include "model/model.h"

#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_cache.h"
#include "kvcache/simple_cache.h"
#include "model/loader.h"
#include "model/modules.h"
#include "model/reference_model.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The full model forward: DecoderLayer/Mlp modules, ReferenceModel end-to-end
// logits, the activation hook, kLast/kAll consistency, and the KV-cache
// invariant at logits level (M5-T07; design: docs/design/model-execution.md §4,
// §5, §11, §13). The oracle is the tiny-llama activations.safetensors golden —
// HF's fp32 forward of the bf16 checkpoint, the same computation the reference
// performs, so agreement is far tighter than the Class-T tolerances stated.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::StatusOr;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::kvcache::PagedKvCache;
using engine::kvcache::SimpleKvCache;
using engine::model::ActivationEvent;
using engine::model::ActivationHook;
using engine::model::Attention;
using engine::model::DecoderLayer;
using engine::model::ForwardRequest;
using engine::model::Linear;
using engine::model::load_model;
using engine::model::LoadedModel;
using engine::model::LogitsMode;
using engine::model::Mlp;
using engine::model::ModelConfig;
using engine::model::ReferenceLinear;
using engine::model::ReferenceModel;
using engine::model::RmsNorm;
using engine::model::Rope;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// tiny-llama dims (fixture invariants — config.json).
constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 16;
constexpr int kHidden = 64;
constexpr int kLayers = 2;
constexpr std::int64_t kVocab = 512;
constexpr float kTheta = 10000.0F;
constexpr float kEps = 1e-5F;
constexpr std::int64_t kMaxPos = 128;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] CacheGeometry ModelGeometry() {
  return {.num_layers = kLayers,
          .num_kv_heads = kKvHeads,
          .head_dim = kHeadDim,
          .dtype = DataType::kFloat32};
}

[[nodiscard]] LoadedModel LoadTiny() {
  auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  EXPECT_TRUE(loaded.ok()) << loaded.status().ToString();
  return *std::move(loaded);
}

[[nodiscard]] std::unique_ptr<ReferenceModel> BuildTiny() {
  return Unwrap(ReferenceModel::Create(LoadTiny()));
}

[[nodiscard]] SafetensorsFile ActsFile() {
  auto file = SafetensorsFile::Open(
      engine::testing::FixturesDir() /
      "models/tiny-llama/expected/activations.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

// The committed prompt ids (fixture input_ids) and their [0..T-1] positions.
[[nodiscard]] std::vector<std::int32_t> PromptIds() {
  const SafetensorsFile acts = ActsFile();
  const Tensor ids = Unwrap(acts.tensor("input_ids"));  // [1, T] i32
  const std::int64_t t = ids.shape().dim(1);
  const auto* p = ids.data_ptr<std::int32_t>();
  return {p, p + t};
}

[[nodiscard]] std::vector<std::int32_t> Iota(std::int64_t n,
                                             std::int32_t start = 0) {
  std::vector<std::int32_t> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), start);
  return v;
}

// Fixture [1, T, C] activation reshaped to the reference's [T, C].
[[nodiscard]] Tensor Golden2D(const SafetensorsFile& acts,
                              const std::string& name) {
  const Tensor a = Unwrap(acts.tensor(name));  // [1, T, C]
  return Unwrap(a.reshape(Shape{a.shape().dim(1), a.shape().dim(2)}));
}

[[nodiscard]] Tensor Weight(const LoadedModel& m, const std::string& name) {
  const auto it = m.weights.find(name);
  EXPECT_NE(it, m.weights.end()) << "missing weight: " << name;
  return it->second;
}

[[nodiscard]] std::unique_ptr<Linear> MakeLinear(Tensor w) {
  return std::make_unique<ReferenceLinear>(
      Unwrap(ReferenceLinear::Create(std::move(w))));
}

// Builds one DecoderLayer from the loaded checkpoint (copying weight handles,
// so the model can be built independently) — used to isolate per-stage goldens.
[[nodiscard]] DecoderLayer BuildLayer(const LoadedModel& m, int layer) {
  const std::string pre = "layers." + std::to_string(layer) + ".";
  RmsNorm attn_norm =
      Unwrap(RmsNorm::Create(Weight(m, pre + "attn_norm.weight"), kEps));
  Rope rope = Unwrap(Rope::Create(kHeadDim, kTheta, std::nullopt, kMaxPos));
  Attention attn = Unwrap(
      Attention::Create(MakeLinear(Weight(m, pre + "attn.q_proj.weight")),
                        MakeLinear(Weight(m, pre + "attn.k_proj.weight")),
                        MakeLinear(Weight(m, pre + "attn.v_proj.weight")),
                        MakeLinear(Weight(m, pre + "attn.o_proj.weight")),
                        std::move(rope), kHeads, kKvHeads, kHeadDim));
  RmsNorm mlp_norm =
      Unwrap(RmsNorm::Create(Weight(m, pre + "mlp_norm.weight"), kEps));
  Mlp mlp =
      Unwrap(Mlp::Create(MakeLinear(Weight(m, pre + "mlp.gate_proj.weight")),
                         MakeLinear(Weight(m, pre + "mlp.up_proj.weight")),
                         MakeLinear(Weight(m, pre + "mlp.down_proj.weight"))));
  return Unwrap(DecoderLayer::Create(std::move(attn_norm), std::move(attn),
                                     std::move(mlp_norm), std::move(mlp)));
}

// Records every activation event for inspection (the debug-hook seam, §11).
class RecordingHook final : public ActivationHook {
 public:
  struct Event {
    std::string name;
    int layer;
    Tensor tensor;  // owned f32 copy (the borrowed view is valid only in-call)
  };

  void on_activation(const ActivationEvent& event) override {
    Tensor copy = Unwrap(ops::zeros(event.tensor.shape(), DataType::kFloat32));
    EXPECT_TRUE(ops::copy(copy, event.tensor).ok());
    events.push_back({std::string(event.name), event.layer, std::move(copy)});
  }

  std::vector<Event> events;
};

// ===========================================================================
// End-to-end logits vs the HF golden (acceptance criterion a).
// ===========================================================================

TEST(ReferenceModelTest, EndToEndLogitsMatchFixture) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const SafetensorsFile acts = ActsFile();
  const std::vector<std::int32_t> ids = PromptIds();
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));

  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const ForwardRequest req{.token_ids = ids,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kAll,
                           .hook = nullptr};
  const Tensor logits = Unwrap(model->forward(req));
  ASSERT_EQ(logits.shape(),
            (Shape{static_cast<std::int64_t>(ids.size()), kVocab}));

  const Tensor expected = Golden2D(acts, "logits");
  // Class T (HF reduces in its own order): the whole 2-layer forward + lm_head.
  // Tolerance documented; observed max-abs-diff printed and far below it.
  constexpr double kRtol = 2e-4;
  constexpr double kAtol = 2e-4;
  const ops::AllCloseResult r =
      Unwrap(ops::allclose(logits, expected, kRtol, kAtol));
  EXPECT_TRUE(r.allclose) << r.Summary();
  std::cerr << "[model] end-to-end logits max_abs_diff=" << r.max_abs_diff
            << "\n";
}

// ===========================================================================
// Per-layer debug hook dumps the layer tensors (acceptance criterion b).
// ===========================================================================

TEST(ReferenceModelTest, HookDumpsPerLayerActivations) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const SafetensorsFile acts = ActsFile();
  const std::vector<std::int32_t> ids = PromptIds();
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));

  RecordingHook hook;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const ForwardRequest req{.token_ids = ids,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kAll,
                           .hook = &hook};
  const Tensor logits = Unwrap(model->forward(req));

  // Event stream: embeddings (-1), layers.0 (0), layers.1 (1), final_norm (-1),
  // logits (-1) — the fixture keys, in forward order.
  ASSERT_EQ(hook.events.size(), 5U);
  EXPECT_EQ(hook.events[0].name, "embeddings");
  EXPECT_EQ(hook.events[0].layer, -1);
  EXPECT_EQ(hook.events[1].name, "layers.0");
  EXPECT_EQ(hook.events[1].layer, 0);
  EXPECT_EQ(hook.events[2].name, "layers.1");
  EXPECT_EQ(hook.events[2].layer, 1);
  EXPECT_EQ(hook.events[3].name, "final_norm");
  EXPECT_EQ(hook.events[3].layer, -1);
  EXPECT_EQ(hook.events[4].name, "logits");
  EXPECT_EQ(hook.events[4].layer, -1);

  // embeddings: pure gather + lossless bf16->f32 widen → bit-exact.
  const ops::AllCloseResult e = Unwrap(ops::allclose(
      hook.events[0].tensor, Golden2D(acts, "embeddings"), 0.0, 0.0));
  EXPECT_TRUE(e.allclose) << e.Summary();

  // The remaining stages are Class T; each localizes a per-layer regression.
  constexpr double kRtol = 2e-4;
  constexpr double kAtol = 2e-4;
  for (const std::string& name :
       {std::string("layers.0"), std::string("layers.1"),
        std::string("final_norm"), std::string("logits")}) {
    SCOPED_TRACE(name);
    const RecordingHook::Event* ev = nullptr;
    for (const auto& event : hook.events) {
      if (event.name == name) {
        ev = &event;
      }
    }
    ASSERT_NE(ev, nullptr);
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(ev->tensor, Golden2D(acts, name), kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[model] hook " << name << " max_abs_diff=" << r.max_abs_diff
              << "\n";
  }

  // The captured logits event equals the returned logits.
  const ops::AllCloseResult same =
      Unwrap(ops::allclose(hook.events[4].tensor, logits, 0.0, 0.0));
  EXPECT_TRUE(same.allclose) << same.Summary();
}

// A null hook makes no calls; the forward result is unchanged.
TEST(ReferenceModelTest, NullHookIsZeroCost) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const ForwardRequest req{.token_ids = ids,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kAll,
                           .hook = nullptr};
  EXPECT_TRUE(model->forward(req).ok());  // no hook, no crash
}

// ===========================================================================
// Stage-isolated goldens (independent of the model wiring) — covers
// DecoderLayer + Mlp against the fixture stage tensors.
// ===========================================================================

TEST(DecoderLayerTest, StageChainMatchesFixture) {
  const LoadedModel loaded = LoadTiny();
  const SafetensorsFile acts = ActsFile();
  const std::vector<std::int32_t> pos =
      Iota(Golden2D(acts, "embeddings").shape().dim(0));

  constexpr double kRtol = 2e-4;
  constexpr double kAtol = 2e-4;

  // layer 0: embeddings -> layers.0.
  const DecoderLayer l0 = BuildLayer(loaded, 0);
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
    const Tensor in = Golden2D(acts, "embeddings");
    Tensor out = Unwrap(ops::zeros(in.shape(), DataType::kFloat32));
    ASSERT_TRUE(l0.forward(in, pos, 0, cache, out).ok());
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, Golden2D(acts, "layers.0"), kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[model] layer0 max_abs_diff=" << r.max_abs_diff << "\n";
  }
  // layer 1: layers.0 -> layers.1.
  const DecoderLayer l1 = BuildLayer(loaded, 1);
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
    const Tensor in = Golden2D(acts, "layers.0");
    Tensor out = Unwrap(ops::zeros(in.shape(), DataType::kFloat32));
    ASSERT_TRUE(l1.forward(in, pos, 1, cache, out).ok());
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, Golden2D(acts, "layers.1"), kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[model] layer1 max_abs_diff=" << r.max_abs_diff << "\n";
  }
  // final norm: layers.1 -> final_norm.
  {
    const RmsNorm fn =
        Unwrap(RmsNorm::Create(Weight(loaded, "final_norm.weight"), kEps));
    const Tensor in = Golden2D(acts, "layers.1");
    Tensor out = Unwrap(ops::zeros(in.shape(), DataType::kFloat32));
    ASSERT_TRUE(fn.forward(in, out).ok());
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, Golden2D(acts, "final_norm"), kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
  }
  // lm_head: final_norm -> logits.
  {
    const std::unique_ptr<Linear> lm =
        MakeLinear(Weight(loaded, "lm_head.weight"));
    const Tensor in = Golden2D(acts, "final_norm");
    Tensor out = Unwrap(
        ops::zeros(Shape{in.shape().dim(0), kVocab}, DataType::kFloat32));
    ASSERT_TRUE(lm->forward(in, out).ok());
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, Golden2D(acts, "logits"), kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
  }
}

// ===========================================================================
// kLast vs kAll: the final row of kAll equals kLast (bit-exact — same op).
// ===========================================================================

TEST(ReferenceModelTest, LastLogitsMatchAllLastRow) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  const auto t = static_cast<std::int64_t>(ids.size());

  SimpleKvCache cache_all =
      Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor all = Unwrap(model->forward({.token_ids = ids,
                                            .positions = pos,
                                            .cache = &cache_all,
                                            .logits_mode = LogitsMode::kAll}));
  ASSERT_EQ(all.shape(), (Shape{t, kVocab}));

  SimpleKvCache cache_last =
      Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor last =
      Unwrap(model->forward({.token_ids = ids,
                             .positions = pos,
                             .cache = &cache_last,
                             .logits_mode = LogitsMode::kLast}));
  ASSERT_EQ(last.shape(), (Shape{1, kVocab}));

  const Tensor all_last = Unwrap(all.slice(0, t - 1, t));
  const ops::AllCloseResult r = Unwrap(ops::allclose(last, all_last, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// ===========================================================================
// The KV-cache invariant at logits level (§6.2, elevated from M5-T06).
// ===========================================================================

TEST(ReferenceModelTest, KvInvariantTokenByTokenMatchesFullPrefill) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const auto t = static_cast<std::int64_t>(ids.size());
  const std::vector<std::int32_t> pos = Iota(t);

  // Full prefill (kAll) → logits at every position.
  SimpleKvCache full = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor full_logits =
      Unwrap(model->forward({.token_ids = ids,
                             .positions = pos,
                             .cache = &full,
                             .logits_mode = LogitsMode::kAll}));

  // Token-by-token (kLast) through a growing cache → each step's last logit
  // must equal the corresponding full-prefill row. Bit-exact (a masked
  // softmax-0 key adds exactly 0.0, so the two schedules produce identical
  // sums).
  SimpleKvCache step = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  for (std::int64_t k = 0; k < t; ++k) {
    const std::span<const std::int32_t> one_id(ids.data() + k, 1);
    const std::span<const std::int32_t> one_pos(pos.data() + k, 1);
    const Tensor step_logits =
        Unwrap(model->forward({.token_ids = one_id,
                               .positions = one_pos,
                               .cache = &step,
                               .logits_mode = LogitsMode::kLast}));
    const Tensor full_row = Unwrap(full_logits.slice(0, k, k + 1));
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(step_logits, full_row, 0.0, 0.0));
    EXPECT_TRUE(r.allclose) << "step " << k << ": " << r.Summary();
  }
  EXPECT_EQ(step.length(), t);
}

TEST(ReferenceModelTest, KvInvariantChunkedPrefillAndTruncate) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const auto t = static_cast<std::int64_t>(ids.size());
  const std::vector<std::int32_t> pos = Iota(t);

  SimpleKvCache full = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor full_logits =
      Unwrap(model->forward({.token_ids = ids,
                             .positions = pos,
                             .cache = &full,
                             .logits_mode = LogitsMode::kAll}));

  // Chunked prefill [0,5) then [5,T) — must reproduce the last-row logit.
  const std::int64_t split = 5;
  SimpleKvCache chunk = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  EXPECT_TRUE(model
                  ->forward({.token_ids = std::span(ids.data(), split),
                             .positions = std::span(pos.data(), split),
                             .cache = &chunk,
                             .logits_mode = LogitsMode::kAll})
                  .ok());
  const Tensor chunk_logits = Unwrap(model->forward(
      {.token_ids =
           std::span(ids.data() + split, static_cast<std::size_t>(t - split)),
       .positions =
           std::span(pos.data() + split, static_cast<std::size_t>(t - split)),
       .cache = &chunk,
       .logits_mode = LogitsMode::kLast}));
  const Tensor full_last = Unwrap(full_logits.slice(0, t - 1, t));
  EXPECT_TRUE(
      Unwrap(ops::allclose(chunk_logits, full_last, 0.0, 0.0)).allclose);
  EXPECT_EQ(chunk.length(), t);

  // Truncate back to `split` and re-decode the suffix — same result, same
  // cache.
  ASSERT_TRUE(chunk.truncate(split).ok());
  EXPECT_EQ(chunk.length(), split);
  const Tensor redo = Unwrap(model->forward(
      {.token_ids =
           std::span(ids.data() + split, static_cast<std::size_t>(t - split)),
       .positions =
           std::span(pos.data() + split, static_cast<std::size_t>(t - split)),
       .cache = &chunk,
       .logits_mode = LogitsMode::kLast}));
  EXPECT_TRUE(Unwrap(ops::allclose(redo, full_last, 0.0, 0.0)).allclose);
}

// ===========================================================================
// Paged KV cache (M8-T06): the reference backend drives a PagedKvCache through
// the KvCache interface — Attention::forward's append + view (the contiguous
// gather) feed the same cpu::attention op. "Prefill with existing paged cache
// content matches the reference backend."
// ===========================================================================

// A block pool sized for the tiny model (bs=8 so the ~8-12-token prompt
// straddles several block boundaries; 8 blocks = 64 token slots).
[[nodiscard]] BlockPool FreshPool(int block_size = 8,
                                  std::int64_t num_blocks = 8) {
  auto p = BlockPool::Create(ModelGeometry(), block_size, num_blocks, nullptr);
  EXPECT_TRUE(p.ok()) << p.status().ToString();
  return *std::move(p);
}

// Prefill-continuation (P>0) through a paged cache is bit-exact to the same
// two-chunk run on a simple cache, and reproduces the one-shot full-prefill
// last-row logit.
TEST(ReferenceModelTest, PagedPrefillContinuationMatchesSimple) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const auto t = static_cast<std::int64_t>(ids.size());
  const std::vector<std::int32_t> pos = Iota(t);
  const std::int64_t split = t / 2;

  // One-shot full prefill (simple) → the reference last-row logit.
  SimpleKvCache full = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor full_logits =
      Unwrap(model->forward({.token_ids = ids,
                             .positions = pos,
                             .cache = &full,
                             .logits_mode = LogitsMode::kAll}));
  const Tensor full_last = Unwrap(full_logits.slice(0, t - 1, t));

  const auto two_chunks = [&](engine::kvcache::KvCache& cache) {
    EXPECT_TRUE(model
                    ->forward({.token_ids = std::span(
                                   ids.data(), static_cast<std::size_t>(split)),
                               .positions = std::span(
                                   pos.data(), static_cast<std::size_t>(split)),
                               .cache = &cache,
                               .logits_mode = LogitsMode::kAll})
                    .ok());
    return Unwrap(model->forward(
        {.token_ids =
             std::span(ids.data() + split, static_cast<std::size_t>(t - split)),
         .positions =
             std::span(pos.data() + split, static_cast<std::size_t>(t - split)),
         .cache = &cache,
         .logits_mode = LogitsMode::kLast}));
  };

  BlockPool pool = FreshPool();
  PagedKvCache paged(&pool);
  const Tensor paged_logits = two_chunks(paged);
  EXPECT_EQ(paged.length(), t);

  SimpleKvCache simple =
      Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor simple_logits = two_chunks(simple);

  // Bit-exact vs the same run on a simple cache.
  EXPECT_TRUE(
      Unwrap(ops::allclose(paged_logits, simple_logits, 0.0, 0.0)).allclose);
  // And reproduces the one-shot full-prefill last-row logit.
  EXPECT_TRUE(
      Unwrap(ops::allclose(paged_logits, full_last, 0.0, 0.0)).allclose);
}

// The KV invariant through the paged reference path: token-by-token decode
// reproduces the full prefill, bit-exact.
TEST(ReferenceModelTest, PagedKvInvariantTokenByTokenMatchesFullPrefill) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const auto t = static_cast<std::int64_t>(ids.size());
  const std::vector<std::int32_t> pos = Iota(t);

  SimpleKvCache full = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const Tensor full_logits =
      Unwrap(model->forward({.token_ids = ids,
                             .positions = pos,
                             .cache = &full,
                             .logits_mode = LogitsMode::kAll}));

  BlockPool pool = FreshPool();
  PagedKvCache step(&pool);
  for (std::int64_t k = 0; k < t; ++k) {
    const std::span<const std::int32_t> one_id(ids.data() + k, 1);
    const std::span<const std::int32_t> one_pos(pos.data() + k, 1);
    const Tensor step_logits =
        Unwrap(model->forward({.token_ids = one_id,
                               .positions = one_pos,
                               .cache = &step,
                               .logits_mode = LogitsMode::kLast}));
    const Tensor full_row = Unwrap(full_logits.slice(0, k, k + 1));
    EXPECT_TRUE(Unwrap(ops::allclose(step_logits, full_row, 0.0, 0.0)).allclose)
        << "step " << k;
  }
  EXPECT_EQ(step.length(), t);
}

// ===========================================================================
// Mlp unit behavior.
// ===========================================================================

TEST(MlpTest, CreateValidatesShapes) {
  const auto lin = [](std::int64_t out, std::int64_t in) {
    return MakeLinear(Unwrap(ops::zeros(Shape{out, in}, DataType::kFloat32)));
  };
  // E=4, I=8: gate/up are [I,E], down is [E,I].
  EXPECT_TRUE(Mlp::Create(lin(8, 4), lin(8, 4), lin(4, 8)).ok());
  EXPECT_TRUE(
      IsInvalidArgument(Mlp::Create(nullptr, lin(8, 4), lin(4, 8)).status()));
  EXPECT_TRUE(IsInvalidArgument(  // up disagrees on E
      Mlp::Create(lin(8, 4), lin(8, 5), lin(4, 8)).status()));
  EXPECT_TRUE(IsInvalidArgument(  // down maps wrong direction
      Mlp::Create(lin(8, 4), lin(8, 4), lin(8, 4)).status()));
}

TEST(MlpTest, ForwardComputesSwiGlu) {
  // E=1, I=1, hand-checked: gate=[[2]], up=[[3]], down=[[1]].
  // silu(2*x) * (3*x), then *1. For x=[[1]]: g=2,
  // silu(2)=2/(1+e^-2)=1.76159416, u=3 -> 5.28478248 -> down*1.
  Tensor gate_w = Unwrap(ops::zeros(Shape{1, 1}, DataType::kFloat32));
  Tensor up_w = Unwrap(ops::zeros(Shape{1, 1}, DataType::kFloat32));
  Tensor down_w = Unwrap(ops::zeros(Shape{1, 1}, DataType::kFloat32));
  gate_w.data_ptr<float>()[0] = 2.0F;
  up_w.data_ptr<float>()[0] = 3.0F;
  down_w.data_ptr<float>()[0] = 1.0F;
  const Mlp mlp = Unwrap(Mlp::Create(MakeLinear(std::move(gate_w)),
                                     MakeLinear(std::move(up_w)),
                                     MakeLinear(std::move(down_w))));
  const Tensor x = Unwrap(ops::zeros(Shape{1, 1}, DataType::kFloat32));
  x.data_ptr<float>()[0] = 1.0F;
  Tensor y = Unwrap(ops::zeros(Shape{1, 1}, DataType::kFloat32));
  ASSERT_TRUE(mlp.forward(x, y).ok());
  const float silu2 = 2.0F / (1.0F + std::exp(-2.0F));
  EXPECT_NEAR(y.data_ptr<float>()[0], silu2 * 3.0F, 1e-5F);

  // Wrong-shape x is rejected.
  const Tensor bad = Unwrap(ops::zeros(Shape{1, 2}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(mlp.forward(bad, y)));
}

// ===========================================================================
// DecoderLayer::Create hidden-size agreement.
// ===========================================================================

TEST(DecoderLayerTest, CreateValidatesHiddenAgreement) {
  const auto lin = [](std::int64_t out, std::int64_t in) {
    return MakeLinear(Unwrap(ops::zeros(Shape{out, in}, DataType::kFloat32)));
  };
  const auto norm = [](std::int64_t e) {
    return Unwrap(RmsNorm::Create(
        Unwrap(ops::zeros(Shape{e}, DataType::kFloat32)), kEps));
  };
  const auto make_attn = [&](std::int64_t e) {
    const std::int64_t qd = static_cast<std::int64_t>(kHeads) * kHeadDim;
    const std::int64_t kvd = static_cast<std::int64_t>(kKvHeads) * kHeadDim;
    return Unwrap(Attention::Create(
        lin(qd, e), lin(kvd, e), lin(kvd, e), lin(e, qd),
        Unwrap(Rope::Create(kHeadDim, kTheta, std::nullopt, kMaxPos)), kHeads,
        kKvHeads, kHeadDim));
  };
  // Consistent E=64: gate/up [I,E], down [E,I]; norms [E]; attn hidden=E.
  EXPECT_TRUE(DecoderLayer::Create(
                  norm(64), make_attn(64), norm(64),
                  Unwrap(Mlp::Create(lin(16, 64), lin(16, 64), lin(64, 16))))
                  .ok());
  // attn_norm disagrees on E.
  EXPECT_TRUE(IsInvalidArgument(
      DecoderLayer::Create(
          norm(32), make_attn(64), norm(64),
          Unwrap(Mlp::Create(lin(16, 64), lin(16, 64), lin(64, 16))))
          .status()));
  // mlp disagrees on E.
  EXPECT_TRUE(IsInvalidArgument(
      DecoderLayer::Create(
          norm(64), make_attn(64), norm(64),
          Unwrap(Mlp::Create(lin(16, 32), lin(16, 32), lin(32, 16))))
          .status()));
}

// ===========================================================================
// ReferenceModel::Create binding + geometry/config accessors.
// ===========================================================================

TEST(ReferenceModelTest, CreateReportsMissingWeight) {
  LoadedModel loaded = LoadTiny();
  loaded.weights.erase("layers.1.mlp.down_proj.weight");
  const auto model = ReferenceModel::Create(std::move(loaded));
  ASSERT_FALSE(model.ok());
  EXPECT_TRUE(IsInvalidArgument(model.status()));
  EXPECT_NE(model.status().message().find("layers.1.mlp.down_proj.weight"),
            std::string::npos)
      << model.status().ToString();
}

TEST(ReferenceModelTest, GeometryAndConfigMatchCheckpoint) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const CacheGeometry g = model->cache_geometry();
  EXPECT_EQ(g.num_layers, kLayers);
  EXPECT_EQ(g.num_kv_heads, kKvHeads);
  EXPECT_EQ(g.head_dim, kHeadDim);
  EXPECT_EQ(g.dtype, DataType::kFloat32);
  const ModelConfig& c = model->config();
  EXPECT_EQ(c.num_layers, kLayers);
  EXPECT_EQ(c.vocab_size, kVocab);
  EXPECT_EQ(c.hidden_size, kHidden);
}

// ===========================================================================
// forward() input validation & error posture (§5.3). On error the cache is
// left unchanged.
// ===========================================================================

TEST(ReferenceModelTest, ForwardRejectsMalformedInputs) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));

  // Empty token_ids.
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
    EXPECT_TRUE(IsInvalidArgument(
        model->forward({.token_ids = {}, .positions = {}, .cache = &cache})
            .status()));
  }
  // positions/token_ids length mismatch.
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
    EXPECT_TRUE(IsInvalidArgument(
        model
            ->forward({.token_ids = ids,
                       .positions = std::span(pos.data(), pos.size() - 1),
                       .cache = &cache})
            .status()));
  }
  // Out-of-range token id (message names index + value).
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
    std::vector<std::int32_t> bad = ids;
    bad[2] = static_cast<std::int32_t>(kVocab);
    const auto s =
        model->forward({.token_ids = bad, .positions = pos, .cache = &cache});
    ASSERT_TRUE(IsInvalidArgument(s.status()));
    EXPECT_NE(s.status().message().find("[2]"), std::string::npos)
        << s.status().ToString();
    EXPECT_EQ(cache.length(), 0);  // unchanged
  }
  // Out-of-range position.
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
    std::vector<std::int32_t> bad = pos;
    bad[0] = static_cast<std::int32_t>(kMaxPos);
    EXPECT_TRUE(IsInvalidArgument(
        model->forward({.token_ids = ids, .positions = bad, .cache = &cache})
            .status()));
  }
  // Null cache.
  EXPECT_TRUE(IsInvalidArgument(
      model->forward({.token_ids = ids, .positions = pos, .cache = nullptr})
          .status()));
  // Geometry mismatch (wrong Hkv).
  {
    SimpleKvCache cache =
        Unwrap(SimpleKvCache::Create({.num_layers = kLayers,
                                      .num_kv_heads = kKvHeads + 1,
                                      .head_dim = kHeadDim,
                                      .dtype = DataType::kFloat32},
                                     kMaxPos));
    EXPECT_TRUE(IsInvalidArgument(
        model->forward({.token_ids = ids, .positions = pos, .cache = &cache})
            .status()));
  }
}

TEST(ReferenceModelTest, ForwardRejectsOverCapacityCacheUnchanged) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const auto t = static_cast<std::int64_t>(ids.size());
  const std::vector<std::int32_t> pos = Iota(t);

  // Capacity smaller than the prompt → ResourceExhausted, cache untouched.
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(ModelGeometry(), t - 1));
  const auto s = model->forward({.token_ids = ids,
                                 .positions = pos,
                                 .cache = &cache,
                                 .logits_mode = LogitsMode::kAll});
  EXPECT_TRUE(IsResourceExhausted(s.status())) << s.status().ToString();
  EXPECT_EQ(cache.length(), 0);
}

// After a successful prefill, an over-capacity decode leaves the cache at P.
TEST(ReferenceModelTest, FailedDecodeLeavesCacheAtCommittedLength) {
  const std::unique_ptr<ReferenceModel> model = BuildTiny();
  const std::vector<std::int32_t> ids = PromptIds();
  const auto t = static_cast<std::int64_t>(ids.size());
  const std::vector<std::int32_t> pos = Iota(t);

  SimpleKvCache cache =
      Unwrap(SimpleKvCache::Create(ModelGeometry(), t));  // exact fit
  EXPECT_TRUE(model
                  ->forward({.token_ids = ids,
                             .positions = pos,
                             .cache = &cache,
                             .logits_mode = LogitsMode::kAll})
                  .ok());
  EXPECT_EQ(cache.length(), t);
  // One more token would exceed capacity == t.
  const std::int32_t next_id = ids[0];
  const auto next_pos = static_cast<std::int32_t>(t);
  const auto s = model->forward({.token_ids = std::span(&next_id, 1),
                                 .positions = std::span(&next_pos, 1),
                                 .cache = &cache});
  EXPECT_TRUE(IsResourceExhausted(s.status()));
  EXPECT_EQ(cache.length(), t);  // still committed at t
}

}  // namespace
