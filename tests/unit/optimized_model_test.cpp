#include "model/optimized_model.h"

#include "common/paths.h"
#include "core/status.h"
#include "engine/generator.h"
#include "kvcache/kv_cache.h"
#include "kvcache/simple_cache.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/reference_model.h"
#include "model/registry.h"
#include "model/safetensors.h"
#include "model/workspace.h"
#include "sampling/params.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// OptimizedModel end-to-end (M6-T07; design:
// docs/design/optimized-cpu-execution.md §5, §6, §9, §10). The `kOptimized`
// backend behind the `Model` interface, validated two ways: (1) against the M5
// `ReferenceModel` (the oracle) token-for-token in greedy generation and
// within-tolerance on logits — the milestone's whole point (§2.2); (2) against
// the committed HF `activations.safetensors` / `generate.json` goldens the
// reference already passes. Both fixtures run: tiny-llama (untied embeddings,
// no bias) and tiny-qwen2 (tied embeddings + q/k/v biases, decoupled head_dim).
//
// Registered SCALAR_PASS (§9): the dispatched kernels run on the host's best
// ISA and, forced-scalar, on the scalar variants — the first full-model forward
// covered by the forced-scalar pass.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::StatusOr;
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::kvcache::CacheGeometry;
using engine::kvcache::KvCache;
using engine::kvcache::SimpleKvCache;
using engine::model::ActivationEvent;
using engine::model::ActivationHook;
using engine::model::Backend;
using engine::model::BuildModel;
using engine::model::BuildOptions;
using engine::model::ForwardRequest;
using engine::model::load_model;
using engine::model::LoadedModel;
using engine::model::LogitsMode;
using engine::model::Model;
using engine::model::OptimizedModel;
using engine::model::SafetensorsFile;
using engine::model::Workspace;
using engine::sampling::SamplingParams;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

constexpr const char* kLlama = "models/tiny-llama";
constexpr const char* kQwen = "models/tiny-qwen2";

// Optimized-vs-reference logit band: the T02/T06 GEMM tolerance, ~2 layers
// deep. Observed max-abs-diff is printed by each test and is far below this.
constexpr double kRtol = 2e-4;
constexpr double kAtol = 2e-4;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

void Unwrap0(const engine::core::Status& status) {
  EXPECT_TRUE(status.ok()) << status.ToString();
}

[[nodiscard]] LoadedModel Load(const std::string& fixture) {
  auto loaded = load_model(engine::testing::FixturesDir() / fixture);
  EXPECT_TRUE(loaded.ok()) << loaded.status().ToString();
  return *std::move(loaded);
}

// Builds a backend through the architecture registry (the acceptance path):
// loads the fixture fresh (BuildModel consumes the LoadedModel) and dispatches.
[[nodiscard]] std::unique_ptr<Model> Build(const std::string& fixture,
                                           Backend backend) {
  return Unwrap(BuildModel(Load(fixture), BuildOptions{.backend = backend}));
}

[[nodiscard]] std::unique_ptr<Model> Optimized(const std::string& fixture) {
  return Build(fixture, Backend::kOptimized);
}

[[nodiscard]] std::unique_ptr<Model> Reference(const std::string& fixture) {
  return Build(fixture, Backend::kReference);
}

[[nodiscard]] SimpleKvCache FreshCache(const Model& model,
                                       std::int64_t capacity = 128) {
  return Unwrap(SimpleKvCache::Create(model.cache_geometry(), capacity));
}

[[nodiscard]] std::vector<std::int32_t> Iota(std::int64_t n) {
  std::vector<std::int32_t> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), 0);
  return v;
}

[[nodiscard]] SafetensorsFile ActsFile(const std::string& fixture) {
  auto file = SafetensorsFile::Open(engine::testing::FixturesDir() / fixture /
                                    "expected/activations.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

[[nodiscard]] std::vector<std::int32_t> PromptIds(const std::string& fixture) {
  const SafetensorsFile acts = ActsFile(fixture);
  const Tensor ids = Unwrap(acts.tensor("input_ids"));  // [1, T] i32
  const std::int64_t t = ids.shape().dim(1);
  const auto* p = ids.data_ptr<std::int32_t>();
  return {p, p + t};
}

[[nodiscard]] Tensor Golden2D(const SafetensorsFile& acts,
                              const std::string& name) {
  const Tensor a = Unwrap(acts.tensor(name));  // [1, T, C]
  return Unwrap(a.reshape(Shape{a.shape().dim(1), a.shape().dim(2)}));
}

// Runs one forward on a fresh cache and returns the logits.
[[nodiscard]] Tensor ForwardOnce(Model& model,
                                 std::span<const std::int32_t> ids,
                                 LogitsMode mode, KvCache& cache) {
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  const ForwardRequest req{.token_ids = ids,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = mode,
                           .hook = nullptr};
  return Unwrap(model.forward(req));
}

// Recording hook (mirrors model_test's) for the event-stream parity check.
class RecordingHook final : public ActivationHook {
 public:
  struct Event {
    std::string name;
    int layer;
    Tensor tensor;
  };
  void on_activation(const ActivationEvent& event) override {
    Tensor copy = Unwrap(ops::zeros(event.tensor.shape(), DataType::kFloat32));
    EXPECT_TRUE(ops::copy(copy, event.tensor).ok());
    events.push_back({std::string(event.name), event.layer, std::move(copy)});
  }
  std::vector<Event> events;
};

// One greedy-generation golden case from generate.json.
struct GenerateCase {
  std::string name;
  std::vector<std::int32_t> prompt_ids;
  std::vector<std::int32_t> generated_ids;
};

[[nodiscard]] std::vector<std::int32_t> IdList(const nlohmann::json& arr) {
  std::vector<std::int32_t> v;
  for (const auto& e : arr) {
    v.push_back(e.get<std::int32_t>());
  }
  return v;
}

[[nodiscard]] nlohmann::json GenerateJson(const std::string& fixture) {
  const std::filesystem::path path =
      engine::testing::FixturesDir() / fixture / "expected/generate.json";
  const std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "open " << path;
  std::stringstream ss;
  ss << in.rdbuf();
  return nlohmann::json::parse(ss.str());
}

[[nodiscard]] std::vector<GenerateCase> LoadGoldenCases(
    const std::string& fixture) {
  const nlohmann::json root = GenerateJson(fixture);
  std::vector<GenerateCase> cases;
  for (const auto& c : root.at("cases")) {
    cases.push_back({c.at("name").get<std::string>(),
                     IdList(c.at("prompt_ids")),
                     IdList(c.at("generated_ids"))});
  }
  EXPECT_FALSE(cases.empty());
  return cases;
}

[[nodiscard]] std::int64_t GoldenMaxNew(const std::string& fixture) {
  return GenerateJson(fixture).at("max_new_tokens").get<std::int64_t>();
}

// ===========================================================================
// Build: registry dispatch, tied/untied embedding storage, geometry.
// ===========================================================================

TEST(OptimizedModelTest, BuildsBothFixturesThroughRegistry) {
  EXPECT_NE(Optimized(kLlama), nullptr);
  EXPECT_NE(Optimized(kQwen), nullptr);
}

TEST(OptimizedModelTest, GeometryAndConfigMatchReference) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    const std::unique_ptr<Model> opt = Optimized(fixture);
    const std::unique_ptr<Model> ref = Reference(fixture);
    const CacheGeometry go = opt->cache_geometry();
    const CacheGeometry gr = ref->cache_geometry();
    EXPECT_EQ(go.num_layers, gr.num_layers);
    EXPECT_EQ(go.num_kv_heads, gr.num_kv_heads);
    EXPECT_EQ(go.head_dim, gr.head_dim);
    EXPECT_EQ(go.dtype, gr.dtype);
    EXPECT_EQ(opt->config().vocab_size, ref->config().vocab_size);
    EXPECT_EQ(opt->config().num_layers, ref->config().num_layers);
  }
}

TEST(OptimizedModelTest, CreateReportsMissingWeight) {
  LoadedModel loaded = Load(kLlama);
  loaded.weights.erase("layers.0.attn.q_proj.weight");
  const auto built = OptimizedModel::Create(std::move(loaded));
  ASSERT_FALSE(built.ok());
  EXPECT_TRUE(IsInvalidArgument(built.status()));
  EXPECT_NE(built.status().message().find("q_proj"), std::string::npos)
      << built.status().ToString();
}

// ===========================================================================
// Logits: vs the reference backend, and vs the HF activations golden.
// ===========================================================================

TEST(OptimizedModelTest, LogitsMatchReference) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    for (const LogitsMode mode : {LogitsMode::kAll, LogitsMode::kLast}) {
      std::unique_ptr<Model> opt = Optimized(fixture);
      std::unique_ptr<Model> ref = Reference(fixture);
      const std::vector<std::int32_t> ids = PromptIds(fixture);

      SimpleKvCache co = FreshCache(*opt);
      SimpleKvCache cr = FreshCache(*ref);
      const Tensor lo = ForwardOnce(*opt, ids, mode, co);
      const Tensor lr = ForwardOnce(*ref, ids, mode, cr);
      ASSERT_EQ(lo.shape(), lr.shape());
      const ops::AllCloseResult r = Unwrap(ops::allclose(lo, lr, kRtol, kAtol));
      EXPECT_TRUE(r.allclose) << r.Summary();
      std::cerr << "[opt] " << fixture
                << (mode == LogitsMode::kAll ? " kAll" : " kLast")
                << " vs reference max_abs_diff=" << r.max_abs_diff << "\n";
    }
  }
}

TEST(OptimizedModelTest, LogitsMatchHfGolden) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    const SafetensorsFile acts = ActsFile(fixture);
    const std::vector<std::int32_t> ids = PromptIds(fixture);
    SimpleKvCache cache = FreshCache(*opt);
    const Tensor logits = ForwardOnce(*opt, ids, LogitsMode::kAll, cache);

    const Tensor expected = Golden2D(acts, "logits");
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(logits, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[opt] " << fixture
              << " vs HF golden max_abs_diff=" << r.max_abs_diff << "\n";
  }
}

// kLast logits equal the kAll last row (a real compute saving that stays
// correct, §6.3). GEMV vs GEMM last row is bit-identical (T02), so exact.
TEST(OptimizedModelTest, LastLogitsMatchAllLastRow) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  SimpleKvCache ca = FreshCache(*opt);
  SimpleKvCache cl = FreshCache(*opt);
  const Tensor all = ForwardOnce(*opt, ids, LogitsMode::kAll, ca);
  const Tensor last = ForwardOnce(*opt, ids, LogitsMode::kLast, cl);
  const Tensor all_last =
      Unwrap(all.slice(0, all.shape().dim(0) - 1, all.shape().dim(0)));
  const ops::AllCloseResult r = Unwrap(ops::allclose(last, all_last, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// ===========================================================================
// Per-layer hook: same event names/order/shapes as the reference.
// ===========================================================================

TEST(OptimizedModelTest, HookEventsMatchReference) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  std::unique_ptr<Model> ref = Reference(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));

  RecordingHook ho;
  RecordingHook hr;
  SimpleKvCache co = FreshCache(*opt);
  SimpleKvCache cr = FreshCache(*ref);
  const ForwardRequest ro{.token_ids = ids,
                          .positions = pos,
                          .cache = &co,
                          .logits_mode = LogitsMode::kAll,
                          .hook = &ho};
  const ForwardRequest rr{.token_ids = ids,
                          .positions = pos,
                          .cache = &cr,
                          .logits_mode = LogitsMode::kAll,
                          .hook = &hr};
  EXPECT_TRUE(opt->forward(ro).ok());
  EXPECT_TRUE(ref->forward(rr).ok());

  ASSERT_EQ(ho.events.size(), hr.events.size());
  ASSERT_EQ(ho.events.size(),
            5U);  // embeddings, layers.0, layers.1, fn, logits
  for (std::size_t i = 0; i < ho.events.size(); ++i) {
    SCOPED_TRACE(ho.events[i].name);
    EXPECT_EQ(ho.events[i].name, hr.events[i].name);
    EXPECT_EQ(ho.events[i].layer, hr.events[i].layer);
    // embeddings is a pure gather+widen → bit-exact; the rest are Class T.
    const bool exact = ho.events[i].name == "embeddings";
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(ho.events[i].tensor, hr.events[i].tensor,
                             exact ? 0.0 : kRtol, exact ? 0.0 : kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
  }
}

// ===========================================================================
// Greedy generation: token-for-token vs reference and vs the HF golden.
// ===========================================================================

TEST(OptimizedModelTest, GreedyMatchesReferenceAndGolden) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    const std::int64_t max_new = GoldenMaxNew(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    std::unique_ptr<Model> ref = Reference(fixture);
    const GenerateOptions options{.sampling = SamplingParams::Greedy(max_new),
                                  .eos_ids = {}};

    for (const GenerateCase& c : LoadGoldenCases(fixture)) {
      SCOPED_TRACE(c.name);
      SimpleKvCache co = FreshCache(*opt, 256);
      SimpleKvCache cr = FreshCache(*ref, 256);
      const std::vector<std::int32_t> got_opt =
          Unwrap(Generate(*opt, co, c.prompt_ids, options));
      const std::vector<std::int32_t> got_ref =
          Unwrap(Generate(*ref, cr, c.prompt_ids, options));
      EXPECT_EQ(got_opt, c.generated_ids);  // vs HF golden
      EXPECT_EQ(got_opt, got_ref);          // vs reference backend
    }
  }
}

TEST(OptimizedModelTest, GenerationIsDeterministic) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const GenerateCase c = LoadGoldenCases(kLlama).front();
  const GenerateOptions options{.sampling = SamplingParams::Greedy(24),
                                .eos_ids = {}};
  SimpleKvCache ca = FreshCache(*opt, 256);
  SimpleKvCache cb = FreshCache(*opt, 256);
  const std::vector<std::int32_t> a =
      Unwrap(Generate(*opt, ca, c.prompt_ids, options));
  const std::vector<std::int32_t> b =
      Unwrap(Generate(*opt, cb, c.prompt_ids, options));
  EXPECT_EQ(a, b);
}

// ===========================================================================
// KV invariant on the optimized backend: full prefill vs token-by-token,
// chunked prefill, and continuation from a non-empty cache.
// ===========================================================================

TEST(OptimizedModelTest, KvInvariantTokenByTokenMatchesFullPrefill) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    const std::vector<std::int32_t> ids = PromptIds(fixture);
    const auto t = static_cast<std::int64_t>(ids.size());

    // Full prefill.
    SimpleKvCache full = FreshCache(*opt);
    const Tensor logits_full = ForwardOnce(*opt, ids, LogitsMode::kLast, full);

    // Token-by-token: one forward per id, position = running length.
    SimpleKvCache step = FreshCache(*opt);
    Tensor logits_step;
    for (std::int64_t i = 0; i < t; ++i) {
      const std::array<std::int32_t, 1> one{ids[static_cast<std::size_t>(i)]};
      const std::array<std::int32_t, 1> pos{static_cast<std::int32_t>(i)};
      const ForwardRequest req{.token_ids = one,
                               .positions = pos,
                               .cache = &step,
                               .logits_mode = LogitsMode::kLast,
                               .hook = nullptr};
      logits_step = Unwrap(opt->forward(req));
    }
    // Both are the last-token logits after the same T tokens: bit-exact
    // (masked keys contribute exactly 0, and each output row's recurrence is
    // within one thread; §5/§10).
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(logits_step, logits_full, 0.0, 0.0));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[opt] " << fixture
              << " KV invariant max_abs_diff=" << r.max_abs_diff << "\n";
  }
}

TEST(OptimizedModelTest, KvInvariantChunkedPrefill) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  const auto t = static_cast<std::int64_t>(ids.size());
  ASSERT_GE(t, 4);

  SimpleKvCache full = FreshCache(*opt);
  const Tensor logits_full = ForwardOnce(*opt, ids, LogitsMode::kLast, full);

  // Two chunks: [0, split) then [split, T).
  const std::int64_t split = t / 2;
  SimpleKvCache chunked = FreshCache(*opt);
  {
    const std::span<const std::int32_t> a(ids.data(),
                                          static_cast<std::size_t>(split));
    const std::vector<std::int32_t> pa = Iota(split);
    const ForwardRequest req{.token_ids = a,
                             .positions = pa,
                             .cache = &chunked,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
    EXPECT_TRUE(opt->forward(req).ok());
  }
  Tensor logits_chunked;
  {
    const std::span<const std::int32_t> b(ids.data() + split,
                                          static_cast<std::size_t>(t - split));
    std::vector<std::int32_t> pb(static_cast<std::size_t>(t - split));
    std::iota(pb.begin(), pb.end(), static_cast<std::int32_t>(split));
    const ForwardRequest req{.token_ids = b,
                             .positions = pb,
                             .cache = &chunked,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
    logits_chunked = Unwrap(opt->forward(req));
  }
  const ops::AllCloseResult r =
      Unwrap(ops::allclose(logits_chunked, logits_full, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// ===========================================================================
// Workspace: monotone growth, no reallocation when T <= capacity, sizing.
// ===========================================================================

TEST(WorkspaceTest, MonotoneGrowthAndSizing) {
  // tiny-llama dims: E=64, H=4, Hkv=2, d=16, I=176.
  Workspace ws = Workspace::Create(64, 4, 2, 16, 176);
  EXPECT_EQ(ws.capacity_tokens(), 0);
  EXPECT_EQ(ws.bytes(), 0);

  Unwrap0(ws.EnsureCapacity(8));
  EXPECT_EQ(ws.capacity_tokens(), 8);
  // §6.2 formula: 4·[4·T·E + T·(H+2·Hkv)·d + T·H·d + 2·T·I], T=8 → 25600.
  EXPECT_EQ(ws.bytes(), 25600);

  // Growing enlarges; shrinking is a no-op (high-water mark).
  Unwrap0(ws.EnsureCapacity(16));
  EXPECT_EQ(ws.capacity_tokens(), 16);
  EXPECT_EQ(ws.bytes(), 51200);
  Unwrap0(ws.EnsureCapacity(1));
  EXPECT_EQ(ws.capacity_tokens(), 16);  // unchanged (decode after prefill)
  EXPECT_EQ(ws.bytes(), 51200);
}

// A decode step after a prefill reuses the prefill-sized workspace (no growth):
// the same OptimizedModel drives prefill (T large) then decode (T=1). We prove
// it indirectly — a prefill then a continuation forward both succeed and the
// continuation's logits match a full recompute (KV invariant already covers
// correctness; this asserts the reuse path runs).
TEST(OptimizedModelTest, DecodeAfterPrefillReusesWorkspace) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  SimpleKvCache cache = FreshCache(*opt, 256);
  // Prefill.
  (void)ForwardOnce(*opt, ids, LogitsMode::kLast, cache);
  // Decode one more token at the next position.
  const std::array<std::int32_t, 1> next{ids.front()};
  const std::array<std::int32_t, 1> pos{static_cast<std::int32_t>(ids.size())};
  const ForwardRequest req{.token_ids = next,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kLast,
                           .hook = nullptr};
  EXPECT_TRUE(opt->forward(req).ok());
}

// ===========================================================================
// Error paths: mirror ReferenceModel; cache unchanged on failure.
// ===========================================================================

TEST(OptimizedModelTest, ForwardRejectsMalformedInputs) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  SimpleKvCache cache = FreshCache(*opt);

  // Empty prompt.
  {
    const std::vector<std::int32_t> empty;
    const ForwardRequest req{.token_ids = empty,
                             .positions = empty,
                             .cache = &cache,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
    EXPECT_TRUE(IsInvalidArgument(opt->forward(req).status()));
  }
  // positions/T mismatch.
  {
    const std::vector<std::int32_t> short_pos(ids.size() - 1, 0);
    const ForwardRequest req{.token_ids = ids,
                             .positions = short_pos,
                             .cache = &cache,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
    EXPECT_TRUE(IsInvalidArgument(opt->forward(req).status()));
  }
  // Out-of-range id.
  {
    std::vector<std::int32_t> bad = ids;
    bad[0] = 1'000'000;
    const ForwardRequest req{.token_ids = bad,
                             .positions = pos,
                             .cache = &cache,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
    EXPECT_TRUE(IsInvalidArgument(opt->forward(req).status()));
  }
  // Null cache.
  {
    const ForwardRequest req{.token_ids = ids,
                             .positions = pos,
                             .cache = nullptr,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
    EXPECT_TRUE(IsInvalidArgument(opt->forward(req).status()));
  }
  // The cache was never touched by any of the rejected calls.
  EXPECT_EQ(cache.length(), 0);
}

TEST(OptimizedModelTest, OverCapacityCacheLeavesCacheUnchanged) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  // Capacity smaller than the prompt.
  SimpleKvCache cache =
      FreshCache(*opt, static_cast<std::int64_t>(ids.size()) - 1);
  const ForwardRequest req{.token_ids = ids,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kLast,
                           .hook = nullptr};
  EXPECT_TRUE(IsResourceExhausted(opt->forward(req).status()));
  EXPECT_EQ(cache.length(), 0);
}

}  // namespace
