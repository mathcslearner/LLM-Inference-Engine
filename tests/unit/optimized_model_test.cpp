#include "model/optimized_model.h"

#include "common/paths.h"
#include "core/status.h"
#include "engine/generator.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_cache.h"
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
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::kvcache::KvCache;
using engine::kvcache::PagedKvCache;
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

// A block pool sized for `model`'s geometry (bs=8 so tiny-fixture prompts of
// ~8-12 tokens straddle several block boundaries; 8 blocks = 64 token slots).
[[nodiscard]] BlockPool FreshPool(const Model& model, int block_size = 8,
                                  std::int64_t num_blocks = 8) {
  return Unwrap(BlockPool::Create(model.cache_geometry(), block_size,
                                  num_blocks, nullptr));
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
          Unwrap(Generate(*opt, co, c.prompt_ids, options)).tokens;
      const std::vector<std::int32_t> got_ref =
          Unwrap(Generate(*ref, cr, c.prompt_ids, options)).tokens;
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
      Unwrap(Generate(*opt, ca, c.prompt_ids, options)).tokens;
  const std::vector<std::int32_t> b =
      Unwrap(Generate(*opt, cb, c.prompt_ids, options)).tokens;
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
// Paged KV cache (M8-T06): the optimized backend drives a PagedKvCache through
// the unchanged KvCache interface — append + view (the contiguous gather) feed
// the same M6 prefill/decode kernels. These are the "prefill with existing
// paged cache matches the reference backend" acceptance tests.
// ===========================================================================

// Prefill-continuation (P>0): a first prefill chunk fills the paged cache, then
// a second chunk continues from it. The paged run must be bit-exact to the same
// two-chunk run on SimpleKvCache (same K/V bytes into the same kernels), and
// within tolerance of a one-shot reference-backend full prefill.
TEST(OptimizedModelTest, PagedPrefillContinuationMatchesSimpleAndReference) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    std::unique_ptr<Model> ref = Reference(fixture);
    const std::vector<std::int32_t> ids = PromptIds(fixture);
    const auto t = static_cast<std::int64_t>(ids.size());
    ASSERT_GE(t, 4);
    const std::int64_t split = t / 2;

    // Two-chunk continuation on a paged cache and on a simple cache.
    const auto run_two_chunks = [&](Model& model, KvCache& cache) {
      {
        const std::span<const std::int32_t> a(ids.data(),
                                              static_cast<std::size_t>(split));
        const std::vector<std::int32_t> pa = Iota(split);
        const ForwardRequest req{.token_ids = a,
                                 .positions = pa,
                                 .cache = &cache,
                                 .logits_mode = LogitsMode::kLast,
                                 .hook = nullptr};
        EXPECT_TRUE(model.forward(req).ok());
      }
      const std::span<const std::int32_t> b(
          ids.data() + split, static_cast<std::size_t>(t - split));
      std::vector<std::int32_t> pb(static_cast<std::size_t>(t - split));
      std::iota(pb.begin(), pb.end(), static_cast<std::int32_t>(split));
      const ForwardRequest req{.token_ids = b,
                               .positions = pb,
                               .cache = &cache,
                               .logits_mode = LogitsMode::kLast,
                               .hook = nullptr};
      return Unwrap(model.forward(req));
    };

    BlockPool pool = FreshPool(*opt);
    PagedKvCache paged(&pool);
    const Tensor logits_paged = run_two_chunks(*opt, paged);
    EXPECT_EQ(paged.length(), t);

    SimpleKvCache simple = FreshCache(*opt);
    const Tensor logits_simple = run_two_chunks(*opt, simple);

    // Bit-exact vs the same run on SimpleKvCache: identical K/V bytes into
    // identical kernels (the gather reproduces the contiguous layout).
    const ops::AllCloseResult rs =
        Unwrap(ops::allclose(logits_paged, logits_simple, 0.0, 0.0));
    EXPECT_TRUE(rs.allclose) << rs.Summary();

    // Within tolerance vs a one-shot full prefill on the reference backend.
    SimpleKvCache ref_cache = FreshCache(*ref);
    const Tensor logits_ref =
        ForwardOnce(*ref, ids, LogitsMode::kLast, ref_cache);
    const ops::AllCloseResult rr =
        Unwrap(ops::allclose(logits_paged, logits_ref, kRtol, kAtol));
    EXPECT_TRUE(rr.allclose) << rr.Summary();
    std::cerr << "[paged] " << fixture
              << " continuation vs simple max_abs_diff=" << rs.max_abs_diff
              << " vs reference max_abs_diff=" << rr.max_abs_diff << "\n";
  }
}

// The KV invariant through the paged cache: token-by-token decode reproduces a
// one-shot full prefill, bit-exact (masked keys contribute exactly 0).
TEST(OptimizedModelTest, PagedKvInvariantTokenByTokenMatchesFullPrefill) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    const std::vector<std::int32_t> ids = PromptIds(fixture);
    const auto t = static_cast<std::int64_t>(ids.size());

    BlockPool full_pool = FreshPool(*opt);
    PagedKvCache full(&full_pool);
    const Tensor logits_full = ForwardOnce(*opt, ids, LogitsMode::kLast, full);
    EXPECT_EQ(full.length(), t);

    BlockPool step_pool = FreshPool(*opt);
    PagedKvCache step(&step_pool);
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
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(logits_step, logits_full, 0.0, 0.0));
    EXPECT_TRUE(r.allclose) << r.Summary();
  }
}

// A paged cache drives the greedy loop identically to a simple cache (a cheap
// pre-check of the M8-T07 integration; the simple-cache path already matches
// generate.json, so transitively the paged path does too).
TEST(OptimizedModelTest, PagedGreedyMatchesSimpleCache) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    const std::vector<std::int32_t> prompt = PromptIds(fixture);
    const GenerateOptions options{.sampling = SamplingParams::Greedy(8),
                                  .eos_ids = {}};

    SimpleKvCache simple = FreshCache(*opt, 256);
    const std::vector<std::int32_t> want =
        Unwrap(Generate(*opt, simple, prompt, options)).tokens;

    BlockPool pool = FreshPool(*opt, /*block_size=*/8, /*num_blocks=*/16);
    PagedKvCache paged(&pool);
    const std::vector<std::int32_t> got =
        Unwrap(Generate(*opt, paged, prompt, options)).tokens;
    EXPECT_EQ(got, want);
  }
}

// A single decode step through the zero-copy `paged_view` fast path (M8-T07)
// is bit-exact to the same decode step through the contiguous `view()` path a
// SimpleKvCache takes: prefill both caches with the identical prompt, then one
// decode forward each, and the logits match to the bit. This locks down the
// specific fast-path swap in OptimizedModel::ForwardLayer (paged-kv-cache.md
// §8.3) — PagedDecodeAttentionF32 == DecodeAttentionF32 on the same K/V.
TEST(OptimizedModelTest, PagedDecodeStepMatchesSimpleDecodeStep) {
  for (const char* fixture : {kLlama, kQwen}) {
    SCOPED_TRACE(fixture);
    std::unique_ptr<Model> opt = Optimized(fixture);
    const std::vector<std::int32_t> ids = PromptIds(fixture);

    const auto decode_after_prefill = [&](KvCache& cache) {
      (void)ForwardOnce(*opt, ids, LogitsMode::kLast, cache);  // prefill P
      const std::array<std::int32_t, 1> one{ids.front()};
      const std::array<std::int32_t, 1> pos{
          static_cast<std::int32_t>(ids.size())};
      const ForwardRequest req{.token_ids = one,
                               .positions = pos,
                               .cache = &cache,
                               .logits_mode = LogitsMode::kLast,
                               .hook = nullptr};
      return Unwrap(opt->forward(req));  // decode T=1
    };

    BlockPool pool = FreshPool(*opt);
    PagedKvCache paged(&pool);
    const Tensor logits_paged = decode_after_prefill(paged);

    SimpleKvCache simple = FreshCache(*opt, 256);
    const Tensor logits_simple = decode_after_prefill(simple);

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(logits_paged, logits_simple, 0.0, 0.0));
    EXPECT_TRUE(r.allclose) << r.Summary();
  }
}

// A KvCache decorator whose `paged_view` fails with a non-Unimplemented status.
// The decode fast path must PROPAGATE such an error (only `Unimplemented`
// triggers the `view()` fallback), so the forward returns it rather than
// silently masking a real cache failure (paged-kv-cache.md §8.3).
class PagedViewErrorCache final : public KvCache {
 public:
  explicit PagedViewErrorCache(PagedKvCache* inner) : inner_(inner) {}

  [[nodiscard]] CacheGeometry geometry() const override {
    return inner_->geometry();
  }
  [[nodiscard]] std::int64_t length() const override {
    return inner_->length();
  }
  [[nodiscard]] std::int64_t capacity() const override {
    return inner_->capacity();
  }
  [[nodiscard]] engine::core::Status append(int layer, const Tensor& k,
                                            const Tensor& v) override {
    return inner_->append(layer, k, v);
  }
  [[nodiscard]] StatusOr<engine::kvcache::KvView> view(
      int layer) const override {
    return inner_->view(layer);
  }
  [[nodiscard]] StatusOr<engine::kvcache::PagedKvView> paged_view(
      int /*layer*/) const override {
    return engine::core::InternalError("paged_view: injected failure");
  }
  [[nodiscard]] engine::core::Status truncate(
      std::int64_t new_length) override {
    return inner_->truncate(new_length);
  }

 private:
  PagedKvCache* inner_;
};

TEST(OptimizedModelTest, PagedViewErrorPropagatesNotFallback) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  BlockPool pool = FreshPool(*opt);
  PagedKvCache paged(&pool);
  PagedViewErrorCache wrapper(&paged);

  // Prefill (T>1) uses view(), which the wrapper delegates cleanly.
  (void)ForwardOnce(*opt, ids, LogitsMode::kLast, wrapper);

  // Decode (T=1) hits paged_view → injected Internal → must propagate.
  const std::array<std::int32_t, 1> one{ids.front()};
  const std::array<std::int32_t, 1> pos{static_cast<std::int32_t>(ids.size())};
  const ForwardRequest req{.token_ids = one,
                           .positions = pos,
                           .cache = &wrapper,
                           .logits_mode = LogitsMode::kLast,
                           .hook = nullptr};
  const StatusOr<Tensor> out = opt->forward(req);
  EXPECT_FALSE(out.ok());
  EXPECT_TRUE(engine::core::IsInternal(out.status()))
      << out.status().ToString();
}

// weight_resident_bytes (M8-T07 §5.3): the fractional KV-budget's weight term.
// Positive, and tied embeddings are counted once (the Qwen fixture shares
// lm_head/embed storage), so the deduplicated figure is below the naive sum of
// every map entry's bytes.
TEST(OptimizedModelTest, WeightResidentBytesDeduplicatesTiedEmbedding) {
  const LoadedModel qwen = Load(kQwen);  // tied embeddings
  const std::int64_t deduped = engine::model::weight_resident_bytes(qwen);
  EXPECT_GT(deduped, 0);

  std::int64_t naive_sum = 0;
  for (const auto& [name, w] : qwen.weights) {
    naive_sum += w.numel() * engine::tensor::itemsize(w.dtype());
  }
  // The tied embedding table appears under two names; dedup drops one copy.
  EXPECT_LT(deduped, naive_sum);
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

// BytesFor (M8-T07 §5.3): the static estimate matches an instantiated
// workspace's bytes() at the same T — the KV-budget resolver relies on it to
// size the workspace term without building a workspace.
TEST(WorkspaceTest, BytesForMatchesInstantiated) {
  Workspace ws = Workspace::Create(64, 4, 2, 16, 176);
  Unwrap0(ws.EnsureCapacity(8));
  EXPECT_EQ(Workspace::BytesFor(64, 4, 2, 16, 176, 8), ws.bytes());
  EXPECT_EQ(Workspace::BytesFor(64, 4, 2, 16, 176, 8), 25600);
  EXPECT_EQ(Workspace::BytesFor(64, 4, 2, 16, 176, 0), 0);
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

// The ragged/batched forward (cu_seqlens + per-sequence caches) is rejected
// with Unimplemented until M9-T07, matching ReferenceModel (M9-T05;
// scheduler-runtime.md §8.1).
TEST(OptimizedModelTest, ForwardRejectsBatchedRequest) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> ids = PromptIds(kLlama);
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  SimpleKvCache cache = FreshCache(*opt);
  const std::vector<std::int32_t> cu = {0,
                                        static_cast<std::int32_t>(ids.size())};
  EXPECT_TRUE(IsUnimplemented(opt->forward({.token_ids = ids,
                                            .positions = pos,
                                            .cache = &cache,
                                            .cu_seqlens = cu})
                                  .status()));
  std::vector<KvCache*> caches = {&cache};
  EXPECT_TRUE(IsUnimplemented(opt->forward({.token_ids = ids,
                                            .positions = pos,
                                            .cache = &cache,
                                            .caches = caches})
                                  .status()));
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

// A hook that drains `count` tokens' worth of blocks from a shared pool via a
// competitor cache, exactly once, when the named activation event fires. Used
// to trip the pool mid-forward — after the model's capacity check, before the
// layer-0 append allocates.
class PoolDrainingHook final : public ActivationHook {
 public:
  PoolDrainingHook(PagedKvCache* hog, std::int64_t count, std::string on_event)
      : hog_(hog), count_(count), on_event_(std::move(on_event)) {}

  void on_activation(const ActivationEvent& event) override {
    if (drained_ || event.name != on_event_) {
      return;
    }
    drained_ = true;
    const CacheGeometry geom = hog_->geometry();
    for (int layer = 0; layer < geom.num_layers; ++layer) {
      const Tensor k = Unwrap(ops::zeros(
          Shape{count_, geom.num_kv_heads, geom.head_dim}, DataType::kFloat32));
      const Tensor v = Unwrap(ops::zeros(
          Shape{count_, geom.num_kv_heads, geom.head_dim}, DataType::kFloat32));
      EXPECT_TRUE(hog_->append(layer, k, v).ok());
    }
  }

 private:
  PagedKvCache* hog_;
  std::int64_t count_;
  std::string on_event_;
  bool drained_ = false;
};

// The §10.2 chain end-to-end from inside a forward: a decode step passes the
// model's capacity check, but the block pool is drained (by the hook) before
// its layer-0 append allocates the crossing block, so BlockPool::Allocate →
// ResourceExhausted propagates out of `forward`. The committed cache is
// untouched, the pool records the exhaustion, and once the competitor releases
// its block the same forward succeeds (M8-T08 recovery).
TEST(OptimizedModelTest, PagedForwardExhaustionFromPoolPropagatesAndRecovers) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  constexpr int kBs = 8;
  BlockPool pool = FreshPool(*opt, /*block_size=*/kBs, /*num_blocks=*/2);
  PagedKvCache cache(&pool);
  PagedKvCache hog(&pool);

  // Prefill exactly one block (kBs tokens) — no hook, no crossing.
  const std::vector<std::int32_t> prefill = Iota(kBs);
  (void)ForwardOnce(*opt, prefill, LogitsMode::kLast, cache);
  ASSERT_EQ(cache.length(), kBs);
  ASSERT_EQ(pool.free_blocks(), 1);  // one block left for the crossing

  // Decode step at position kBs needs a new block. Its capacity check passes
  // (1 free block), but the hook drains that block on the `embeddings` event
  // before the layer-0 append allocates it.
  PoolDrainingHook drain(&hog, /*count=*/kBs, /*on_event=*/"embeddings");
  const std::array<std::int32_t, 1> one{prefill.front()};
  const std::array<std::int32_t, 1> at{static_cast<std::int32_t>(kBs)};
  const ForwardRequest req{.token_ids = one,
                           .positions = at,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kLast,
                           .hook = &drain};
  const StatusOr<Tensor> failed = opt->forward(req);
  ASSERT_FALSE(failed.ok());
  EXPECT_TRUE(IsResourceExhausted(failed.status()))
      << failed.status().ToString();
  EXPECT_EQ(cache.length(), kBs);          // committed prefix untouched
  EXPECT_GE(pool.stats().exhaustions, 1);  // the pool Allocate path fired

  // The competitor releases its block; the identical forward now succeeds
  // (no hook this time).
  hog.reset();
  ASSERT_EQ(pool.free_blocks(), 1);
  const ForwardRequest retry{.token_ids = one,
                             .positions = at,
                             .cache = &cache,
                             .logits_mode = LogitsMode::kLast,
                             .hook = nullptr};
  EXPECT_TRUE(opt->forward(retry).ok());
  EXPECT_EQ(cache.length(), kBs + 1);
}

// Greedy generation on the optimized backend (the `engine generate` default)
// over a shared pool drained mid-run: the run fails gracefully with
// ResourceExhausted, and after both caches drop every block is reclaimed
// (pool stats zero — the M8-T08 no-leak acceptance, on the optimized path and,
// via SCALAR_PASS, under forced-scalar).
TEST(OptimizedModelTest, PagedGreedyExhaustionReclaimsAllBlocks) {
  std::unique_ptr<Model> opt = Optimized(kLlama);
  const std::vector<std::int32_t> prompt = PromptIds(kLlama);
  constexpr int kBs = 8;
  constexpr std::int64_t kMax = 32;
  const GenerateOptions options{.sampling = SamplingParams::Greedy(kMax),
                                .eos_ids = {}};

  // Pool sized so the up-front worst-case check passes (num_blocks·bs >=
  // prompt + kMax - 1) but the free blocks are drained mid-run.
  const auto blocks =
      static_cast<std::int64_t>(prompt.size()) + kMax;  // generous, /bs rounded
  BlockPool pool = FreshPool(*opt, kBs, ((blocks + kBs - 1) / kBs) + 1);
  {
    PagedKvCache cache(&pool);
    PagedKvCache hog(&pool);
    bool drained = false;
    const auto on_token = [&](const engine::engine::TokenEvent& /*ev*/) {
      if (!drained) {
        drained = true;
        // Consume every remaining free block so the sequence cannot grow.
        const std::int64_t free_now = pool.free_blocks();
        if (free_now > 0) {
          const CacheGeometry geom = hog.geometry();
          const std::int64_t count = free_now * kBs;
          for (int layer = 0; layer < geom.num_layers; ++layer) {
            const Tensor k = Unwrap(
                ops::zeros(Shape{count, geom.num_kv_heads, geom.head_dim},
                           DataType::kFloat32));
            const Tensor v = Unwrap(
                ops::zeros(Shape{count, geom.num_kv_heads, geom.head_dim},
                           DataType::kFloat32));
            ASSERT_TRUE(hog.append(layer, k, v).ok());
          }
        }
      }
    };
    const auto got = Generate(*opt, cache, prompt, options, on_token);
    ASSERT_FALSE(got.ok());
    EXPECT_TRUE(IsResourceExhausted(got.status())) << got.status().ToString();
  }
  // Both caches dropped → RAII reclaims every block.
  const engine::kvcache::BlockPoolStats st = pool.stats();
  EXPECT_EQ(st.used, 0);  // no leaked blocks
  EXPECT_EQ(st.free, st.total);
}

}  // namespace
