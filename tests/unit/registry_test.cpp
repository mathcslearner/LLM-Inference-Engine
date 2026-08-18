#include "model/registry.h"

#include "common/paths.h"
#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "kvcache/simple_cache.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/reference_model.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The architecture registry & builder (M5-T08; design:
// docs/design/model-execution.md §9, §13 T08). Covers: both built-in families
// resolve; an unknown architecture is a clean `Unimplemented` listing the
// supported names; `BuildModel` produces a `Model` whose forward is
// bit-identical to `ReferenceModel::Create` (the wiring is a pass-through, no
// new numerics); the Qwen2 arch string routes to the same family builder;
// registering a test-local dummy architecture is *one call* and makes it
// buildable (the extensibility criterion); duplicate/empty registration and the
// `kOptimized` backend are rejected.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsAlreadyExists;
using engine::core::IsInvalidArgument;
using engine::core::IsUnimplemented;
using engine::core::Status;
using engine::core::StatusOr;
using engine::kvcache::CacheGeometry;
using engine::kvcache::SimpleKvCache;
using engine::model::Backend;
using engine::model::BuildModel;
using engine::model::BuildOptions;
using engine::model::ForwardRequest;
using engine::model::load_model;
using engine::model::LoadedModel;
using engine::model::LogitsMode;
using engine::model::Model;
using engine::model::ModelConfig;
using engine::model::ReferenceModel;
using engine::model::RegisterArchitecture;
using engine::model::SupportedArchitectures;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// tiny-llama dims (fixture invariants — shared with model_test).
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 16;
constexpr int kLayers = 2;
constexpr std::int64_t kMaxPos = 128;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] LoadedModel LoadTiny() {
  auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  EXPECT_TRUE(loaded.ok()) << loaded.status().ToString();
  return *std::move(loaded);
}

[[nodiscard]] CacheGeometry ModelGeometry() {
  return {.num_layers = kLayers,
          .num_kv_heads = kKvHeads,
          .head_dim = kHeadDim,
          .dtype = DataType::kFloat32};
}

// A short in-range prompt; the registry test does not need the golden ids —
// only that the built model runs and agrees with a direct build.
[[nodiscard]] std::vector<std::int32_t> Prompt() { return {1, 5, 9, 13}; }

[[nodiscard]] std::vector<std::int32_t> Iota(std::int64_t n) {
  std::vector<std::int32_t> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// Runs a kAll forward for `Prompt()` and returns the logits.
[[nodiscard]] Tensor RunForward(Model& model) {
  const std::vector<std::int32_t> ids = Prompt();
  const std::vector<std::int32_t> pos =
      Iota(static_cast<std::int64_t>(ids.size()));
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(ModelGeometry(), kMaxPos));
  const ForwardRequest req{.token_ids = ids,
                           .positions = pos,
                           .cache = &cache,
                           .logits_mode = LogitsMode::kAll,
                           .hook = nullptr};
  return Unwrap(model.forward(req));
}

// ===========================================================================
// Family coverage & unknown-architecture rejection.
// ===========================================================================

TEST(RegistryTest, SupportsBothBuiltInFamilies) {
  const std::vector<std::string> names = SupportedArchitectures();
  EXPECT_NE(std::ranges::find(names, "LlamaForCausalLM"), names.end());
  EXPECT_NE(std::ranges::find(names, "Qwen2ForCausalLM"), names.end());
  // SupportedArchitectures is sorted (std::map key order).
  EXPECT_TRUE(std::ranges::is_sorted(names));
}

TEST(RegistryTest, UnknownArchitectureIsUnimplementedAndListsSupported) {
  LoadedModel loaded = LoadTiny();
  loaded.config.architecture_name = "FooForCausalLM";
  const auto built = BuildModel(std::move(loaded));
  ASSERT_FALSE(built.ok());
  EXPECT_TRUE(IsUnimplemented(built.status()));
  const std::string msg = built.status().ToString();
  EXPECT_NE(msg.find("FooForCausalLM"), std::string::npos);
  EXPECT_NE(msg.find("LlamaForCausalLM"), std::string::npos);
  EXPECT_NE(msg.find("Qwen2ForCausalLM"), std::string::npos);
}

// ===========================================================================
// BuildModel is a pass-through to ReferenceModel::Create — bit-identical.
// ===========================================================================

TEST(RegistryTest, LlamaBuildMatchesDirectReferenceModel) {
  // Built through the registry (config.architecture_name == LlamaForCausalLM).
  const std::unique_ptr<Model> via_registry = Unwrap(BuildModel(LoadTiny()));
  ASSERT_EQ(via_registry->config().architecture_name, "LlamaForCausalLM");
  ASSERT_EQ(via_registry->cache_geometry().num_layers, kLayers);
  const Tensor a = RunForward(*via_registry);

  // Built directly, same weights, same math.
  const std::unique_ptr<ReferenceModel> direct =
      Unwrap(ReferenceModel::Create(LoadTiny()));
  const Tensor b = RunForward(*direct);

  // Bit-identical: zero tolerance (the registry only chooses a builder).
  const ops::AllCloseResult r = Unwrap(ops::allclose(a, b, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
  EXPECT_EQ(r.max_abs_diff, 0.0);
}

// The Qwen2 arch string routes to the same family builder. This uses tiny-llama
// relabelled (bias-free binding) to isolate the *routing* decision; the real
// Qwen fixture — biases, decoupled head_dim, tied lm_head, golden logits — is
// exercised end-to-end in qwen2_family_test.cpp (M5-T10). Here we only assert
// the string resolves and builds — logits equal the Llama-labelled build.
TEST(RegistryTest, Qwen2ArchStringRoutesToFamilyBuilder) {
  LoadedModel loaded = LoadTiny();
  loaded.config.architecture_name = "Qwen2ForCausalLM";
  loaded.config.attention_bias = false;  // tiny-llama carries no q/k/v biases
  const std::unique_ptr<Model> qwen = Unwrap(BuildModel(std::move(loaded)));
  EXPECT_EQ(qwen->config().architecture_name, "Qwen2ForCausalLM");
  const Tensor a = RunForward(*qwen);

  const std::unique_ptr<Model> llama = Unwrap(BuildModel(LoadTiny()));
  const Tensor b = RunForward(*llama);
  const ops::AllCloseResult r = Unwrap(ops::allclose(a, b, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// ===========================================================================
// The optimized backend is not available in M5.
// ===========================================================================

TEST(RegistryTest, OptimizedBackendIsUnimplemented) {
  const BuildOptions opts{.backend = Backend::kOptimized};
  const auto built = BuildModel(LoadTiny(), opts);
  ASSERT_FALSE(built.ok());
  EXPECT_TRUE(IsUnimplemented(built.status()));
}

// ===========================================================================
// Extensibility: adding an architecture is one RegisterArchitecture call.
// ===========================================================================

// A trivial `Model` returning constant logits — enough to prove a test-local
// architecture builds and runs without touching the reference family.
class DummyModel final : public Model {
 public:
  explicit DummyModel(ModelConfig config) : config_(std::move(config)) {}

  [[nodiscard]] StatusOr<Tensor> forward(
      const ForwardRequest& request) override {
    const auto t = static_cast<std::int64_t>(request.token_ids.size());
    const std::int64_t rows = request.logits_mode == LogitsMode::kLast ? 1 : t;
    return ops::zeros(Shape{rows, config_.vocab_size}, DataType::kFloat32);
  }
  [[nodiscard]] const ModelConfig& config() const override { return config_; }
  [[nodiscard]] CacheGeometry cache_geometry() const override {
    return {.num_layers = config_.num_layers,
            .num_kv_heads = config_.num_kv_heads,
            .head_dim = config_.head_dim,
            .dtype = DataType::kFloat32};
  }

 private:
  ModelConfig config_;
};

TEST(RegistryTest, RegisteringDummyArchitectureIsOneCall) {
  constexpr std::string_view kDummy = "DummyForCausalLM";

  const Status reg = RegisterArchitecture(
      kDummy,
      [](LoadedModel loaded,
         const BuildOptions& /*options*/) -> StatusOr<std::unique_ptr<Model>> {
        return std::unique_ptr<Model>(
            std::make_unique<DummyModel>(std::move(loaded.config)));
      });
  ASSERT_TRUE(reg.ok()) << reg.ToString();

  // Now visible in the supported set...
  const std::vector<std::string> names = SupportedArchitectures();
  EXPECT_NE(std::ranges::find(names, std::string(kDummy)), names.end());

  // ...and buildable via the same BuildModel entry point.
  LoadedModel loaded = LoadTiny();
  loaded.config.architecture_name = std::string(kDummy);
  const std::unique_ptr<Model> model = Unwrap(BuildModel(std::move(loaded)));
  const Tensor logits = RunForward(*model);
  EXPECT_EQ(logits.shape().dim(1), model->config().vocab_size);

  // Duplicate registration is rejected (not a silent overwrite).
  const Status dup = RegisterArchitecture(
      kDummy,
      [](const LoadedModel&,
         const BuildOptions&) -> StatusOr<std::unique_ptr<Model>> {
        return engine::core::InternalError("unused");
      });
  EXPECT_TRUE(IsAlreadyExists(dup)) << dup.ToString();
}

TEST(RegistryTest, EmptyArchitectureNameIsRejected) {
  const Status s = RegisterArchitecture(
      "",
      [](const LoadedModel&,
         const BuildOptions&) -> StatusOr<std::unique_ptr<Model>> {
        return engine::core::InternalError("unused");
      });
  EXPECT_TRUE(IsInvalidArgument(s)) << s.ToString();
}

}  // namespace
