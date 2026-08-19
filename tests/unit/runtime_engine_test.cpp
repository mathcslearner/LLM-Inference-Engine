#include "core/status.h"
#include "engine/generator.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/simple_cache.h"
#include "model/config.h"
#include "model/model.h"
#include "runtime/channel.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "sampling/params.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// Engine API & request queue tests (M9-T03; design:
// docs/design/scheduler-runtime.md §5, §13). A `CannedModel` (deterministic
// canned logits) drives the async surface: `Step()`-driven determinism,
// per-request output equal to a standalone `engine::Generate`, FCFS admission
// under batch/token/block caps, cancellation in every state, per-request
// failure isolation, and a multi-threaded stress test (many submitters, random
// cancels, concurrent consumers) with no deadlock, every channel closed, and no
// block leaks. Pure orchestration over a mock model — no kernel, no forced
// scalar pass, so no SCALAR_PASS. `runtime` label.

namespace {

using engine::core::IsFailedPrecondition;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::Status;
using engine::core::StatusOr;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::kvcache::SimpleKvCache;
using engine::model::ForwardRequest;
using engine::model::LogitsMode;
using engine::model::Model;
using engine::model::ModelConfig;
using engine::runtime::Engine;
using engine::runtime::EngineConfig;
using engine::runtime::FinishInfo;
using engine::runtime::OutputItem;
using engine::runtime::Request;
using engine::runtime::RequestHandle;
using engine::runtime::SeqState;
using engine::sampling::SamplingParams;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;
using FinishReason = engine::engine::FinishReason;

namespace ops = engine::tensor::ops;

constexpr int kLayers = 2;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 16;
constexpr std::int64_t kVocab = 16;
constexpr int kBs = 8;
constexpr std::int32_t kEos = static_cast<std::int32_t>(kVocab) - 1;  // 15

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

template <typename T>
[[nodiscard]] T Require(std::optional<T> opt) {
  if (!opt.has_value()) {
    ADD_FAILURE() << "expected an engaged optional";
    return T{};
  }
  return std::move(opt).value();
}

// A deterministic mock model. Its forward appends real (zeroed) K/V to the
// cache — so block allocation/exhaustion is genuine — and returns a one-hot
// logits row whose argmax is `(last_token + 1) % V`. Greedy generation from a
// prompt ending in `p` therefore produces `p+1, p+2, …`, hitting the EOS id
// `V-1` deterministically. A prompt whose final id is `poison_` instead yields
// an all-NaN row, so the greedy argmax fails — the per-request fault path.
class CannedModel final : public Model {
 public:
  CannedModel() {
    config_.architecture_name = "Canned";
    config_.num_layers = kLayers;
    config_.num_heads = kKvHeads;
    config_.num_kv_heads = kKvHeads;
    config_.head_dim = kHeadDim;
    config_.vocab_size = kVocab;
    config_.max_position_embeddings = 4096;
  }

  void set_poison_token(std::int32_t token) { poison_ = token; }

  [[nodiscard]] StatusOr<Tensor> forward(
      const ForwardRequest& request) override {
    const auto t = static_cast<std::int64_t>(request.token_ids.size());
    // Append this call's K/V, layer by layer, exactly as a real model does.
    for (int layer = 0; layer < kLayers; ++layer) {
      const Tensor k =
          Unwrap(ops::zeros(Shape{t, kKvHeads, kHeadDim}, DataType::kFloat32));
      const Tensor v =
          Unwrap(ops::zeros(Shape{t, kKvHeads, kHeadDim}, DataType::kFloat32));
      if (Status s = request.cache->append(layer, k, v); !s.ok()) {
        return s;
      }
    }
    const std::int64_t rows = request.logits_mode == LogitsMode::kLast ? 1 : t;
    Tensor logits = Unwrap(ops::zeros(Shape{rows, kVocab}, DataType::kFloat32));
    const std::int32_t last = request.token_ids.back();
    auto* base = logits.data_ptr<float>();
    for (std::int64_t r = 0; r < rows; ++r) {
      float* row = base + (r * kVocab);
      if (poison_ >= 0 && last == poison_) {
        for (std::int64_t i = 0; i < kVocab; ++i) {
          row[i] = std::numeric_limits<float>::quiet_NaN();
        }
      } else {
        const std::int32_t argmax =
            (last + 1) % static_cast<std::int32_t>(kVocab);
        row[argmax] = 1.0F;
      }
    }
    return logits;
  }

  [[nodiscard]] const ModelConfig& config() const override { return config_; }
  [[nodiscard]] CacheGeometry cache_geometry() const override {
    return {.num_layers = kLayers,
            .num_kv_heads = kKvHeads,
            .head_dim = kHeadDim,
            .dtype = DataType::kFloat32};
  }

 private:
  ModelConfig config_;
  std::int32_t poison_ = -1;
};

[[nodiscard]] CacheGeometry TinyGeom() {
  return CacheGeometry{.num_layers = kLayers,
                       .num_kv_heads = kKvHeads,
                       .head_dim = kHeadDim,
                       .dtype = DataType::kFloat32};
}

[[nodiscard]] BlockPool MakePool(std::int64_t num_blocks = 64) {
  return Unwrap(BlockPool::Create(TinyGeom(), kBs, num_blocks, nullptr));
}

[[nodiscard]] Request GreedyRequest(std::vector<std::int32_t> prompt,
                                    std::int64_t max_tokens = 32) {
  Request req;
  req.prompt_ids = std::move(prompt);
  req.params = SamplingParams::Greedy(max_tokens);
  req.eos_ids = {kEos};
  return req;
}

// A request with no EOS set, so it runs to `max_tokens` — a long-lived
// sequence for the Stop/destructor teardown tests.
[[nodiscard]] Request NoEosRequest(std::vector<std::int32_t> prompt,
                                   std::int64_t max_tokens) {
  Request req = GreedyRequest(std::move(prompt), max_tokens);
  req.eos_ids = {};
  return req;
}

// The greedy trajectory a standalone `Generate` produces for `prompt` — the
// output-equivalence oracle.
[[nodiscard]] std::vector<std::int32_t> ReferenceTokens(
    CannedModel& model, const std::vector<std::int32_t>& prompt,
    std::int64_t max_tokens = 32) {
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(TinyGeom(), 4096));
  engine::engine::GenerateOptions opts;
  opts.sampling = SamplingParams::Greedy(max_tokens);
  opts.eos_ids = {kEos};
  auto result = Unwrap(engine::engine::Generate(model, cache, prompt, opts));
  return result.tokens;
}

// Drive an engine's `Step()` to quiescence (single-threaded, no loop thread).
void RunToIdle(Engine& engine) {
  while (engine.Step()) {
  }
}

// Drain all tokens delivered on a handle whose channel is already closed
// (single-thread tests drain after `RunToIdle`).
[[nodiscard]] std::vector<std::int32_t> DrainTokens(RequestHandle& handle) {
  std::vector<std::int32_t> tokens;
  while (auto item = handle.try_next_token()) {
    tokens.push_back(item->token_id);
  }
  return tokens;
}

// --- Create / config validation --------------------------------------------

TEST(EngineTest, CreateRejectsBadConfig) {
  CannedModel model;
  BlockPool pool = MakePool();
  EXPECT_TRUE(IsInvalidArgument(
      Engine::Create(model, pool, EngineConfig{.max_num_seqs = 0}).status()));
  EXPECT_TRUE(IsInvalidArgument(
      Engine::Create(model, pool, EngineConfig{.max_num_batched_tokens = 0})
          .status()));
}

TEST(EngineTest, CreateRejectsGeometryMismatch) {
  CannedModel model;
  // A pool with a different head_dim than the model expects.
  CacheGeometry g = TinyGeom();
  g.head_dim = 8;
  BlockPool pool = Unwrap(BlockPool::Create(g, kBs, 8, nullptr));
  EXPECT_TRUE(
      IsInvalidArgument(Engine::Create(model, pool, EngineConfig{}).status()));
}

TEST(EngineTest, CreateSucceeds) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  ASSERT_NE(engine, nullptr);
}

// --- Submit validation ------------------------------------------------------

TEST(EngineTest, SubmitRejectsEmptyPrompt) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  EXPECT_TRUE(IsInvalidArgument(engine->Submit(GreedyRequest({})).status()));
  EXPECT_EQ(engine->num_live(), 0U);
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(EngineTest, SubmitRejectsPromptOverTokenBudget) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(
      Engine::Create(model, pool, EngineConfig{.max_num_batched_tokens = 4}));
  EXPECT_TRUE(IsInvalidArgument(
      engine->Submit(GreedyRequest({1, 2, 3, 4, 5})).status()));
}

TEST(EngineTest, SubmitRejectsPromptOverPoolCapacity) {
  CannedModel model;
  // Pool holds 2 blocks × 8 = 16 tokens; a peak beyond that is rejected.
  BlockPool pool = MakePool(/*num_blocks=*/2);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));
  EXPECT_TRUE(IsResourceExhausted(
      engine->Submit(GreedyRequest({1, 2, 3}, /*max_tokens=*/100)).status()));
}

TEST(EngineTest, SubmitRejectsInvalidParams) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  Request req = GreedyRequest({1, 2, 3});
  req.params.temperature = -1.0F;  // fails ValidateSamplingParams
  EXPECT_TRUE(IsInvalidArgument(engine->Submit(std::move(req)).status()));
}

TEST(EngineTest, SubmitAssignsMonotonicIds) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  auto h1 = Unwrap(engine->Submit(GreedyRequest({1, 2})));
  auto h2 = Unwrap(engine->Submit(GreedyRequest({3, 4})));
  EXPECT_LT(h1.id(), h2.id());
}

// --- Step-driven determinism & output equivalence --------------------------

TEST(EngineTest, IdleStepReturnsFalse) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  EXPECT_FALSE(engine->Step());
}

TEST(EngineTest, SingleRequestMatchesGenerate) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));

  const std::vector<std::int32_t> prompt = {2, 5, 3};
  auto handle = Unwrap(engine->Submit(GreedyRequest(prompt)));
  RunToIdle(*engine);

  const std::vector<std::int32_t> expected = ReferenceTokens(model, prompt);
  EXPECT_EQ(DrainTokens(handle), expected);

  const FinishInfo info = handle.await_completion();
  EXPECT_EQ(info.terminal, SeqState::kFinished);
  EXPECT_EQ(info.finish_reason, FinishReason::kStop);
  EXPECT_EQ(engine->num_live(), 0U);
  EXPECT_EQ(pool.stats().used, 0);  // no leaks
}

TEST(EngineTest, MaxTokensFinishReasonLength) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));

  // Prompt ending in 2 would run to EOS in 13 tokens; cap at 4 → kLength.
  auto handle =
      Unwrap(engine->Submit(GreedyRequest({0, 1, 2}, /*max_tokens=*/4)));
  RunToIdle(*engine);

  auto tokens = DrainTokens(handle);
  EXPECT_EQ(tokens.size(), 4U);
  EXPECT_EQ(handle.await_completion().finish_reason, FinishReason::kLength);
}

TEST(EngineTest, OneTokenPerStep) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  auto handle = Unwrap(engine->Submit(GreedyRequest({3}, /*max_tokens=*/5)));

  // Each Step delivers exactly one token for the single sequence.
  EXPECT_TRUE(engine->Step());  // prefill → token 0
  EXPECT_EQ(handle.try_next_token().has_value(), true);
  EXPECT_FALSE(handle.try_next_token().has_value());
  EXPECT_TRUE(engine->Step());                              // decode → token 1
  EXPECT_EQ(Require(handle.try_next_token()).token_id, 5);  // 3→4→5
}

TEST(EngineTest, LogprobsFlowThrough) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  Request req = GreedyRequest({1, 2}, /*max_tokens=*/3);
  req.params.logprobs = 3;
  auto handle = Unwrap(engine->Submit(std::move(req)));
  RunToIdle(*engine);

  const auto item = Require(handle.try_next_token());
  const auto lp = Require(item.logprobs);
  EXPECT_FALSE(lp.top.empty());
}

// --- FCFS admission ---------------------------------------------------------

TEST(EngineTest, MaxNumSeqsBoundsBatchWidth) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine =
      Unwrap(Engine::Create(model, pool, EngineConfig{.max_num_seqs = 1}));

  auto h1 = Unwrap(engine->Submit(GreedyRequest({2}, /*max_tokens=*/3)));
  auto h2 = Unwrap(engine->Submit(GreedyRequest({4}, /*max_tokens=*/3)));

  EXPECT_TRUE(engine->Step());  // only the first is admitted
  EXPECT_EQ(engine->num_running(), 1U);
  EXPECT_EQ(engine->num_waiting(), 1U);

  RunToIdle(*engine);
  // Both still complete correctly, in FCFS order.
  EXPECT_EQ(DrainTokens(h1), ReferenceTokens(model, {2}, 3));
  EXPECT_EQ(DrainTokens(h2), ReferenceTokens(model, {4}, 3));
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(EngineTest, AdmissionWaitsOnBlockAvailability) {
  CannedModel model;
  // 3 blocks × 8 = 24 token-slots. Two 8-token prompts need 1 block each for
  // prefill, leaving room; a third cannot be admitted until blocks free.
  BlockPool pool = MakePool(/*num_blocks=*/3);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));

  auto h1 = Unwrap(engine->Submit(GreedyRequest({1, 2, 3}, /*max_tokens=*/2)));
  auto h2 = Unwrap(engine->Submit(GreedyRequest({1, 2, 3}, /*max_tokens=*/2)));
  auto h3 = Unwrap(engine->Submit(GreedyRequest({1, 2, 3}, /*max_tokens=*/2)));
  RunToIdle(*engine);

  // All three eventually complete; no leaks.
  EXPECT_FALSE(DrainTokens(h1).empty());
  EXPECT_FALSE(DrainTokens(h2).empty());
  EXPECT_FALSE(DrainTokens(h3).empty());
  EXPECT_EQ(pool.stats().used, 0);
}

// --- Cancellation -----------------------------------------------------------

TEST(EngineTest, CancelBeforeAdmission) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  auto handle = Unwrap(engine->Submit(GreedyRequest({2, 3})));
  engine->Cancel(handle.id());
  RunToIdle(*engine);

  EXPECT_TRUE(DrainTokens(handle).empty());
  EXPECT_EQ(handle.await_completion().terminal, SeqState::kCancelled);
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(EngineTest, CancelDuringDecode) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  auto handle = Unwrap(engine->Submit(GreedyRequest({2}, /*max_tokens=*/32)));

  EXPECT_TRUE(engine->Step());  // prefill: sequence is now RUNNING
  EXPECT_EQ(engine->num_running(), 1U);
  engine->Cancel(handle.id());
  // The cancel is applied at this boundary; retiring the only sequence leaves
  // the engine idle, so Step reports no further work.
  EXPECT_FALSE(engine->Step());

  EXPECT_EQ(handle.await_completion().terminal, SeqState::kCancelled);
  EXPECT_EQ(engine->num_live(), 0U);
  EXPECT_EQ(pool.stats().used, 0);  // decode blocks reclaimed
}

TEST(EngineTest, CancelAfterFinishIsNoOp) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  auto handle = Unwrap(engine->Submit(GreedyRequest({2}, /*max_tokens=*/3)));
  RunToIdle(*engine);
  const FinishInfo before = handle.await_completion();
  ASSERT_EQ(before.terminal, SeqState::kFinished);

  engine->Cancel(handle.id());  // id already retired — no-op
  EXPECT_FALSE(engine->Step());
  EXPECT_EQ(handle.await_completion().terminal, SeqState::kFinished);
}

TEST(EngineTest, CancelUnknownIdIgnored) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  engine->Cancel(9999);  // never submitted
  EXPECT_FALSE(engine->Step());
}

// --- Per-request failure isolation -----------------------------------------

TEST(EngineTest, PerRequestFaultIsolated) {
  CannedModel model;
  model.set_poison_token(7);  // a prompt ending in 7 yields NaN logits → fail
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));

  auto good = Unwrap(engine->Submit(GreedyRequest({2, 3}, /*max_tokens=*/4)));
  auto bad = Unwrap(engine->Submit(GreedyRequest({1, 7}, /*max_tokens=*/4)));
  RunToIdle(*engine);

  // The bad request fails; the good one is unaffected.
  const FinishInfo bad_info = bad.await_completion();
  EXPECT_EQ(bad_info.terminal, SeqState::kFailed);
  EXPECT_FALSE(bad_info.error.ok());
  EXPECT_TRUE(DrainTokens(bad).empty());

  EXPECT_EQ(good.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(DrainTokens(good), ReferenceTokens(model, {2, 3}, 4));
  EXPECT_EQ(pool.stats().used, 0);
}

// --- Loop thread: Start/Stop -----------------------------------------------

TEST(EngineTest, StartStopStreamsTokens) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  engine->Start();

  const std::vector<std::int32_t> prompt = {2, 5, 3};
  auto handle = Unwrap(engine->Submit(GreedyRequest(prompt, /*max_tokens=*/6)));

  std::vector<std::int32_t> tokens;
  while (auto item = handle.next_token()) {
    tokens.push_back(item->token_id);
  }
  EXPECT_EQ(tokens, ReferenceTokens(model, prompt, 6));
  EXPECT_EQ(handle.await_completion().terminal, SeqState::kFinished);

  engine->Stop();
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(EngineTest, StopClosesInFlightChannels) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  // A long-running request (no EOS, near the pool's token capacity) that may
  // still be running when Stop aborts.
  auto handle = Unwrap(engine->Submit(NoEosRequest({2}, /*max_tokens=*/400)));
  engine->Start();
  engine->Stop();  // abort-and-close

  // The channel closes (terminal is kFinished if it happened to reach EOS, else
  // kCancelled from the abort). Either way it must be terminal, not hang.
  const FinishInfo info = handle.await_completion();
  EXPECT_TRUE(info.terminal == SeqState::kCancelled ||
              info.terminal == SeqState::kFinished);
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(EngineTest, SubmitAfterStopFails) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
  engine->Start();
  engine->Stop();
  EXPECT_TRUE(
      IsFailedPrecondition(engine->Submit(GreedyRequest({1})).status()));
}

TEST(EngineTest, DestructorClosesChannels) {
  CannedModel model;
  BlockPool pool = MakePool();
  RequestHandle handle;
  {
    auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));
    handle = Unwrap(engine->Submit(NoEosRequest({2}, /*max_tokens=*/50)));
    // Engine destroyed here without Start/Step — a WAITING sequence.
  }
  EXPECT_EQ(handle.await_completion().terminal, SeqState::kCancelled);
  EXPECT_EQ(pool.stats().used, 0);
}

// --- Stress test ------------------------------------------------------------

TEST(EngineTest, ConcurrentSubmitConsumeCancelStress) {
  CannedModel model;
  BlockPool pool = MakePool(/*num_blocks=*/256);
  auto engine = Unwrap(Engine::Create(
      model, pool,
      EngineConfig{.max_num_seqs = 32, .max_num_batched_tokens = 4096}));
  engine->Start();

  constexpr int kSubmitters = 6;
  constexpr int kPerSubmitter = 40;
  std::atomic<int> completed{0};

  auto worker = [&](unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> tok(0, 6);  // avoid EOS/poison
    std::uniform_int_distribution<int> len(1, 3);
    std::uniform_int_distribution<int> cap(2, 8);
    std::bernoulli_distribution do_cancel(0.25);
    for (int i = 0; i < kPerSubmitter; ++i) {
      std::vector<std::int32_t> prompt(static_cast<std::size_t>(len(rng)));
      for (auto& p : prompt) {
        p = tok(rng);
      }
      auto sub = engine->Submit(GreedyRequest(prompt, cap(rng)));
      if (!sub.ok()) {
        continue;
      }
      RequestHandle handle = *std::move(sub);
      const bool cancel = do_cancel(rng);
      if (cancel) {
        engine->Cancel(handle.id());
      }
      // Drain to completion regardless — the channel must always close.
      while (handle.next_token().has_value()) {
      }
      const FinishInfo info = handle.await_completion();
      EXPECT_TRUE(info.terminal == SeqState::kFinished ||
                  info.terminal == SeqState::kCancelled ||
                  info.terminal == SeqState::kFailed);
      completed.fetch_add(1);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kSubmitters);
  for (int s = 0; s < kSubmitters; ++s) {
    threads.emplace_back(worker, static_cast<unsigned>(1234 + s));
  }
  for (auto& th : threads) {
    th.join();
  }
  engine->Stop();

  EXPECT_EQ(completed.load(), kSubmitters * kPerSubmitter);
  EXPECT_EQ(pool.stats().used, 0);  // no block leaks
}

}  // namespace
