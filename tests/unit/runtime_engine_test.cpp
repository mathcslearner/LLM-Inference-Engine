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

  // Model a shared/externally-drained pool (§11.2): when armed, the next decode
  // (single-token) forward grabs every currently-free block *before* its own
  // K/V append, so the append exhausts the pool even though the scheduler's
  // start-of-step snapshot saw free blocks. Disarms after firing once. The
  // grabbed blocks are held until `release_drained()`, simulating a persistent
  // external consumer.
  void arm_pool_drain(BlockPool* pool) {
    drain_pool_ = pool;
    drain_armed_ = true;
  }
  void release_drained() {
    if (drain_pool_ != nullptr) {
      for (const std::int32_t b : drained_) {
        drain_pool_->Release(b);
      }
    }
    drained_.clear();
  }

  [[nodiscard]] StatusOr<Tensor> forward(
      const ForwardRequest& request) override {
    // The batched path (cu_seqlens/caches non-empty) is realized exactly like
    // ReferenceModel::ForwardBatched — each member run as a standalone forward
    // over its subspan and its own cache, per-member logits concatenated into
    // `[B, V]` (kLast) — so the batched loop's output equals sequential runs.
    if (!request.caches.empty() || !request.cu_seqlens.empty()) {
      return ForwardBatched(request);
    }
    return ForwardSingle(request);
  }

  [[nodiscard]] const ModelConfig& config() const override { return config_; }
  [[nodiscard]] CacheGeometry cache_geometry() const override {
    return {.num_layers = kLayers,
            .num_kv_heads = kKvHeads,
            .head_dim = kHeadDim,
            .dtype = DataType::kFloat32};
  }

 private:
  // One sequence's forward: append (zeroed) K/V layer by layer to its cache,
  // then a one-hot `[rows, V]` whose argmax is `(last_token + 1) % V` (or an
  // all-NaN row when the last token is the poison token — the fault path).
  [[nodiscard]] StatusOr<Tensor> ForwardSingle(
      const ForwardRequest& request) const {
    const auto t = static_cast<std::int64_t>(request.token_ids.size());
    // Shared-pool drain hook: a single-token (decode) forward grabs all free
    // blocks before appending, so the append below exhausts the pool (§11.2).
    if (drain_armed_ && drain_pool_ != nullptr && t == 1) {
      for (;;) {
        StatusOr<std::int32_t> b = drain_pool_->Allocate();
        if (!b.ok()) {
          break;
        }
        drained_.push_back(*b);
      }
      drain_armed_ = false;
    }
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

  // The batched contract: slice the flattened tokens/positions by cu_seqlens,
  // run each member through `ForwardSingle` over its own cache, and concatenate
  // the per-member logits into `[out_rows, V]` (out_rows = B for kLast).
  [[nodiscard]] StatusOr<Tensor> ForwardBatched(
      const ForwardRequest& request) const {
    const auto num_seqs = static_cast<std::int64_t>(request.caches.size());
    const bool last_only = request.logits_mode == LogitsMode::kLast;
    const std::int64_t t_dim =
        request.cu_seqlens[static_cast<std::size_t>(num_seqs)];
    const std::int64_t out_rows = last_only ? num_seqs : t_dim;
    Tensor logits =
        Unwrap(ops::zeros(Shape{out_rows, kVocab}, DataType::kFloat32));
    auto* out = logits.data_ptr<float>();
    std::int64_t row_cursor = 0;
    for (std::int64_t b = 0; b < num_seqs; ++b) {
      const std::int64_t t0 = request.cu_seqlens[static_cast<std::size_t>(b)];
      const std::int64_t t1 =
          request.cu_seqlens[static_cast<std::size_t>(b + 1)];
      const ForwardRequest sub{
          .token_ids = request.token_ids.subspan(
              static_cast<std::size_t>(t0), static_cast<std::size_t>(t1 - t0)),
          .positions = request.positions.subspan(
              static_cast<std::size_t>(t0), static_cast<std::size_t>(t1 - t0)),
          .cache = request.caches[static_cast<std::size_t>(b)],
          .logits_mode = request.logits_mode,
      };
      StatusOr<Tensor> member = ForwardSingle(sub);
      if (!member.ok()) {
        return member.status();
      }
      const std::int64_t member_rows = last_only ? 1 : (t1 - t0);
      const auto* mbase = member->data_ptr<float>();
      for (std::int64_t r = 0; r < member_rows; ++r) {
        for (std::int64_t i = 0; i < kVocab; ++i) {
          out[((row_cursor + r) * kVocab) + i] = mbase[(r * kVocab) + i];
        }
      }
      row_cursor += member_rows;
    }
    return logits;
  }

  ModelConfig config_;
  std::int32_t poison_ = -1;
  // Shared-pool drain hook (see arm_pool_drain). Mutable so it can fire from
  // the const forward path.
  BlockPool* drain_pool_ = nullptr;
  mutable bool drain_armed_ = false;
  mutable std::vector<std::int32_t> drained_;
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

// §10.3 liveness: an explicitly pinned `max_model_len` the pool cannot hold is
// rejected at Create — otherwise the oldest-alone sequence could not fit and
// preemption could not guarantee progress. (The auto-resolved case, where the
// model's positional limit may dwarf a tiny test pool, is not rejected — every
// other Create test uses a CannedModel with max_position_embeddings=4096 over a
// small pool and must still succeed.)
TEST(EngineTest, CreateRejectsExplicitMaxModelLenOverPoolCapacity) {
  CannedModel model;
  BlockPool pool = MakePool(/*num_blocks=*/2);  // 2 × 8 = 16 token capacity
  EXPECT_TRUE(IsInvalidArgument(
      Engine::Create(model, pool, EngineConfig{.max_model_len = 17}).status()));
}

TEST(EngineTest, CreateAcceptsExplicitMaxModelLenFittingPool) {
  CannedModel model;
  BlockPool pool = MakePool(/*num_blocks=*/2);  // 16 token capacity
  auto engine =
      Unwrap(Engine::Create(model, pool, EngineConfig{.max_model_len = 16}));
  ASSERT_NE(engine, nullptr);  // peak capacity == max_model_len (boundary)
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

// §10.2 config-sizing: a small prompt with a large `max_tokens` whose resume
// re-prefill peak (`prompt + max_tokens - 1`) exceeds the token budget is
// rejected up front — a preempted such sequence could never be re-admitted (no
// chunked prefill until M12-T06) and would stall the queue forever.
TEST(EngineTest, SubmitRejectsPeakOverTokenBudget) {
  CannedModel model;
  BlockPool pool = MakePool(/*num_blocks=*/256);  // ample blocks
  auto engine = Unwrap(
      Engine::Create(model, pool, EngineConfig{.max_num_batched_tokens = 16}));
  // peak = 2 + (20 - 1) = 21 > 16 → rejected, even though the prompt (2) fits.
  EXPECT_TRUE(IsInvalidArgument(
      engine->Submit(GreedyRequest({1, 2}, /*max_tokens=*/20)).status()));
  EXPECT_EQ(engine->num_live(), 0U);
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(EngineTest, SubmitAcceptsPeakAtTokenBudget) {
  CannedModel model;
  BlockPool pool = MakePool(/*num_blocks=*/256);
  auto engine = Unwrap(
      Engine::Create(model, pool, EngineConfig{.max_num_batched_tokens = 16}));
  // peak = 2 + (15 - 1) = 16 == budget → admitted (boundary).
  EXPECT_TRUE(engine->Submit(GreedyRequest({1, 2}, /*max_tokens=*/15)).ok());
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

// --- Scheduler-driven preemption & resume (M9-T04) --------------------------

TEST(EngineTest, PreemptionResumesWithIdenticalOutput) {
  CannedModel model;
  // 2 blocks × 8 = 16 token-slots. Two 8-token prompts each take exactly one
  // block to prefill (used=2, free=0). Next step, both want to decode past
  // their block boundary (pos 8 → block 1) but only one block can be freed, so
  // the scheduler preempts the latest-arrived (r2) to let the oldest (r1)
  // proceed. r2 later resumes by re-prefilling prompt ++ generated, producing
  // output bit-identical to an uninterrupted run (design §10.2).
  BlockPool pool = MakePool(/*num_blocks=*/2);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));

  const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
  auto h1 = Unwrap(engine->Submit(GreedyRequest(prompt, /*max_tokens=*/6)));
  auto h2 = Unwrap(engine->Submit(GreedyRequest(prompt, /*max_tokens=*/6)));

  ASSERT_TRUE(engine->Step());  // step 1: admit + prefill both
  EXPECT_EQ(engine->num_running(), 2U);
  EXPECT_EQ(engine->num_waiting(), 0U);

  ASSERT_TRUE(engine->Step());  // step 2: preempt the latest, decode the oldest
  EXPECT_EQ(engine->num_running(), 1U);  // r1 advances
  EXPECT_EQ(engine->num_waiting(), 1U);  // r2 preempted back to the queue
  EXPECT_EQ(engine->num_live(), 2U);     // r2 is not dropped

  RunToIdle(*engine);

  // Both complete with output identical to a standalone Generate — the
  // preempted-and-resumed r2 included (the resume-equivalence invariant §9.2).
  const std::vector<std::int32_t> reference = ReferenceTokens(model, prompt, 6);
  EXPECT_EQ(DrainTokens(h1), reference);
  EXPECT_EQ(DrainTokens(h2), reference);
  EXPECT_EQ(h1.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(h2.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(engine->num_preemptions(), 1U);  // exactly one preemption occurred
  EXPECT_EQ(pool.stats().used, 0);           // no leaks after resume
}

// M9-T09 acceptance (§13): an artificially tiny pool shared by more sequences
// than it can hold at once forces the scheduler to preempt and re-admit
// repeatedly. Every request still completes with output identical to its
// standalone `Generate`, and the pool is fully reclaimed at the end (no leaks).
TEST(EngineTest, RepeatedPreemptionsAllComplete) {
  CannedModel model;
  // 2 blocks × 8 = 16 token slots. Three sequences each reach a peak of
  // 4 + (8 - 1) = 11 tokens → up to 2 blocks apiece, so a single sequence can
  // consume the whole pool: the three cannot coexist and must be cycled through
  // via preemption. Liveness holds — the oldest-alone peak (11) fits the pool
  // capacity (16), so the oldest always makes progress (§10.3).
  BlockPool pool = MakePool(/*num_blocks=*/2);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));

  const std::vector<std::vector<std::int32_t>> prompts = {
      {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}};
  constexpr std::int64_t kMaxTokens = 8;
  std::vector<RequestHandle> handles;
  handles.reserve(prompts.size());
  for (const auto& p : prompts) {
    handles.push_back(Unwrap(engine->Submit(GreedyRequest(p, kMaxTokens))));
  }
  RunToIdle(*engine);

  EXPECT_GT(engine->num_preemptions(), 0U);  // preemptions were forced
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    EXPECT_EQ(DrainTokens(handles[i]),
              ReferenceTokens(model, prompts[i], kMaxTokens))
        << "sequence " << i << " diverged after preemption";
    EXPECT_EQ(handles[i].await_completion().terminal, SeqState::kFinished);
  }
  EXPECT_EQ(engine->num_live(), 0U);
  EXPECT_EQ(pool.stats().used, 0);  // no leaks
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

// Cancel a sequence right after prefill (RUNNING, before any decode): its
// prompt blocks are reclaimed promptly at the next boundary, and a second
// concurrent request — untouched — keeps its own blocks and completes.
TEST(EngineTest, CancelAfterPrefillReclaimsPromptBlocks) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));
  auto victim = Unwrap(engine->Submit(GreedyRequest({1, 2, 3}, 8)));
  auto other = Unwrap(engine->Submit(GreedyRequest({4, 5, 6}, 8)));

  ASSERT_TRUE(engine->Step());  // prefill both: both RUNNING, both hold a block
  EXPECT_EQ(engine->num_running(), 2U);
  const std::int64_t used_after_prefill = pool.stats().used;
  EXPECT_EQ(used_after_prefill, 2);  // one block each

  engine->Cancel(victim.id());
  ASSERT_TRUE(engine->Step());  // cancel applied: victim retired, blocks freed
  EXPECT_EQ(victim.await_completion().terminal, SeqState::kCancelled);
  EXPECT_EQ(pool.stats().used, 1);  // only `other`'s block remains

  RunToIdle(*engine);
  // `other` was undisturbed: its output matches a standalone Generate.
  EXPECT_EQ(DrainTokens(other), ReferenceTokens(model, {4, 5, 6}, 8));
  EXPECT_EQ(other.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(pool.stats().used, 0);  // no leaks
}

// Cancel a PREEMPTED sequence (cache already released): the cancel is honored,
// nothing double-frees, and the surviving sequence completes unchanged (§11.1).
TEST(EngineTest, CancelWhilePreempted) {
  CannedModel model;
  // The 2-block preemption recipe of PreemptionResumesWithIdenticalOutput: two
  // full-block prompts, the latest-arrived gets preempted on the first decode.
  BlockPool pool = MakePool(/*num_blocks=*/2);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));
  const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
  auto older = Unwrap(engine->Submit(GreedyRequest(prompt, /*max_tokens=*/6)));
  auto younger =
      Unwrap(engine->Submit(GreedyRequest(prompt, /*max_tokens=*/6)));

  ASSERT_TRUE(engine->Step());  // prefill both
  ASSERT_TRUE(engine->Step());  // preempt the younger, decode the older
  ASSERT_EQ(engine->num_waiting(), 1U);  // younger is PREEMPTED in the queue
  ASSERT_EQ(engine->num_preemptions(), 1U);

  engine->Cancel(younger.id());
  RunToIdle(*engine);

  EXPECT_EQ(younger.await_completion().terminal, SeqState::kCancelled);
  (void)DrainTokens(younger);  // drain whatever it emitted before preemption
  // The older sequence is unaffected by the cancel of its preempted peer.
  EXPECT_EQ(DrainTokens(older), ReferenceTokens(model, prompt, 6));
  EXPECT_EQ(older.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(engine->num_live(), 0U);
  EXPECT_EQ(pool.stats().used, 0);  // no leaks, no double-free
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

// A sampler fault that fires mid-decode (not at prefill) inside a batch with
// two healthy neighbors fails only the faulting row — the per-row
// `BatchedSampler` status path (§11.2). The mock generates p+1, p+2, …; `bad`
// (prompt ending in 5) generates 6 then 7 and, decoding *from* the poison token
// 7, faults — so the error lands in a batched decode pass. The healthy
// neighbors run short enough that they never feed 7 back as input, so they
// never touch the poison.
TEST(EngineTest, DecodeFaultIsolatedFromHealthyNeighbors) {
  CannedModel model;
  model.set_poison_token(7);
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));

  auto good1 = Unwrap(engine->Submit(GreedyRequest({1, 2}, /*max_tokens=*/4)));
  auto bad = Unwrap(engine->Submit(GreedyRequest({4, 5}, /*max_tokens=*/6)));
  auto good2 = Unwrap(engine->Submit(GreedyRequest({0, 1}, /*max_tokens=*/4)));
  RunToIdle(*engine);

  // `bad` fails when it decodes the poison token; the fault is a decode-pass
  // per-row sampler error, isolated to `bad`.
  const FinishInfo bad_info = bad.await_completion();
  EXPECT_EQ(bad_info.terminal, SeqState::kFailed);
  EXPECT_FALSE(bad_info.error.ok());

  // Both healthy neighbors produce exactly their standalone trajectories.
  EXPECT_EQ(good1.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(DrainTokens(good1), ReferenceTokens(model, {1, 2}, 4));
  EXPECT_EQ(good2.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(DrainTokens(good2), ReferenceTokens(model, {0, 1}, 4));
  EXPECT_EQ(pool.stats().used, 0);
}

// --- Reactive decode-exhaustion routing (M9-T10, §11.2) --------------------

// A decode append that exhausts a shared/externally-drained pool is routed to
// graceful preemption of a younger sequence rather than per-request failure:
// the older sequence proceeds, the younger is preempted and later resumes with
// bit-identical output. Without the routing the older sequence would fail.
TEST(EngineTest, SharedPoolDecodeExhaustionPreemptsYounger) {
  CannedModel model;
  // 3 blocks × 8 slots. `older`'s 8-token prompt fills block 0 exactly, so its
  // first decode (pos 8) needs a new block; `younger`'s 7-token prompt leaves
  // room for its first decode (pos 7) in the same block, so it needs none. The
  // scheduler (snapshot: 1 free block) therefore schedules both to decode
  // without preemption — then the drain hook fires during `older`'s append.
  BlockPool pool = MakePool(/*num_blocks=*/3);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));
  const std::vector<std::int32_t> older_prompt = {1, 1, 1, 1, 1, 1, 1, 1};
  const std::vector<std::int32_t> younger_prompt = {2, 2, 2, 2, 2, 2, 2};
  auto older = Unwrap(engine->Submit(GreedyRequest(older_prompt, 4)));
  auto younger = Unwrap(engine->Submit(GreedyRequest(younger_prompt, 4)));

  ASSERT_TRUE(engine->Step());  // prefill both; used = 2, free = 1
  ASSERT_EQ(engine->num_running(), 2U);
  ASSERT_EQ(pool.stats().used, 2);

  model.arm_pool_drain(&pool);  // the next decode append will find 0 free
  ASSERT_TRUE(engine->Step());  // decode: older exhausts → preempt younger
  // Older advanced; younger was reactively preempted (not failed).
  EXPECT_GE(engine->num_preemptions(), 1U);
  EXPECT_EQ(engine->num_running(), 1U);
  EXPECT_EQ(engine->num_waiting(), 1U);

  model.release_drained();  // the external consumer frees its blocks
  RunToIdle(*engine);

  // Both complete successfully with standalone-identical output — the older
  // never failed, the younger resumed.
  EXPECT_EQ(older.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(DrainTokens(older), ReferenceTokens(model, older_prompt, 4));
  EXPECT_EQ(younger.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(DrainTokens(younger), ReferenceTokens(model, younger_prompt, 4));
  EXPECT_EQ(pool.stats().used, 0);  // no leaks
}

// The sole-occupant case: a decode exhaustion with no younger sequence to
// preempt genuinely cannot proceed, so that one request fails
// (ResourceExhausted) while the engine loop survives (§11.2).
TEST(EngineTest, SoleOccupantDecodeExhaustionFails) {
  CannedModel model;
  BlockPool pool = MakePool(/*num_blocks=*/2);
  auto engine = Unwrap(Engine::Create(
      model, pool, EngineConfig{.max_num_batched_tokens = 4096}));
  const std::vector<std::int32_t> prompt = {1, 1, 1, 1, 1, 1, 1, 1};
  auto only = Unwrap(engine->Submit(GreedyRequest(prompt, /*max_tokens=*/4)));

  ASSERT_TRUE(engine->Step());  // prefill: fills block 0, used = 1, free = 1
  model.arm_pool_drain(&pool);  // decode append will find 0 free
  ASSERT_TRUE(
      engine->Step());  // decode: exhausts, no younger to preempt → fail

  const FinishInfo info = only.await_completion();
  EXPECT_EQ(info.terminal, SeqState::kFailed);
  EXPECT_TRUE(IsResourceExhausted(info.error));
  EXPECT_EQ(engine->num_preemptions(), 0U);  // nothing was preempted
  EXPECT_EQ(engine->num_live(), 0U);

  model.release_drained();
  EXPECT_EQ(pool.stats().used, 0);  // no leaks once the drain is released
}

// --- Continuous-batching invariant (M9-T08) --------------------------------

// The milestone's headline property (§8.5, §9.2 invariant 3): N concurrent
// greedy requests produce output identical to running each sequentially. Here
// the mock's per-member batched forward is bit-identical to its single-sequence
// forward by construction, so the batched loop's output must equal the
// standalone `Generate` oracle for every sequence.
TEST(EngineTest, EightConcurrentGreedyMatchesSequential) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));

  const std::vector<std::vector<std::int32_t>> prompts = {
      {2}, {3, 4}, {1}, {5, 6, 7}, {2, 2}, {4}, {6}, {1, 3}};
  std::vector<RequestHandle> handles;
  handles.reserve(prompts.size());
  for (const auto& p : prompts) {
    handles.push_back(
        Unwrap(engine->Submit(GreedyRequest(p, /*max_tokens=*/16))));
  }
  RunToIdle(*engine);

  for (std::size_t i = 0; i < prompts.size(); ++i) {
    EXPECT_EQ(DrainTokens(handles[i]), ReferenceTokens(model, prompts[i], 16))
        << "sequence " << i << " diverged from its standalone run";
    EXPECT_EQ(handles[i].await_completion().terminal, SeqState::kFinished);
  }
  EXPECT_EQ(pool.stats().used, 0);  // no leaks
}

// A request arriving mid-flight joins batching without perturbing the running
// sequence (§9.2, staggered submission). Advancing A several steps before B is
// submitted forces a *mixed* step — A decodes while B prefills in the same step
// — and A's output must be unchanged from a run where B never existed.
TEST(EngineTest, MidFlightArrivalDoesNotDisturbRunning) {
  CannedModel model;
  BlockPool pool = MakePool();
  auto engine = Unwrap(Engine::Create(model, pool, EngineConfig{}));

  const std::vector<std::int32_t> pa = {2};
  const std::vector<std::int32_t> pb = {6};
  auto a = Unwrap(engine->Submit(GreedyRequest(pa, /*max_tokens=*/16)));
  // Advance A a few steps (prefill + two decodes) before B arrives.
  EXPECT_TRUE(engine->Step());
  EXPECT_TRUE(engine->Step());
  EXPECT_TRUE(engine->Step());
  auto b = Unwrap(engine->Submit(GreedyRequest(pb, /*max_tokens=*/16)));
  RunToIdle(*engine);

  EXPECT_EQ(DrainTokens(a), ReferenceTokens(model, pa, 16));
  EXPECT_EQ(DrainTokens(b), ReferenceTokens(model, pb, 16));
  EXPECT_EQ(a.await_completion().terminal, SeqState::kFinished);
  EXPECT_EQ(b.await_completion().terminal, SeqState::kFinished);
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
