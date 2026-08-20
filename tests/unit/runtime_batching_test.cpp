#include "common/paths.h"
#include "core/status.h"
#include "engine/generator.h"
#include "kvcache/block_pool.h"
#include "kvcache/simple_cache.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/registry.h"
#include "runtime/channel.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "sampling/params.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Continuous-batching integration on real fixtures (M9-T08; design:
// docs/design/scheduler-runtime.md §9.2, §9.3, §13). The milestone's headline
// property: N concurrent greedy requests driven through the `Engine`'s batched
// loop produce output **identical** to running each sequentially through the
// single-sequence `engine::Generate` — the continuous-batching invariant, which
// §8.5 guarantees bit-for-bit on a fixed ISA (every op is row-/sequence-local
// and the batched GEMM equals the single-row GEMV). Run on both fixtures
// (tiny-llama untied, tiny-qwen2 tied+biases) on the optimized backend, plus a
// recorded throughput-sanity ratio (§9.3).
//
// Full forwards run inside, so this suite registers SCALAR_PASS (the forced-
// scalar pass covers the shipped kernel bytes) — the first runtime suite to do
// so.

namespace {

using engine::core::StatusOr;
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::kvcache::BlockPool;
using engine::kvcache::SimpleKvCache;
using engine::model::Backend;
using engine::model::BuildModel;
using engine::model::BuildOptions;
using engine::model::load_model;
using engine::model::Model;
using engine::runtime::Engine;
using engine::runtime::EngineConfig;
using engine::runtime::Request;
using engine::runtime::RequestHandle;
using engine::runtime::SeqState;
using engine::sampling::SamplingParams;

constexpr const char* kLlama = "models/tiny-llama";
constexpr const char* kQwen = "models/tiny-qwen2";

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

// Load and build the optimized backend through the architecture registry (the
// acceptance path — the same graph the CLI runs).
[[nodiscard]] std::unique_ptr<Model> OptimizedModel(
    const std::string& fixture) {
  auto loaded = load_model(engine::testing::FixturesDir() / fixture);
  EXPECT_TRUE(loaded.ok()) << loaded.status().ToString();
  return Unwrap(BuildModel(*std::move(loaded),
                           BuildOptions{.backend = Backend::kOptimized}));
}

[[nodiscard]] Request GreedyRequest(std::vector<std::int32_t> prompt,
                                    std::vector<std::int32_t> eos,
                                    std::int64_t max_tokens) {
  Request req;
  req.prompt_ids = std::move(prompt);
  req.params = SamplingParams::Greedy(max_tokens);
  req.eos_ids = std::move(eos);
  return req;
}

// The single-sequence oracle: a standalone greedy `Generate` over a fresh
// `SimpleKvCache` (the paged/simple equivalence is M8-T06/T07's bit-exact
// guarantee, so the Engine's paged run must match this).
[[nodiscard]] std::vector<std::int32_t> SequentialGreedy(
    Model& model, const std::vector<std::int32_t>& prompt,
    const std::vector<std::int32_t>& eos, std::int64_t max_tokens) {
  SimpleKvCache cache =
      Unwrap(SimpleKvCache::Create(model.cache_geometry(), /*capacity=*/512));
  GenerateOptions opts;
  opts.sampling = SamplingParams::Greedy(max_tokens);
  opts.eos_ids = eos;
  return Unwrap(Generate(model, cache, prompt, opts)).tokens;
}

void RunToIdle(Engine& engine) {
  while (engine.Step()) {
  }
}

[[nodiscard]] std::vector<std::int32_t> DrainTokens(RequestHandle& handle) {
  std::vector<std::int32_t> tokens;
  while (auto item = handle.try_next_token()) {
    tokens.push_back(item->token_id);
  }
  return tokens;
}

// Eight varied prompts (small valid ids; vocab is 512 for both fixtures).
[[nodiscard]] std::vector<std::vector<std::int32_t>> EightPrompts() {
  return {{1, 5, 9},   {7, 2},     {13, 4, 8, 1}, {3},
          {11, 6, 10}, {2, 14, 5}, {9, 9},        {4, 1, 7, 12}};
}

// The headline invariant on one fixture: 8 concurrent greedy requests through
// the Engine == 8 sequential `Generate` runs, token-for-token, no block leaks.
void RunBatchingInvariant(const std::string& fixture) {
  std::unique_ptr<Model> model = OptimizedModel(fixture);
  const std::vector<std::int32_t> eos = model->config().eos_token_ids;
  // bs=16, 64 blocks = 1024 token slots — ample for 8×(≤4+24), so this test
  // exercises pure batching with no preemption (preemption is M9-T09).
  BlockPool pool =
      Unwrap(BlockPool::Create(model->cache_geometry(), 16, 64, nullptr));
  auto engine = Unwrap(Engine::Create(*model, pool, EngineConfig{}));

  const auto prompts = EightPrompts();
  constexpr std::int64_t kMaxTokens = 24;
  std::vector<RequestHandle> handles;
  handles.reserve(prompts.size());
  for (const auto& p : prompts) {
    handles.push_back(
        Unwrap(engine->Submit(GreedyRequest(p, eos, kMaxTokens))));
  }
  RunToIdle(*engine);

  for (std::size_t i = 0; i < prompts.size(); ++i) {
    const std::vector<std::int32_t> got = DrainTokens(handles[i]);
    const std::vector<std::int32_t> want =
        SequentialGreedy(*model, prompts[i], eos, kMaxTokens);
    EXPECT_EQ(got, want) << fixture << " sequence " << i
                         << " diverged from its standalone Generate";
    EXPECT_TRUE(handles[i].await_completion().terminal == SeqState::kFinished);
  }
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(RuntimeBatchingTest, LlamaEightConcurrentMatchesSequential) {
  RunBatchingInvariant(kLlama);
}

TEST(RuntimeBatchingTest, QwenEightConcurrentMatchesSequential) {
  RunBatchingInvariant(kQwen);
}

// Staggered submission: requests arriving mid-flight join batching without
// perturbing running sequences (§9.2). Submit four, step several times, then
// submit four more; every request's output still equals its standalone run.
TEST(RuntimeBatchingTest, StaggeredArrivalsMatchSequential) {
  std::unique_ptr<Model> model = OptimizedModel(kLlama);
  const std::vector<std::int32_t> eos = model->config().eos_token_ids;
  BlockPool pool =
      Unwrap(BlockPool::Create(model->cache_geometry(), 16, 64, nullptr));
  auto engine = Unwrap(Engine::Create(*model, pool, EngineConfig{}));

  const auto prompts = EightPrompts();
  constexpr std::int64_t kMaxTokens = 20;
  std::vector<RequestHandle> handles;
  handles.reserve(prompts.size());

  for (std::size_t i = 0; i < 4; ++i) {
    handles.push_back(
        Unwrap(engine->Submit(GreedyRequest(prompts[i], eos, kMaxTokens))));
  }
  // Advance the first wave a few steps (they are mid-decode when the rest
  // join).
  for (int s = 0; s < 5; ++s) {
    (void)engine->Step();
  }
  for (std::size_t i = 4; i < prompts.size(); ++i) {
    handles.push_back(
        Unwrap(engine->Submit(GreedyRequest(prompts[i], eos, kMaxTokens))));
  }
  RunToIdle(*engine);

  for (std::size_t i = 0; i < prompts.size(); ++i) {
    EXPECT_EQ(DrainTokens(handles[i]),
              SequentialGreedy(*model, prompts[i], eos, kMaxTokens))
        << "staggered sequence " << i;
  }
  EXPECT_EQ(pool.stats().used, 0);
}

// Throughput sanity (§9.3): 8 concurrent requests complete in well under 8× a
// single request's wall-clock, because the decode steps batch (8 sequences'
// decode is one forward, not 8). Recorded, not asserted against a threshold —
// the test still asserts output equality so it is not vacuous.
TEST(RuntimeBatchingTest, ThroughputSanityRatioRecorded) {
  using clock = std::chrono::steady_clock;
  std::unique_ptr<Model> model = OptimizedModel(kLlama);
  const std::vector<std::int32_t> eos = model->config().eos_token_ids;
  const auto prompts = EightPrompts();
  constexpr std::int64_t kMaxTokens = 32;

  // Sequential baseline: 8 standalone Generate runs.
  const auto seq_start = clock::now();
  std::vector<std::vector<std::int32_t>> sequential;
  sequential.reserve(prompts.size());
  for (const auto& p : prompts) {
    sequential.push_back(SequentialGreedy(*model, p, eos, kMaxTokens));
  }
  const auto seq_elapsed = clock::now() - seq_start;

  // Concurrent: all 8 through the batched Engine loop.
  BlockPool pool =
      Unwrap(BlockPool::Create(model->cache_geometry(), 16, 64, nullptr));
  auto engine = Unwrap(Engine::Create(*model, pool, EngineConfig{}));
  const auto cc_start = clock::now();
  std::vector<RequestHandle> handles;
  handles.reserve(prompts.size());
  for (const auto& p : prompts) {
    handles.push_back(
        Unwrap(engine->Submit(GreedyRequest(p, eos, kMaxTokens))));
  }
  RunToIdle(*engine);
  std::vector<std::vector<std::int32_t>> concurrent;
  concurrent.reserve(prompts.size());
  for (auto& h : handles) {
    concurrent.push_back(DrainTokens(h));
  }
  const auto cc_elapsed = clock::now() - cc_start;

  for (std::size_t i = 0; i < prompts.size(); ++i) {
    EXPECT_EQ(concurrent[i], sequential[i]) << "throughput-run sequence " << i;
  }
  EXPECT_EQ(pool.stats().used, 0);

  const double seq_ms =
      std::chrono::duration<double, std::milli>(seq_elapsed).count();
  const double cc_ms =
      std::chrono::duration<double, std::milli>(cc_elapsed).count();
  std::cout << "[ throughput ] 8x sequential = " << seq_ms
            << " ms, 8 concurrent = " << cc_ms
            << " ms, concurrent/sequential ratio = "
            << (seq_ms > 0.0 ? cc_ms / seq_ms : 0.0) << " (recorded, §9.3)\n";
}

}  // namespace
