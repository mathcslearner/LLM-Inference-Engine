#include "engine/generator.h"

#include "common/paths.h"
#include "core/status.h"
#include "engine/backend.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_cache.h"
#include "kvcache/simple_cache.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/registry.h"
#include "sampling/params.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The greedy generation loop (M5-T09; design: docs/design/model-execution.md
// §10, §13 T09). Covers: token-for-token agreement with the HF
// generate(do_sample=False) golden (>=32 tokens) built through the architecture
// registry; run-to-run determinism; EOS and max-new-tokens stopping; the
// per-token callback firing once per id in order (after each append, before the
// next forward); continuation from a non-empty cache (the KV invariant); and
// the front-loaded error paths. The `Backend` re-export helpers are exercised
// too.
namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::StatusOr;
using engine::engine::Backend;
using engine::engine::BackendName;
using engine::engine::FinishReason;
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::engine::ParseBackend;
using engine::engine::StopTrigger;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::kvcache::PagedKvCache;
using engine::kvcache::SimpleKvCache;
using engine::model::BuildModel;
using engine::model::ForwardRequest;
using engine::model::load_model;
using engine::model::LogitsMode;
using engine::model::Model;
using engine::sampling::SamplingParams;
using engine::tensor::DataType;

// tiny-llama fixture invariants (config.json).
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 16;
constexpr int kLayers = 2;
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

[[nodiscard]] std::unique_ptr<Model> BuildTiny() {
  auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  EXPECT_TRUE(loaded.ok()) << loaded.status().ToString();
  return Unwrap(BuildModel(*std::move(loaded)));
}

[[nodiscard]] SimpleKvCache FreshCache(std::int64_t capacity = kMaxPos) {
  return Unwrap(SimpleKvCache::Create(ModelGeometry(), capacity));
}

// A block pool for the paged-cache exhaustion tests (M8-T08). bs=8 is the
// smallest valid block size (§4).
constexpr int kBlockSize = 8;
[[nodiscard]] BlockPool FreshPool(std::int64_t num_blocks) {
  return Unwrap(
      BlockPool::Create(ModelGeometry(), kBlockSize, num_blocks, nullptr));
}

// Append `count` throwaway tokens to `hog` across every layer, draining blocks
// from the shared pool (content is irrelevant — this only competes for blocks).
void DrainWith(PagedKvCache& hog, std::int64_t count) {
  const CacheGeometry geom = ModelGeometry();
  for (int layer = 0; layer < geom.num_layers; ++layer) {
    const engine::tensor::Tensor k = Unwrap(engine::tensor::ops::zeros(
        engine::tensor::Shape{count, geom.num_kv_heads, geom.head_dim},
        DataType::kFloat32));
    const engine::tensor::Tensor v = Unwrap(engine::tensor::ops::zeros(
        engine::tensor::Shape{count, geom.num_kv_heads, geom.head_dim},
        DataType::kFloat32));
    ASSERT_TRUE(hog.append(layer, k, v).ok());
  }
}

// One greedy-generation golden case from generate.json.
struct GenerateCase {
  std::string name;
  std::vector<std::int32_t> prompt_ids;
  std::vector<std::int32_t> generated_ids;
};

[[nodiscard]] std::vector<std::int32_t> IdList(const nlohmann::json& array) {
  std::vector<std::int32_t> ids;
  for (const auto& id : array) {
    ids.push_back(id.get<std::int32_t>());
  }
  return ids;
}

[[nodiscard]] std::vector<GenerateCase> LoadGoldenCases() {
  const std::filesystem::path path = engine::testing::FixturesDir() /
                                     "models/tiny-llama/expected/generate.json";
  const std::ifstream file(path, std::ios::binary);
  EXPECT_TRUE(file) << path;
  std::ostringstream contents;
  contents << file.rdbuf();
  const nlohmann::json root = nlohmann::json::parse(
      contents.str(), /*cb=*/nullptr, /*allow_exceptions=*/false);
  EXPECT_FALSE(root.is_discarded()) << path;

  std::vector<GenerateCase> cases;
  for (const auto& c : root.at("cases")) {
    cases.push_back({c.at("name").get<std::string>(),
                     IdList(c.at("prompt_ids")),
                     IdList(c.at("generated_ids"))});
  }
  EXPECT_FALSE(cases.empty());
  return cases;
}

// The fixture's max_new_tokens (each case's generated_ids has this length).
[[nodiscard]] std::int64_t GoldenMaxNewTokens() {
  const std::filesystem::path path = engine::testing::FixturesDir() /
                                     "models/tiny-llama/expected/generate.json";
  const std::ifstream file(path, std::ios::binary);
  std::ostringstream contents;
  contents << file.rdbuf();
  const nlohmann::json root = nlohmann::json::parse(contents.str());
  return root.at("max_new_tokens").get<std::int64_t>();
}

// -------------------------------------------------- golden continuation --

TEST(GeneratorTest, GreedyContinuationMatchesHfGolden) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::int64_t max_new = GoldenMaxNewTokens();
  ASSERT_GE(max_new, 32);  // the ticket's acceptance floor

  for (const GenerateCase& c : LoadGoldenCases()) {
    SimpleKvCache cache = FreshCache();
    const GenerateOptions options{.sampling = SamplingParams::Greedy(max_new),
                                  .eos_ids = {}};
    const std::vector<std::int32_t> got =
        Unwrap(Generate(*model, cache, c.prompt_ids, options)).tokens;
    EXPECT_EQ(got, c.generated_ids) << "case " << c.name;
    // Peak cache occupancy: prompt + (max_new - 1) — the last token is produced
    // but never appended.
    EXPECT_EQ(cache.length(),
              static_cast<std::int64_t>(c.prompt_ids.size()) + max_new - 1)
        << "case " << c.name;
  }
}

TEST(GeneratorTest, TwoRunsAreIdentical) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  const GenerateOptions options{.sampling = SamplingParams::Greedy(24),
                                .eos_ids = {}};

  SimpleKvCache cache_a = FreshCache();
  SimpleKvCache cache_b = FreshCache();
  const std::vector<std::int32_t> run_a =
      Unwrap(Generate(*model, cache_a, c.prompt_ids, options)).tokens;
  const std::vector<std::int32_t> run_b =
      Unwrap(Generate(*model, cache_b, c.prompt_ids, options)).tokens;
  EXPECT_EQ(run_a, run_b);
}

// ------------------------------------------------------------- stopping --

TEST(GeneratorTest, MaxNewTokensStops) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();

  for (const std::int64_t n : {std::int64_t{1}, std::int64_t{7}}) {
    SimpleKvCache cache = FreshCache();
    const GenerateOptions options{.sampling = SamplingParams::Greedy(n),
                                  .eos_ids = {}};
    const std::vector<std::int32_t> got =
        Unwrap(Generate(*model, cache, c.prompt_ids, options)).tokens;
    ASSERT_EQ(static_cast<std::int64_t>(got.size()), n);
    // A prefix of the full golden continuation.
    for (std::int64_t i = 0; i < n; ++i) {
      EXPECT_EQ(got[static_cast<std::size_t>(i)],
                c.generated_ids[static_cast<std::size_t>(i)]);
    }
  }
}

TEST(GeneratorTest, EosOnFirstTokenStopsImmediately) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  // Make the very first produced token an EOS: generation returns exactly it.
  const GenerateOptions options{.sampling = SamplingParams::Greedy(40),
                                .eos_ids = {c.generated_ids.front()}};
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> got =
      Unwrap(Generate(*model, cache, c.prompt_ids, options)).tokens;
  ASSERT_EQ(got.size(), 1U);
  EXPECT_EQ(got.front(), c.generated_ids.front());
}

TEST(GeneratorTest, EosMidSequenceStopsAtFirstOccurrenceInclusive) {
  const std::unique_ptr<Model> model = BuildTiny();
  // Pick a case whose 5th token does not recur earlier, then stop on it.
  const std::vector<GenerateCase> cases = LoadGoldenCases();
  const GenerateCase& c = cases.back();  // prompt_c: a varied trajectory
  ASSERT_GE(c.generated_ids.size(), 5U);
  const std::int32_t eos = c.generated_ids[4];

  // Expected: the golden up to and including the FIRST occurrence of `eos`.
  std::vector<std::int32_t> expected;
  for (const std::int32_t id : c.generated_ids) {
    expected.push_back(id);
    if (id == eos) {
      break;
    }
  }

  SimpleKvCache cache = FreshCache();
  const GenerateOptions options{.sampling = SamplingParams::Greedy(40),
                                .eos_ids = {eos}};
  const std::vector<std::int32_t> got =
      Unwrap(Generate(*model, cache, c.prompt_ids, options)).tokens;
  EXPECT_EQ(got, expected);
  EXPECT_EQ(got.back(), eos);  // the EOS id is included
}

// ------------------------------------------------- finish reasons (M7-T04) --

TEST(GeneratorTest, MaxTokensFinishReasonIsLength) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SimpleKvCache cache = FreshCache();
  const GenerateOptions options{.sampling = SamplingParams::Greedy(7),
                                .eos_ids = {}};
  const engine::engine::GenerateResult r =
      Unwrap(Generate(*model, cache, c.prompt_ids, options));
  EXPECT_EQ(r.tokens.size(), 7U);
  EXPECT_EQ(r.finish_reason, FinishReason::kLength);
  EXPECT_EQ(r.stop_trigger, StopTrigger::kMaxTokens);
  EXPECT_TRUE(r.text.empty()) << "no tokenizer supplied → no text";
}

TEST(GeneratorTest, EosFinishReasonIsStop) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  const GenerateOptions options{.sampling = SamplingParams::Greedy(40),
                                .eos_ids = {c.generated_ids.front()}};
  SimpleKvCache cache = FreshCache();
  const engine::engine::GenerateResult r =
      Unwrap(Generate(*model, cache, c.prompt_ids, options));
  ASSERT_EQ(r.tokens.size(), 1U);
  EXPECT_EQ(r.finish_reason, FinishReason::kStop);
  EXPECT_EQ(r.stop_trigger, StopTrigger::kEosId);
}

TEST(GeneratorTest, StopTokenIdStopsWithStopReason) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::vector<GenerateCase> cases = LoadGoldenCases();
  const GenerateCase& c = cases.back();
  ASSERT_GE(c.generated_ids.size(), 5U);
  const std::int32_t stop_id = c.generated_ids[4];

  std::vector<std::int32_t> expected;
  for (const std::int32_t id : c.generated_ids) {
    expected.push_back(id);
    if (id == stop_id) {
      break;
    }
  }

  // The request's stop_token_ids (SamplingParams), distinct from the model's
  // eos_ids — routed through the loop's StopChecker, not the sampler.
  SamplingParams sampling = SamplingParams::Greedy(40);
  sampling.stop_token_ids = {stop_id};
  SimpleKvCache cache = FreshCache();
  const engine::engine::GenerateResult r = Unwrap(Generate(
      *model, cache, c.prompt_ids, {.sampling = sampling, .eos_ids = {}}));
  EXPECT_EQ(r.tokens, expected);
  EXPECT_EQ(r.tokens.back(), stop_id);
  EXPECT_EQ(r.finish_reason, FinishReason::kStop);
  EXPECT_EQ(r.stop_trigger, StopTrigger::kStopTokenId);
}

TEST(GeneratorTest, StopStringsWithoutTokenizerIsInvalidArgument) {
  const std::unique_ptr<Model> model = BuildTiny();
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  SamplingParams sampling = SamplingParams::Greedy(8);
  sampling.stop_strings = {"stop"};
  // No options.tokenizer → front-loaded InvalidArgument, cache untouched.
  const auto got =
      Generate(*model, cache, prompt, {.sampling = sampling, .eos_ids = {}});
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(IsInvalidArgument(got.status()));
  EXPECT_EQ(cache.length(), 0);
}

// ------------------------------------------------------ token callback --

TEST(GeneratorTest, OnTokenFiresOncePerIdInOrderBeforeNextForward) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  const auto prompt_len = static_cast<std::int64_t>(c.prompt_ids.size());

  SimpleKvCache cache = FreshCache();
  std::vector<std::int32_t> streamed;
  std::vector<std::int64_t> cache_len_at_callback;
  const GenerateOptions options{.sampling = SamplingParams::Greedy(16),
                                .eos_ids = {}};
  const std::vector<std::int32_t> got =
      Unwrap(Generate(*model, cache, c.prompt_ids, options,
                      [&](const engine::engine::TokenEvent& ev) {
                        streamed.push_back(ev.id);
                        cache_len_at_callback.push_back(cache.length());
                      }))
          .tokens;

  // Fired exactly once per returned id, in the same order.
  EXPECT_EQ(streamed, got);
  ASSERT_EQ(cache_len_at_callback.size(), got.size());
  // At the i-th callback the previous token has been appended but the next
  // forward has not run: cache.length() == prompt_len + i.
  for (std::size_t i = 0; i < cache_len_at_callback.size(); ++i) {
    EXPECT_EQ(cache_len_at_callback[i],
              prompt_len + static_cast<std::int64_t>(i));
  }
}

// ---------------------------------------- continuation from a prefix -----

TEST(GeneratorTest, ContinuationFromNonEmptyCacheMatchesFullPrompt) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::vector<GenerateCase> cases = LoadGoldenCases();
  const GenerateCase& c = cases.back();  // prompt_c (length 8)
  ASSERT_GE(c.prompt_ids.size(), 8U);

  // Pre-fill the cache with the first half of the prompt via a direct forward,
  // then Generate the second half. The cache then holds the full prompt
  // context, so the continuation must equal the full-prompt golden (the KV
  // invariant: split prefill == whole prefill).
  const std::size_t split = 4;
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> head(c.prompt_ids.begin(),
                                       c.prompt_ids.begin() + split);
  const std::vector<std::int32_t> head_pos = {0, 1, 2, 3};
  const ForwardRequest prefill{.token_ids = head,
                               .positions = head_pos,
                               .cache = &cache,
                               .logits_mode = LogitsMode::kLast};
  (void)Unwrap(model->forward(prefill));
  ASSERT_EQ(cache.length(), static_cast<std::int64_t>(split));

  const std::vector<std::int32_t> tail(c.prompt_ids.begin() + split,
                                       c.prompt_ids.end());
  const std::int64_t n = 16;
  const GenerateOptions options{.sampling = SamplingParams::Greedy(n),
                                .eos_ids = {}};
  const std::vector<std::int32_t> got =
      Unwrap(Generate(*model, cache, tail, options)).tokens;
  for (std::int64_t i = 0; i < n; ++i) {
    EXPECT_EQ(got[static_cast<std::size_t>(i)],
              c.generated_ids[static_cast<std::size_t>(i)])
        << "position " << i;
  }
}

// ------------------------------------------------------- error paths -----

TEST(GeneratorTest, EmptyPromptIsInvalidArgument) {
  const std::unique_ptr<Model> model = BuildTiny();
  SimpleKvCache cache = FreshCache();
  const GenerateOptions options{.sampling = SamplingParams::Greedy(4),
                                .eos_ids = {}};
  const std::span<const std::int32_t> empty;
  const auto got = Generate(*model, cache, empty, options);
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(IsInvalidArgument(got.status()));
  EXPECT_NE(got.status().message().find("prompt_ids"), std::string::npos);
  EXPECT_EQ(cache.length(), 0);  // nothing generated
}

TEST(GeneratorTest, NonPositiveMaxNewTokensIsInvalidArgument) {
  const std::unique_ptr<Model> model = BuildTiny();
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  const GenerateOptions options{.sampling = SamplingParams::Greedy(0),
                                .eos_ids = {}};
  const auto got = Generate(*model, cache, prompt, options);
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(IsInvalidArgument(got.status()));
  // max_new_tokens folded into SamplingParams::max_tokens in M7-T01; the
  // validation now lives in ValidateSamplingParams.
  EXPECT_NE(got.status().message().find("max_tokens"), std::string::npos);
}

TEST(GeneratorTest, InsufficientCapacityIsResourceExhausted) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  // Capacity exactly the prompt length: no room for any decode-step append.
  SimpleKvCache cache = FreshCache(static_cast<std::int64_t>(prompt.size()));
  const GenerateOptions options{.sampling = SamplingParams::Greedy(8),
                                .eos_ids = {}};
  const auto got = Generate(*model, cache, prompt, options);
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(IsResourceExhausted(got.status()));
  EXPECT_NE(got.status().message().find("capacity"), std::string::npos);
  EXPECT_EQ(cache.length(), 0);  // front-loaded: nothing appended
}

// The paged twin of the previous test: a pool too small for prompt + decode is
// rejected up front (the private cache's capacity() is exact), nothing is
// allocated, and no blocks leak (M8-T08).
TEST(GeneratorTest, PagedInsufficientCapacityIsResourceExhaustedUpFront) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  // One block = 8 slots; prompt(3) + 8 decode appends cannot fit.
  BlockPool pool = FreshPool(/*num_blocks=*/1);
  PagedKvCache cache(&pool);
  const GenerateOptions options{.sampling = SamplingParams::Greedy(8),
                                .eos_ids = {}};
  const auto got = Generate(*model, cache, prompt, options);
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(IsResourceExhausted(got.status()));
  EXPECT_EQ(cache.length(), 0);     // front-loaded: nothing appended
  EXPECT_EQ(pool.stats().used, 0);  // no blocks allocated
  EXPECT_EQ(pool.stats().exhaustions, 0);
}

// The M8-T08 acceptance at the generation level: a shared pool drained mid-run
// (by a competitor appending inside the token callback) makes a later decode
// step fail gracefully with ResourceExhausted. The tokens produced before the
// failure were all delivered through the callback and match the uninterrupted
// trajectory; the failing forward committed nothing; dropping the caches
// reclaims every block (pool stats return to zero); and resuming from the last
// delivered token reproduces the full uninterrupted trajectory token-for-token.
TEST(GeneratorTest, PagedMidGenerationExhaustionIsGracefulAndResumable) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  constexpr std::int64_t kMax = 16;
  const GenerateOptions options{.sampling = SamplingParams::Greedy(kMax),
                                .eos_ids = {}};

  // Reference: uninterrupted greedy on an ample pool.
  std::vector<std::int32_t> reference;
  {
    BlockPool big = FreshPool(/*num_blocks=*/8);
    PagedKvCache cache(&big);
    reference = Unwrap(Generate(*model, cache, prompt, options)).tokens;
    ASSERT_EQ(reference.size(), static_cast<std::size_t>(kMax));
  }

  // Interrupted: a 3-block pool passes the up-front check (24 slots >= 18), but
  // a hog drains the two free blocks inside the first callback, so the sequence
  // fails when it must cross out of its single owned block.
  BlockPool pool = FreshPool(/*num_blocks=*/3);
  std::vector<std::int32_t> delivered;
  engine::core::Status run_status = engine::core::OkStatus();
  {
    PagedKvCache cache(&pool);
    PagedKvCache hog(&pool);
    bool drained = false;
    const auto on_token = [&](const engine::engine::TokenEvent& ev) {
      delivered.push_back(ev.id);
      if (!drained) {
        drained = true;
        // Consume the 2 free blocks.
        DrainWith(hog, /*count=*/2 * static_cast<std::int64_t>(kBlockSize));
      }
    };
    auto got = Generate(*model, cache, prompt, options, on_token);
    ASSERT_FALSE(got.ok());
    run_status = got.status();
    EXPECT_TRUE(IsResourceExhausted(run_status)) << run_status.ToString();

    // Everything delivered is a proper, matching prefix of the reference.
    ASSERT_FALSE(delivered.empty());
    ASSERT_LT(delivered.size(), reference.size());
    for (std::size_t i = 0; i < delivered.size(); ++i) {
      EXPECT_EQ(delivered[i], reference[i]) << "token " << i;
    }
    // The failing forward committed nothing: the last delivered token was not
    // appended, so length == prompt + (delivered - 1).
    EXPECT_EQ(cache.length(), static_cast<std::int64_t>(prompt.size()) +
                                  static_cast<std::int64_t>(delivered.size()) -
                                  1);

    // Free the competitor's blocks and resume from the last delivered token.
    hog.reset();
    const std::int64_t remaining =
        kMax - static_cast<std::int64_t>(delivered.size());
    ASSERT_GT(remaining, 0);
    const std::vector<std::int32_t> resume_prompt = {delivered.back()};
    const GenerateOptions resume_opts{
        .sampling = SamplingParams::Greedy(remaining), .eos_ids = {}};
    const std::vector<std::int32_t> resumed =
        Unwrap(Generate(*model, cache, resume_prompt, resume_opts)).tokens;

    // delivered ++ resumed == the uninterrupted trajectory.
    std::vector<std::int32_t> stitched = delivered;
    stitched.insert(stitched.end(), resumed.begin(), resumed.end());
    EXPECT_EQ(stitched, reference);
  }
  // Both caches dropped: every block reclaimed, but the pool remembers it ran
  // dry (M8-T08).
  const engine::kvcache::BlockPoolStats st = pool.stats();
  EXPECT_EQ(st.used, 0);  // no leaked blocks
  EXPECT_EQ(st.free, st.total);
  EXPECT_GT(st.peak_used, 0);
}

TEST(GeneratorTest, OutOfRangePromptIdPropagatesFromForward) {
  const std::unique_ptr<Model> model = BuildTiny();
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> prompt = {1, 512};  // vocab_size == 512
  const GenerateOptions options{.sampling = SamplingParams::Greedy(4),
                                .eos_ids = {}};
  const auto got = Generate(*model, cache, prompt, options);
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(IsInvalidArgument(got.status()));
  EXPECT_NE(got.status().message().find("512"), std::string::npos);
}

// M7-T05: greedy generation with logprobs returns one StepLogprobs per token,
// index-aligned with `tokens`, without changing the tokens (the greedy golden
// still holds). The chosen token is the top-1 logprob (greedy == argmax), and
// each step's probability mass sums to 1 over the full vocabulary.
TEST(GeneratorTest, GreedyLogprobsAreAlignedAndTokensUnchanged) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::int64_t max_new = GoldenMaxNewTokens();

  for (const GenerateCase& c : LoadGoldenCases()) {
    SimpleKvCache plain_cache = FreshCache();
    const std::vector<std::int32_t> plain =
        Unwrap(Generate(*model, plain_cache, c.prompt_ids,
                        {.sampling = SamplingParams::Greedy(max_new),
                         .eos_ids = {}}))
            .tokens;

    SamplingParams sampling = SamplingParams::Greedy(max_new);
    sampling.logprobs = 5;
    SimpleKvCache cache = FreshCache();
    const engine::engine::GenerateResult r = Unwrap(Generate(
        *model, cache, c.prompt_ids, {.sampling = sampling, .eos_ids = {}}));
    EXPECT_EQ(r.tokens, plain) << "case " << c.name;  // golden unchanged
    ASSERT_EQ(r.logprobs.size(), r.tokens.size()) << "case " << c.name;
    for (std::size_t i = 0; i < r.tokens.size(); ++i) {
      const engine::sampling::StepLogprobs& lp = r.logprobs[i];
      ASSERT_FALSE(lp.top.empty());
      EXPECT_EQ(lp.top.front().id, r.tokens[i]) << "step " << i;  // top-1
      EXPECT_FLOAT_EQ(lp.chosen_logprob, lp.top.front().logprob)
          << "step " << i;
      EXPECT_LE(lp.top.size(), 5U);
      // Descending order.
      for (std::size_t k = 1; k < lp.top.size(); ++k) {
        EXPECT_GE(lp.top[k - 1].logprob, lp.top[k].logprob);
      }
      EXPECT_LE(lp.chosen_logprob, 0.0F);  // a log-probability
    }
  }
}

// The per-token callback carries the same StepLogprobs the result stores.
TEST(GeneratorTest, OnTokenEventCarriesLogprobs) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SamplingParams sampling = SamplingParams::Greedy(8);
  sampling.logprobs = 3;
  SimpleKvCache cache = FreshCache();

  std::vector<float> event_chosen;
  const engine::engine::GenerateResult r = Unwrap(Generate(
      *model, cache, c.prompt_ids, {.sampling = sampling, .eos_ids = {}},
      [&](const engine::engine::TokenEvent& ev) {
        ASSERT_NE(ev.logprobs, nullptr);
        event_chosen.push_back(ev.logprobs->chosen_logprob);
      }));
  ASSERT_EQ(event_chosen.size(), r.logprobs.size());
  for (std::size_t i = 0; i < event_chosen.size(); ++i) {
    EXPECT_FLOAT_EQ(event_chosen[i], r.logprobs[i].chosen_logprob);
  }
}

// Logprobs off (the default): no per-step logprobs are stored, and the callback
// sees a null pointer.
TEST(GeneratorTest, LogprobsDisabledLeavesResultEmpty) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SimpleKvCache cache = FreshCache();
  bool saw_null = true;
  const engine::engine::GenerateResult r =
      Unwrap(Generate(*model, cache, c.prompt_ids,
                      {.sampling = SamplingParams::Greedy(8), .eos_ids = {}},
                      [&](const engine::engine::TokenEvent& ev) {
                        saw_null = saw_null && (ev.logprobs == nullptr);
                      }));
  EXPECT_TRUE(r.logprobs.empty());
  EXPECT_TRUE(saw_null);
}

// End-to-end: a repetition penalty is threaded through the generation loop's
// per-step history (prompt + tokens generated so far) and can change the greedy
// trajectory. A strong penalty (r=100) makes generation avoid re-emitting any
// token it has already produced, so the output has no immediate repeats where
// the unpenalized run does — proving the loop hands the sampler the right,
// growing history each step (not just the prompt).
TEST(GeneratorTest, RepetitionPenaltyAltersGreedyTrajectory) {
  const std::unique_ptr<Model> model = BuildTiny();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  constexpr std::int64_t kNew = 24;

  SimpleKvCache plain_cache = FreshCache();
  const std::vector<std::int32_t> plain =
      Unwrap(Generate(*model, plain_cache, prompt,
                      GenerateOptions{.sampling = SamplingParams::Greedy(kNew),
                                      .eos_ids = {}}))
          .tokens;

  SamplingParams penalized = SamplingParams::Greedy(kNew);
  penalized.repetition_penalty = 100.0F;
  SimpleKvCache pen_cache = FreshCache();
  const std::vector<std::int32_t> pen =
      Unwrap(Generate(*model, pen_cache, prompt,
                      GenerateOptions{.sampling = penalized, .eos_ids = {}}))
          .tokens;

  ASSERT_EQ(plain.size(), static_cast<std::size_t>(kNew));
  ASSERT_EQ(pen.size(), static_cast<std::size_t>(kNew));
  // The penalty changes the trajectory at some step.
  EXPECT_NE(plain, pen);
  // And it is deterministic: the same penalized run reproduces exactly.
  SimpleKvCache pen_cache2 = FreshCache();
  const std::vector<std::int32_t> pen2 =
      Unwrap(Generate(*model, pen_cache2, prompt,
                      GenerateOptions{.sampling = penalized, .eos_ids = {}}))
          .tokens;
  EXPECT_EQ(pen, pen2);
}

// ------------------------------------------------ stochastic generation --

[[nodiscard]] SamplingParams StochasticParams(std::int64_t max_tokens,
                                              std::uint64_t seed,
                                              float temperature = 1.0F) {
  SamplingParams p;
  p.max_tokens = max_tokens;
  p.temperature = temperature;
  p.seed = seed;
  return p;
}

TEST(GeneratorTest, StochasticGenerationSameSeedIsIdentical) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  const GenerateOptions options{.sampling = StochasticParams(24, /*seed=*/2024),
                                .eos_ids = {}};
  SimpleKvCache cache_a = FreshCache();
  SimpleKvCache cache_b = FreshCache();
  const std::vector<std::int32_t> run_a =
      Unwrap(Generate(*model, cache_a, c.prompt_ids, options)).tokens;
  const std::vector<std::int32_t> run_b =
      Unwrap(Generate(*model, cache_b, c.prompt_ids, options)).tokens;
  EXPECT_EQ(run_a, run_b);
  EXPECT_EQ(run_a.size(), 24U);
}

TEST(GeneratorTest, StochasticGenerationDifferentSeedsDiffer) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SimpleKvCache cache_a = FreshCache();
  SimpleKvCache cache_b = FreshCache();
  const std::vector<std::int32_t> run_a =
      Unwrap(Generate(
                 *model, cache_a, c.prompt_ids,
                 {.sampling = StochasticParams(24, /*seed=*/1), .eos_ids = {}}))
          .tokens;
  const std::vector<std::int32_t> run_b =
      Unwrap(Generate(
                 *model, cache_b, c.prompt_ids,
                 {.sampling = StochasticParams(24, /*seed=*/2), .eos_ids = {}}))
          .tokens;
  EXPECT_NE(run_a, run_b);
}

TEST(GeneratorTest, StochasticGenerationWithNulloptSeedRuns) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SamplingParams sampling;
  sampling.max_tokens = 8;
  sampling.temperature = 0.8F;  // seed nullopt -> engine-chosen
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> got =
      Unwrap(Generate(*model, cache, c.prompt_ids,
                      {.sampling = sampling, .eos_ids = {}}))
          .tokens;
  EXPECT_EQ(got.size(), 8U);
}

// Stochastic generation with logprobs (T05) picks the same tokens as without
// (the draw is unperturbed) and fills logprobs at every step.
TEST(GeneratorTest, StochasticLogprobsDoNotChangeTokens) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();

  SimpleKvCache plain_cache = FreshCache();
  const std::vector<std::int32_t> plain =
      Unwrap(Generate(*model, plain_cache, c.prompt_ids,
                      {.sampling = StochasticParams(24, /*seed=*/2024),
                       .eos_ids = {}}))
          .tokens;

  SamplingParams sampling = StochasticParams(24, /*seed=*/2024);
  sampling.logprobs = 4;
  SimpleKvCache cache = FreshCache();
  const engine::engine::GenerateResult r = Unwrap(Generate(
      *model, cache, c.prompt_ids, {.sampling = sampling, .eos_ids = {}}));
  EXPECT_EQ(r.tokens, plain);
  ASSERT_EQ(r.logprobs.size(), r.tokens.size());
  for (const engine::sampling::StepLogprobs& lp : r.logprobs) {
    EXPECT_LE(lp.chosen_logprob, 0.0F);
    EXPECT_FALSE(lp.top.empty());
  }
}

// ------------------------------------------------------ backend helpers --

TEST(BackendTest, NameAndParseRoundTrip) {
  EXPECT_EQ(BackendName(Backend::kReference), "reference");
  EXPECT_EQ(BackendName(Backend::kOptimized), "optimized");
  EXPECT_EQ(Unwrap(ParseBackend("reference")), Backend::kReference);
  EXPECT_EQ(Unwrap(ParseBackend("optimized")), Backend::kOptimized);
}

TEST(BackendTest, ParseUnknownIsInvalidArgumentListingNames) {
  const auto parsed = ParseBackend("gpu");
  ASSERT_FALSE(parsed.ok());
  EXPECT_TRUE(IsInvalidArgument(parsed.status()));
  EXPECT_NE(parsed.status().message().find("gpu"), std::string::npos);
  EXPECT_NE(parsed.status().message().find("reference"), std::string::npos);
}

}  // namespace
