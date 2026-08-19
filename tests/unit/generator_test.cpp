#include "engine/generator.h"

#include "common/paths.h"
#include "core/status.h"
#include "engine/backend.h"
#include "kvcache/kv_cache.h"
#include "kvcache/simple_cache.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/registry.h"
#include "sampling/params.h"
#include "tensor/dtype.h"

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
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::engine::ParseBackend;
using engine::kvcache::CacheGeometry;
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
        Unwrap(Generate(*model, cache, c.prompt_ids, options));
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
      Unwrap(Generate(*model, cache_a, c.prompt_ids, options));
  const std::vector<std::int32_t> run_b =
      Unwrap(Generate(*model, cache_b, c.prompt_ids, options));
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
        Unwrap(Generate(*model, cache, c.prompt_ids, options));
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
      Unwrap(Generate(*model, cache, c.prompt_ids, options));
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
      Unwrap(Generate(*model, cache, c.prompt_ids, options));
  EXPECT_EQ(got, expected);
  EXPECT_EQ(got.back(), eos);  // the EOS id is included
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
  const std::vector<std::int32_t> got = Unwrap(
      Generate(*model, cache, c.prompt_ids, options, [&](std::int32_t id) {
        streamed.push_back(id);
        cache_len_at_callback.push_back(cache.length());
      }));

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
      Unwrap(Generate(*model, cache, tail, options));
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

TEST(GeneratorTest, NotYetImplementedSamplingIsRejectedBeforeGenerating) {
  const std::unique_ptr<Model> model = BuildTiny();
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> prompt = {1, 5, 9};
  // A still-unimplemented knob (penalties land in M7-T03): the sampler's Create
  // rejects it with Unimplemented, front-loaded so nothing is generated and the
  // cache is untouched.
  SamplingParams sampling = SamplingParams::Greedy(8);
  sampling.repetition_penalty = 1.1F;
  const GenerateOptions options{.sampling = sampling, .eos_ids = {}};
  const auto got = Generate(*model, cache, prompt, options);
  ASSERT_FALSE(got.ok());
  EXPECT_TRUE(engine::core::IsUnimplemented(got.status()))
      << got.status().ToString();
  EXPECT_EQ(cache.length(), 0);
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
      Unwrap(Generate(*model, cache_a, c.prompt_ids, options));
  const std::vector<std::int32_t> run_b =
      Unwrap(Generate(*model, cache_b, c.prompt_ids, options));
  EXPECT_EQ(run_a, run_b);
  EXPECT_EQ(run_a.size(), 24U);
}

TEST(GeneratorTest, StochasticGenerationDifferentSeedsDiffer) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SimpleKvCache cache_a = FreshCache();
  SimpleKvCache cache_b = FreshCache();
  const std::vector<std::int32_t> run_a = Unwrap(
      Generate(*model, cache_a, c.prompt_ids,
               {.sampling = StochasticParams(24, /*seed=*/1), .eos_ids = {}}));
  const std::vector<std::int32_t> run_b = Unwrap(
      Generate(*model, cache_b, c.prompt_ids,
               {.sampling = StochasticParams(24, /*seed=*/2), .eos_ids = {}}));
  EXPECT_NE(run_a, run_b);
}

TEST(GeneratorTest, StochasticGenerationWithNulloptSeedRuns) {
  const std::unique_ptr<Model> model = BuildTiny();
  const GenerateCase c = LoadGoldenCases().front();
  SamplingParams sampling;
  sampling.max_tokens = 8;
  sampling.temperature = 0.8F;  // seed nullopt -> engine-chosen
  SimpleKvCache cache = FreshCache();
  const std::vector<std::int32_t> got = Unwrap(Generate(
      *model, cache, c.prompt_ids, {.sampling = sampling, .eos_ids = {}}));
  EXPECT_EQ(got.size(), 8U);
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
