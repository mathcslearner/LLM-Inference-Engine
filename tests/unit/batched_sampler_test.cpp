#include "sampling/batched_sampler.h"

#include "core/status.h"
#include "parallel/thread_pool.h"
#include "sampling/params.h"
#include "sampling/sampler.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

// M7-T06 (design: docs/design/model-execution.md §15.6): the batched sampler.
// The load-bearing acceptance criterion is **token identity with the
// single-sequence reference `Sampler` given identical RNG counters**,
// tie-breaks and logprobs included — so most of this suite samples a batch two
// ways (once through `BatchedSampler`, once row-by-row through the reference
// `Sampler`) and asserts the results are bit-for-bit equal. Because the shared
// per-row pipeline now runs the dispatched `kernels::ExpF32`, the suite
// registers SCALAR_PASS: the identity must hold on the host ISA and under
// forced-scalar alike (and since both paths run the same selected variant
// in-process, it does).

namespace {

using engine::core::IsInternal;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::parallel::ThreadPool;
using engine::sampling::BatchedSampler;
using engine::sampling::BatchRow;
using engine::sampling::SampleContext;
using engine::sampling::Sampler;
using engine::sampling::SampleResult;
using engine::sampling::SamplingParams;
using engine::sampling::StepLogprobs;

// A deterministic pseudo-random logits value from a 3-part key (no RNG dep).
[[nodiscard]] float LogitAt(unsigned row, unsigned col, unsigned salt) {
  unsigned state = (row * 2654435761U) ^ (col * 40503U) ^ (salt * 2246822519U);
  state = state * 1664525U + 1013904223U;
  state ^= state >> 15U;
  state = state * 1664525U + 1013904223U;
  // Map to roughly [-8, 8).
  return ((static_cast<float>(state >> 8U) / static_cast<float>(1U << 24U)) *
          16.0F) -
         8.0F;
}

// A `[batch, vocab]` row-major logits block; row `r` seeded by `(r, salt)`. A
// few rows carry injected `-inf` masks and exact ties to exercise those paths.
[[nodiscard]] std::vector<float> MakeBlock(std::size_t batch, std::size_t vocab,
                                           unsigned salt) {
  std::vector<float> block(batch * vocab);
  for (std::size_t r = 0; r < batch; ++r) {
    for (std::size_t c = 0; c < vocab; ++c) {
      block[(r * vocab) + c] =
          LogitAt(static_cast<unsigned>(r), static_cast<unsigned>(c), salt);
    }
    // Every 3rd row: mask a couple of entries (if the vocab is wide enough).
    if (r % 3 == 1 && vocab >= 4) {
      block[(r * vocab) + 1] = -std::numeric_limits<float>::infinity();
      block[(r * vocab) + (vocab / 2)] =
          -std::numeric_limits<float>::infinity();
    }
    // Every 5th row: create an exact top-2 tie at indices 0 and 2.
    if (r % 5 == 2 && vocab >= 3) {
      block[(r * vocab) + 0] = 5.0F;
      block[(r * vocab) + 2] = 5.0F;
    }
  }
  return block;
}

// A rotating set of per-row sampling configurations spanning every knob.
[[nodiscard]] std::vector<SamplingParams> ParamMenu() {
  std::vector<SamplingParams> menu;
  {  // greedy, no logprobs
    const SamplingParams p = SamplingParams::Greedy(8);
    menu.push_back(p);
  }
  {  // greedy with logprobs
    SamplingParams p = SamplingParams::Greedy(8);
    p.logprobs = 5;
    menu.push_back(p);
  }
  {  // greedy with penalties + logprobs
    SamplingParams p = SamplingParams::Greedy(8);
    p.repetition_penalty = 1.3F;
    p.frequency_penalty = 0.4F;
    p.presence_penalty = 0.2F;
    p.logprobs = 3;
    menu.push_back(p);
  }
  {  // plain temperature
    SamplingParams p;
    p.temperature = 1.0F;
    p.seed = 11U;
    p.max_tokens = 8;
    menu.push_back(p);
  }
  {  // hot temperature + top-k
    SamplingParams p;
    p.temperature = 1.7F;
    p.top_k = 5;
    p.seed = 22U;
    p.max_tokens = 8;
    menu.push_back(p);
  }
  {  // cool temperature + top-p + logprobs
    SamplingParams p;
    p.temperature = 0.5F;
    p.top_p = 0.9F;
    p.logprobs = 4;
    p.seed = 33U;
    p.max_tokens = 8;
    menu.push_back(p);
  }
  {  // top-k + top-p + penalties + logprobs (the works)
    SamplingParams p;
    p.temperature = 0.8F;
    p.top_k = 20;
    p.top_p = 0.95F;
    p.repetition_penalty = 1.2F;
    p.frequency_penalty = 0.3F;
    p.presence_penalty = 0.5F;
    p.logprobs = 20;
    p.seed = 44U;
    p.max_tokens = 8;
    menu.push_back(p);
  }
  return menu;
}

// A short token history for row `r`, all ids in `[0, vocab)`.
[[nodiscard]] std::vector<std::int32_t> MakeHistory(std::size_t r,
                                                    std::size_t vocab,
                                                    std::size_t len) {
  std::vector<std::int32_t> h(len);
  for (std::size_t i = 0; i < len; ++i) {
    h[i] = static_cast<std::int32_t>(((r * 7) + (i * 13)) % vocab);
  }
  return h;
}

void ExpectResultEq(const SampleResult& got, const SampleResult& want,
                    const char* where) {
  EXPECT_EQ(got.token, want.token) << where;
  ASSERT_EQ(got.logprobs.has_value(), want.logprobs.has_value()) << where;
  // Guard both optionals in one condition so the analyzer sees each checked.
  if (!got.logprobs.has_value() || !want.logprobs.has_value()) {
    return;
  }
  const StepLogprobs& g = got.logprobs.value();
  const StepLogprobs& w = want.logprobs.value();
  // Bit-exact: both paths ran identical arithmetic, so no tolerance.
  EXPECT_EQ(g.chosen_logprob, w.chosen_logprob) << where;
  ASSERT_EQ(g.top.size(), w.top.size()) << where;
  for (std::size_t i = 0; i < g.top.size(); ++i) {
    EXPECT_EQ(g.top[i].id, w.top[i].id) << where;
    EXPECT_EQ(g.top[i].logprob, w.top[i].logprob) << where;
  }
}

// Build per-row samplers + contexts, sample the block through the reference and
// through the batched sampler, and assert they match row for row.
void CheckBitExact(std::size_t batch, std::size_t vocab, unsigned salt,
                   BatchedSampler& batched) {
  const std::vector<float> block = MakeBlock(batch, vocab, salt);
  const std::vector<SamplingParams> menu = ParamMenu();

  // Owning storage for samplers, histories, contexts (spans must outlive use).
  std::vector<Sampler> samplers;
  samplers.reserve(batch);
  std::vector<std::vector<std::int32_t>> prompts(batch);
  std::vector<std::vector<std::int32_t>> generated(batch);
  std::vector<BatchRow> rows(batch);
  std::vector<SampleResult> ref(batch);

  for (std::size_t r = 0; r < batch; ++r) {
    const SamplingParams& params = menu[r % menu.size()];
    StatusOr<Sampler> s = Sampler::Create(params);
    ASSERT_TRUE(s.ok()) << s.status().ToString();
    samplers.push_back(*std::move(s));
    prompts[r] = MakeHistory(r, vocab, /*len=*/2 + (r % 3));
    generated[r] = MakeHistory(r + 100, vocab, /*len=*/r % 4);  // step index
  }
  for (std::size_t r = 0; r < batch; ++r) {
    const SampleContext ctx{.prompt_ids = prompts[r],
                            .generated_ids = generated[r]};
    rows[r] = BatchRow{.sampler = &samplers[r], .context = ctx};
    const std::span<const float> row_logits(block.data() + (r * vocab), vocab);
    StatusOr<SampleResult> ref_result =
        samplers[r].SampleWithLogprobs(row_logits, ctx);
    ASSERT_TRUE(ref_result.ok()) << ref_result.status().ToString();
    ref[r] = *std::move(ref_result);
  }

  std::vector<SampleResult> out(batch);
  const Status status =
      batched.Sample(block, static_cast<std::int64_t>(vocab), rows, out);
  ASSERT_TRUE(status.ok()) << status.ToString();
  for (std::size_t r = 0; r < batch; ++r) {
    ExpectResultEq(out[r], ref[r], "batched-vs-reference");
  }
}

// --------------------------------------------------- bit-exact vs reference --

TEST(BatchedSamplerTest, MatchesReferenceAcrossShapesAndParams) {
  BatchedSampler batched;
  unsigned salt = 1U;
  for (const std::size_t batch :
       {std::size_t{1}, std::size_t{3}, std::size_t{17}, std::size_t{64}}) {
    for (const std::size_t vocab :
         {std::size_t{1}, std::size_t{7}, std::size_t{50}, std::size_t{1000},
          std::size_t{32000}}) {
      CheckBitExact(batch, vocab, salt++, batched);
    }
  }
}

// A row sampled in a batch is identical to the same row sampled alone (the
// draw is keyed on (seed, step), never on batch composition).
TEST(BatchedSamplerTest, BatchCompositionIndependent) {
  BatchedSampler batched;
  const std::size_t vocab = 500;
  const std::size_t batch = 32;
  const std::vector<float> block = MakeBlock(batch, vocab, /*salt=*/7);
  const std::vector<SamplingParams> menu = ParamMenu();

  std::vector<Sampler> samplers;
  samplers.reserve(batch);
  std::vector<std::vector<std::int32_t>> gen(batch);
  std::vector<BatchRow> rows(batch);
  for (std::size_t r = 0; r < batch; ++r) {
    StatusOr<Sampler> s = Sampler::Create(menu[r % menu.size()]);
    ASSERT_TRUE(s.ok());
    samplers.push_back(*std::move(s));
    gen[r] = MakeHistory(r + 5, vocab, r % 4);
  }
  for (std::size_t r = 0; r < batch; ++r) {
    rows[r] = BatchRow{.sampler = &samplers[r],
                       .context = {.prompt_ids = {}, .generated_ids = gen[r]}};
  }
  std::vector<SampleResult> full(batch);
  ASSERT_TRUE(
      batched.Sample(block, static_cast<std::int64_t>(vocab), rows, full).ok());

  // Now sample each row as a batch of one.
  for (std::size_t r = 0; r < batch; ++r) {
    const std::span<const float> row_logits(block.data() + (r * vocab), vocab);
    const std::array<BatchRow, 1> one{
        BatchRow{.sampler = &samplers[r],
                 .context = {.prompt_ids = {}, .generated_ids = gen[r]}}};
    std::array<SampleResult, 1> out;
    ASSERT_TRUE(
        batched.Sample(row_logits, static_cast<std::int64_t>(vocab), one, out)
            .ok());
    ExpectResultEq(out[0], full[r], "singleton-vs-batch");
  }
}

// -------------------------------------------------------- thread invariance --

TEST(BatchedSamplerTest, InvariantAcrossThreadCounts) {
  const std::size_t vocab = 777;
  const std::size_t batch = 40;
  const std::vector<float> block = MakeBlock(batch, vocab, /*salt=*/9);
  const std::vector<SamplingParams> menu = ParamMenu();

  std::vector<Sampler> samplers;
  samplers.reserve(batch);
  std::vector<std::vector<std::int32_t>> gen(batch);
  std::vector<BatchRow> rows(batch);
  for (std::size_t r = 0; r < batch; ++r) {
    StatusOr<Sampler> s = Sampler::Create(menu[r % menu.size()]);
    ASSERT_TRUE(s.ok());
    samplers.push_back(*std::move(s));
    gen[r] = MakeHistory(r + 3, vocab, (r + 1) % 4);
    rows[r] = BatchRow{.sampler = &samplers[r],
                       .context = {.prompt_ids = {}, .generated_ids = gen[r]}};
  }

  std::vector<SampleResult> baseline;
  for (const int threads : {1, 2, 3, 8}) {
    ThreadPool pool(threads);
    BatchedSampler batched(pool);
    std::vector<SampleResult> out(batch);
    ASSERT_TRUE(
        batched.Sample(block, static_cast<std::int64_t>(vocab), rows, out)
            .ok());
    if (baseline.empty()) {
      baseline = out;
    } else {
      for (std::size_t r = 0; r < batch; ++r) {
        ExpectResultEq(out[r], baseline[r], "thread-count");
      }
    }
  }
}

// ------------------------------------------------------------ error posture --

TEST(BatchedSamplerTest, RejectsShapeMismatches) {
  BatchedSampler batched;
  const Sampler s = *Sampler::Create(SamplingParams::Greedy(4));
  std::vector<float> logits(3, 0.0F);
  std::array<BatchRow, 1> rows{BatchRow{.sampler = &s, .context = {}}};
  std::array<SampleResult, 1> out;

  // vocab <= 0
  EXPECT_TRUE(IsInvalidArgument(batched.Sample(logits, 0, rows, out)));
  // logits size != batch * vocab (1 row * 3 != 4)
  EXPECT_TRUE(IsInvalidArgument(batched.Sample(logits, 4, rows, out)));
  // out size != rows size
  std::array<SampleResult, 2> out2;
  EXPECT_TRUE(IsInvalidArgument(batched.Sample(logits, 3, rows, out2)));
  // null sampler
  std::array<BatchRow, 1> bad{BatchRow{.sampler = nullptr, .context = {}}};
  EXPECT_TRUE(IsInvalidArgument(batched.Sample(logits, 3, bad, out)));
}

TEST(BatchedSamplerTest, EmptyBatchIsOk) {
  BatchedSampler batched;
  const std::span<const float> logits;
  const std::span<const BatchRow> rows;
  const std::span<SampleResult> out;
  EXPECT_TRUE(batched.Sample(logits, 128, rows, out).ok());
}

TEST(BatchedSamplerTest, SurfacesPerRowErrors) {
  BatchedSampler batched;
  const std::size_t vocab = 16;
  const std::size_t batch = 8;
  // A stochastic sampler so a NaN row reaches CheckFinite.
  SamplingParams temp;
  temp.temperature = 1.0F;
  temp.seed = 1U;
  temp.max_tokens = 4;
  const Sampler s = *Sampler::Create(temp);

  std::vector<float> block(batch * vocab, 0.1F);
  // Row 5 has a NaN → Internal.
  block[(5 * vocab) + 2] = std::numeric_limits<float>::quiet_NaN();
  std::vector<BatchRow> rows(batch, BatchRow{.sampler = &s, .context = {}});
  std::vector<SampleResult> out(batch);
  EXPECT_TRUE(IsInternal(
      batched.Sample(block, static_cast<std::int64_t>(vocab), rows, out)));

  // A bad history id (out of [0, vocab)) → InvalidArgument. Give row 2 a
  // penalty so ApplyPenalties inspects the history.
  SamplingParams pen = temp;
  pen.repetition_penalty = 1.2F;
  const Sampler sp = *Sampler::Create(pen);
  std::vector<float> clean(batch * vocab, 0.1F);
  const std::array<std::int32_t, 1> bad_hist{static_cast<std::int32_t>(vocab)};
  std::vector<BatchRow> rows2(batch, BatchRow{.sampler = &s, .context = {}});
  rows2[2] = BatchRow{.sampler = &sp,
                      .context = {.prompt_ids = bad_hist, .generated_ids = {}}};
  EXPECT_TRUE(IsInvalidArgument(
      batched.Sample(clean, static_cast<std::int64_t>(vocab), rows2, out)));
}

// -------------------------------------------------- allocation-free steady --

TEST(BatchedSamplerTest, AllocationFreeAfterWarmup) {
  BatchedSampler batched;
  const std::size_t vocab = 4096;
  const std::size_t batch = 16;
  const std::vector<float> block = MakeBlock(batch, vocab, /*salt=*/2);
  SamplingParams p;
  p.temperature = 1.0F;
  p.top_p = 0.9F;
  p.logprobs = 5;
  p.seed = 99U;
  p.max_tokens = 8;
  const Sampler s = *Sampler::Create(p);
  std::vector<BatchRow> rows(batch, BatchRow{.sampler = &s, .context = {}});
  std::vector<SampleResult> out(batch);

  ASSERT_TRUE(
      batched.Sample(block, static_cast<std::int64_t>(vocab), rows, out).ok());
  const std::size_t after_first = batched.scratch_bytes();
  EXPECT_GT(after_first, 0U);
  // A second identical-shape step must not grow the scratch.
  ASSERT_TRUE(
      batched.Sample(block, static_cast<std::int64_t>(vocab), rows, out).ok());
  EXPECT_EQ(batched.scratch_bytes(), after_first);
}

}  // namespace
