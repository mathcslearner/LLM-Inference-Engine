#include "sampling/sampler.h"

#include "core/status.h"
#include "sampling/logprobs.h"
#include "sampling/params.h"
#include "sampling/philox.h"
#include "sampling/stages.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

// M7-T06: the sampler's per-row pipeline factored into `detail::SampleRow`, the
// single arithmetic implementation shared with the batched sampler
// (batched_sampler.{h,cpp}) so the two pick identical tokens by construction.
// `Sample` is the token-only path; `SampleWithLogprobs` additionally returns
// the per-step logprobs — both delegate to `SampleRow`. The history-based
// penalties (repetition/frequency/presence) are stage 1, applied to the raw
// logits *before* temperature (design §15.2) — and ahead of the greedy argmax
// too, so a penalty can move the chosen token. `temperature == 0` is greedy
// argmax (over the penalized logits); `temperature > 0` runs the full
// stochastic pipeline (penalties -> temperature -> top-k -> top-p -> Philox
// categorical draw). Logprobs, when requested, are the natural-log softmax of
// the post-penalty/post-temperature logits taken *before* top-k/top-p masking,
// so they describe the full-vocabulary distribution (§15.2 stage 6). Stop
// conditions (T04) are the generation loop's, not the sampler's, so `Create`
// accepts `stop_token_ids`/`stop_strings` and ignores them. As of T05 every
// field is implemented — `Create` no longer rejects any knob with
// `Unimplemented`.

namespace engine::sampling {
namespace {

// Resolve the RNG seed: honour an explicit request seed, else draw a
// nondeterministic one. `random_device` is mixed with a steady-clock tick so
// two samplers created back-to-back get distinct seeds even where
// `random_device` is weak/deterministic.
[[nodiscard]] std::uint64_t ResolveSeed(const SamplingParams& params) {
  if (params.seed.has_value()) {
    return *params.seed;
  }
  std::random_device rd;
  std::uint64_t seed = (static_cast<std::uint64_t>(rd()) << 32) ^
                       static_cast<std::uint64_t>(rd());
  seed ^= static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  return seed;
}

// True when any history penalty is non-default, i.e. `ApplyPenalties` would
// change the logits. Used to keep the default greedy path allocation-free and
// its result bitwise identical to the pre-T03 sampler.
[[nodiscard]] bool PenaltiesActive(const SamplingParams& params) {
  return params.repetition_penalty != 1.0F || params.presence_penalty != 0.0F ||
         params.frequency_penalty != 0.0F;
}

// Argmax over a `[V]` fp32 logits row with a strict `>` scan so the lowest
// vocab index wins ties (deterministic). A NaN maximum is surfaced as
// `Internal`. Ported verbatim from the M5-T09 generation loop's `ArgmaxLastRow`
// — moving greedy selection into the sampler is the whole point of M7-T01.
[[nodiscard]] core::StatusOr<std::int32_t> ArgmaxRow(
    std::span<const float> logits) {
  if (logits.empty()) {
    return core::InvalidArgumentError("logits row must be non-empty");
  }
  std::size_t best = 0;
  float best_val = logits[0];
  for (std::size_t v = 1; v < logits.size(); ++v) {
    if (logits[v] > best_val) {
      best_val = logits[v];
      best = v;
    }
  }
  if (std::isnan(best_val)) {
    return core::InternalError("argmax over non-finite logits (max is NaN)");
  }
  return static_cast<std::int32_t>(best);
}

// The greedy branch of `SampleRow`: penalties (stage 1) then argmax.
// `want_logprobs` adds the log-softmax of the (penalized) row. The
// no-penalty/no-logprobs case keeps the allocation-free, bitwise-unchanged
// pre-T03 argmax path (reading the caller's logits directly).
[[nodiscard]] core::StatusOr<SampleResult> SampleGreedy(
    std::span<const float> logits, const SampleContext& context,
    const SamplingParams& params, bool want_logprobs,
    detail::RowScratch& scratch) {
  const bool penalties = PenaltiesActive(params);
  if (!penalties && !want_logprobs) {
    core::StatusOr<std::int32_t> token = ArgmaxRow(logits);
    if (!token.ok()) {
      return token.status();
    }
    return SampleResult{.token = *token, .logprobs = std::nullopt};
  }
  if (logits.empty()) {
    return core::InvalidArgumentError("logits row must be non-empty");
  }
  // A penalized row needs its own buffer; otherwise argmax/log-softmax read the
  // caller's logits directly.
  std::span<const float> row = logits;
  if (penalties) {
    scratch.logits.assign(logits.begin(), logits.end());
    if (core::Status status = detail::ApplyPenalties(
            scratch.logits, context.prompt_ids, context.generated_ids,
            params.repetition_penalty, params.presence_penalty,
            params.frequency_penalty);
        !status.ok()) {
      return status;
    }
    row = scratch.logits;
  }
  core::StatusOr<std::int32_t> token = ArgmaxRow(row);
  if (!token.ok()) {
    return token.status();
  }
  SampleResult result{.token = *token, .logprobs = std::nullopt};
  if (want_logprobs) {
    // Reject a non-finite row (NaN/+inf) before the log-sum-exp — the greedy
    // fast path only guards NaN-as-max; logprobs demand a well-formed softmax.
    if (core::Status status = detail::CheckFinite(row); !status.ok()) {
      return status;
    }
    detail::LogSoftmax(row, scratch.lp, scratch.exp_scratch);
    result.logprobs =
        detail::ExtractLogprobs(scratch.lp, *token, params.logprobs);
  }
  return result;
}

// The stochastic branch of `SampleRow`: penalties -> temperature -> top-k ->
// top-p -> Philox draw. `step` is the decode step index (the RNG counter),
// `seed` the resolved request seed; `context` carries the token history the
// penalties read. When `want_logprobs`, the log-softmax is captured over the
// post-penalty/post-temperature logits *before* top-k masks them, so it covers
// the whole vocabulary; the RNG draw is unaffected, so the chosen token matches
// the token-only path.
[[nodiscard]] core::StatusOr<SampleResult> SampleStochastic(
    std::span<const float> logits, const SampleContext& context,
    const SamplingParams& params, std::uint64_t seed, std::uint64_t step,
    bool want_logprobs, detail::RowScratch& scratch) {
  if (logits.empty()) {
    return core::InvalidArgumentError("logits row must be non-empty");
  }
  if (core::Status status = detail::CheckFinite(logits); !status.ok()) {
    return status;
  }
  // Work on a mutable copy — the caller's logits buffer is const, and the
  // stages mask/scale in place.
  scratch.logits.assign(logits.begin(), logits.end());
  const std::span<float> work{scratch.logits};
  if (core::Status status = detail::ApplyPenalties(
          work, context.prompt_ids, context.generated_ids,
          params.repetition_penalty, params.presence_penalty,
          params.frequency_penalty);
      !status.ok()) {
    return status;
  }
  detail::ApplyTemperature(work, params.temperature);
  if (want_logprobs) {
    // Before top-k masking — full-vocab logprobs (§15.2 stage 6).
    detail::LogSoftmax(work, scratch.lp, scratch.exp_scratch);
  }
  detail::ApplyTopK(work, params.top_k);
  detail::Softmax(work, scratch.probs, scratch.exp_scratch);
  detail::ApplyTopP(scratch.probs, params.top_p, scratch.top_p_order);
  const double u = PhiloxUniformDouble(seed, step, /*draw=*/0U);
  const std::int32_t token = detail::SelectByCdf(scratch.probs, u);
  SampleResult result{.token = token, .logprobs = std::nullopt};
  if (want_logprobs) {
    result.logprobs =
        detail::ExtractLogprobs(scratch.lp, token, params.logprobs);
  }
  return result;
}

}  // namespace

namespace detail {

core::StatusOr<SampleResult> SampleRow(std::span<const float> logits,
                                       const SampleContext& context,
                                       const SamplingParams& params,
                                       std::uint64_t seed, bool want_logprobs,
                                       RowScratch& scratch) {
  if (params.temperature == 0.0F) {
    return SampleGreedy(logits, context, params, want_logprobs, scratch);
  }
  // The step index the RNG keys on: the number of tokens produced so far this
  // request (0 for the first sampled token). Independent of batch composition,
  // so the draw is reproducible across runs and batch layouts.
  const auto step = static_cast<std::uint64_t>(context.generated_ids.size());
  return SampleStochastic(logits, context, params, seed, step, want_logprobs,
                          scratch);
}

}  // namespace detail

core::StatusOr<Sampler> Sampler::Create(SamplingParams params) {
  if (core::Status status = ValidateSamplingParams(params); !status.ok()) {
    return status;
  }
  const std::uint64_t seed = ResolveSeed(params);
  return Sampler(std::move(params), seed);
}

core::StatusOr<std::int32_t> Sampler::Sample(
    std::span<const float> logits, const SampleContext& context) const {
  core::StatusOr<SampleResult> result =
      SampleWithLogprobsImpl(logits, context, /*want_logprobs=*/false);
  if (!result.ok()) {
    return result.status();
  }
  return result->token;
}

core::StatusOr<SampleResult> Sampler::SampleWithLogprobs(
    std::span<const float> logits, const SampleContext& context) const {
  return SampleWithLogprobsImpl(logits, context,
                                /*want_logprobs=*/params_.logprobs > 0);
}

core::StatusOr<SampleResult> Sampler::SampleWithLogprobsImpl(
    std::span<const float> logits, const SampleContext& context,
    bool want_logprobs) const {
  // The reference sampler allocates a throwaway scratch per call; the batched
  // sampler (T06) reuses one `RowScratch` per sequence across steps.
  detail::RowScratch scratch;
  return detail::SampleRow(logits, context, params_, seed_, want_logprobs,
                           scratch);
}

}  // namespace engine::sampling
