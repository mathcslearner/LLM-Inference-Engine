#include "sampling/sampler.h"

#include "core/status.h"
#include "sampling/params.h"
#include "sampling/philox.h"
#include "sampling/stages.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

// M7-T02: the sampler with both selection branches. `temperature == 0` is
// greedy argmax (unchanged from T01); `temperature > 0` runs the stochastic
// pipeline (temperature -> top-k -> top-p -> Philox categorical draw, design
// §15.2). The penalty (T03), stop-condition (T04) and logprobs (T05) stages are
// still guarded off at `Create` so a requested knob is never silently dropped.

namespace engine::sampling {
namespace {

// True when `params` stays within the stages M7-T02 implements: any
// temperature, top-k and top-p (all handled by `Sample`), but no penalties,
// stop machinery (T04 owns those in the loop), or logprobs.
// `ValidateSamplingParams` has already run, so the values are in range here.
[[nodiscard]] core::Status CheckImplementedSubset(
    const SamplingParams& params) {
  if (params.repetition_penalty != 1.0F) {
    return core::UnimplementedError(
        "repetition_penalty is not implemented until M7-T03; got {}",
        params.repetition_penalty);
  }
  if (params.presence_penalty != 0.0F) {
    return core::UnimplementedError(
        "presence_penalty is not implemented until M7-T03; got {}",
        params.presence_penalty);
  }
  if (params.frequency_penalty != 0.0F) {
    return core::UnimplementedError(
        "frequency_penalty is not implemented until M7-T03; got {}",
        params.frequency_penalty);
  }
  if (!params.stop_token_ids.empty()) {
    return core::UnimplementedError(
        "stop_token_ids are not implemented until M7-T04");
  }
  if (!params.stop_strings.empty()) {
    return core::UnimplementedError(
        "stop_strings are not implemented until M7-T04");
  }
  if (params.logprobs != 0) {
    return core::UnimplementedError(
        "logprobs are not implemented until M7-T05; got {}", params.logprobs);
  }
  return core::OkStatus();
}

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

// The stochastic branch: temperature -> top-k -> top-p -> Philox draw. `step`
// is the decode step index (the RNG counter), `seed` the resolved request seed.
[[nodiscard]] core::StatusOr<std::int32_t> SampleStochastic(
    std::span<const float> logits, const SamplingParams& params,
    std::uint64_t seed, std::uint64_t step) {
  if (logits.empty()) {
    return core::InvalidArgumentError("logits row must be non-empty");
  }
  if (core::Status status = detail::CheckFinite(logits); !status.ok()) {
    return status;
  }
  // Work on a mutable copy — the caller's logits buffer is const, and top-k
  // masks in place.
  std::vector<float> scratch(logits.begin(), logits.end());
  const std::span<float> work{scratch};
  detail::ApplyTemperature(work, params.temperature);
  detail::ApplyTopK(work, params.top_k);
  std::vector<double> probs;
  detail::Softmax(work, probs);
  detail::ApplyTopP(probs, params.top_p);
  const double u = PhiloxUniformDouble(seed, step, /*draw=*/0U);
  return detail::SelectByCdf(probs, u);
}

}  // namespace

core::StatusOr<Sampler> Sampler::Create(SamplingParams params) {
  if (core::Status status = ValidateSamplingParams(params); !status.ok()) {
    return status;
  }
  if (core::Status status = CheckImplementedSubset(params); !status.ok()) {
    return status;
  }
  const std::uint64_t seed = ResolveSeed(params);
  return Sampler(std::move(params), seed);
}

core::StatusOr<std::int32_t> Sampler::Sample(
    std::span<const float> logits, const SampleContext& context) const {
  if (params_.temperature == 0.0F) {
    return ArgmaxRow(logits);
  }
  // The step index the RNG keys on: the number of tokens produced so far this
  // request (0 for the first sampled token). Independent of batch composition,
  // so the draw is reproducible across runs and batch layouts.
  const auto step = static_cast<std::uint64_t>(context.generated_ids.size());
  return SampleStochastic(logits, params_, seed_, step);
}

}  // namespace engine::sampling
