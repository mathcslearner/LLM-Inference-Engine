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

// M7-T03: the sampler with penalties wired into both selection branches. The
// history-based penalties (repetition/frequency/presence) are stage 1, applied
// to the raw logits *before* temperature (design §15.2) — and ahead of the
// greedy argmax too, so a penalty can move the chosen token. `temperature == 0`
// is greedy argmax (over the penalized logits); `temperature > 0` runs the full
// stochastic pipeline (penalties -> temperature -> top-k -> top-p -> Philox
// categorical draw). The stop-condition (T04) and logprobs (T05) stages are
// still guarded off at `Create` so a requested knob is never silently dropped.

namespace engine::sampling {
namespace {

// True when `params` stays within the stages implemented through M7-T03: any
// temperature, top-k, top-p and the three penalties (all handled by `Sample`),
// but no stop machinery (T04 owns those in the loop) or logprobs (T05).
// `ValidateSamplingParams` has already run, so the values are in range here.
[[nodiscard]] core::Status CheckImplementedSubset(
    const SamplingParams& params) {
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

// True when any history penalty is non-default, i.e. `ApplyPenalties` would
// change the logits. Used to keep the default path allocation-free and its
// result bitwise identical to the pre-T03 sampler.
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

// The stochastic branch: penalties -> temperature -> top-k -> top-p -> Philox
// draw. `step` is the decode step index (the RNG counter), `seed` the resolved
// request seed; `context` carries the token history the penalties read.
[[nodiscard]] core::StatusOr<std::int32_t> SampleStochastic(
    std::span<const float> logits, const SampleContext& context,
    const SamplingParams& params, std::uint64_t seed, std::uint64_t step) {
  if (logits.empty()) {
    return core::InvalidArgumentError("logits row must be non-empty");
  }
  if (core::Status status = detail::CheckFinite(logits); !status.ok()) {
    return status;
  }
  // Work on a mutable copy — the caller's logits buffer is const, and the
  // stages mask/scale in place.
  std::vector<float> scratch(logits.begin(), logits.end());
  const std::span<float> work{scratch};
  if (core::Status status = detail::ApplyPenalties(
          work, context.prompt_ids, context.generated_ids,
          params.repetition_penalty, params.presence_penalty,
          params.frequency_penalty);
      !status.ok()) {
    return status;
  }
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
    // Greedy: penalties (stage 1) can still move the argmax. When no penalty is
    // active this stays the allocation-free, bitwise-unchanged pre-T03 path.
    if (!PenaltiesActive(params_)) {
      return ArgmaxRow(logits);
    }
    if (logits.empty()) {
      return core::InvalidArgumentError("logits row must be non-empty");
    }
    std::vector<float> scratch(logits.begin(), logits.end());
    if (core::Status status = detail::ApplyPenalties(
            scratch, context.prompt_ids, context.generated_ids,
            params_.repetition_penalty, params_.presence_penalty,
            params_.frequency_penalty);
        !status.ok()) {
      return status;
    }
    return ArgmaxRow(scratch);
  }
  // The step index the RNG keys on: the number of tokens produced so far this
  // request (0 for the first sampled token). Independent of batch composition,
  // so the draw is reproducible across runs and batch layouts.
  const auto step = static_cast<std::uint64_t>(context.generated_ids.size());
  return SampleStochastic(logits, context, params_, seed_, step);
}

}  // namespace engine::sampling
