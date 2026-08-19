#include "sampling/sampler.h"

#include "core/status.h"
#include "sampling/params.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

// M7-T01: the sampler skeleton with the greedy selection branch only. The
// stage-by-stage pipeline (penalties → temperature → top-k/p → categorical
// draw) fills in across M7-T02…T05; until then `Create` rejects any request
// outside the greedy subset with `Unimplemented`, so a knob is never silently
// dropped.

namespace engine::sampling {
namespace {

// True when `params` selects the pure argmax path M7-T01 implements: greedy
// temperature, every filter/penalty at its no-op default, no logprobs, and no
// stop-string / stop-token machinery (T04 owns those). `ValidateSamplingParams`
// has already run, so the values are in range here.
[[nodiscard]] core::Status CheckGreedySubset(const SamplingParams& params) {
  if (params.temperature != 0.0F) {
    return core::UnimplementedError(
        "temperature > 0 (stochastic sampling) is not implemented until "
        "M7-T02; got {}",
        params.temperature);
  }
  if (params.top_k != 0) {
    return core::UnimplementedError(
        "top_k filtering is not implemented until M7-T02; got {}",
        params.top_k);
  }
  if (params.top_p != 1.0F) {
    return core::UnimplementedError(
        "top_p filtering is not implemented until M7-T02; got {}",
        params.top_p);
  }
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

}  // namespace

core::StatusOr<Sampler> Sampler::Create(SamplingParams params) {
  if (core::Status status = ValidateSamplingParams(params); !status.ok()) {
    return status;
  }
  if (core::Status status = CheckGreedySubset(params); !status.ok()) {
    return status;
  }
  return Sampler(std::move(params));
}

core::StatusOr<std::int32_t> Sampler::Sample(
    std::span<const float> logits, const SampleContext& context) const {
  // Dispatch on the selection mode — the seam T02 plugs the stochastic path
  // into. `Create` has already guaranteed the greedy subset, so
  // `temperature == 0` is the only reachable branch in T01; the explicit check
  // keeps `Sample` self-consistent and gives T02 its insertion point. The
  // context is unused until penalties (T03) and the seeded RNG (T02) read the
  // token history.
  (void)context;
  if (params_.temperature == 0.0F) {
    return ArgmaxRow(logits);
  }
  return core::UnimplementedError(
      "stochastic sampling (temperature > 0) is not implemented until M7-T02");
}

}  // namespace engine::sampling
