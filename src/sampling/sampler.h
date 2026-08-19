#pragma once

#include "core/status.h"
#include "sampling/params.h"

#include <cstdint>
#include <span>
#include <utility>

// The sampling pipeline entry point (M7; design: docs/design/model-execution.md
// §15). `Sampler` turns a single position's raw logits into a chosen token id,
// running the documented stage order: penalties → temperature → top-k → top-p →
// selection (greedy argmax when `temperature == 0`, else a seeded categorical
// draw). It is the single-sequence *reference* sampler; the batched optimized
// sampler (T06) must match it token-for-token given identical RNG counters.
//
// M7-T02 added the stochastic branch (temperature scaling, top-k/top-p
// filtering, a counter-based Philox draw keyed on `(seed, step)`); M7-T03 adds
// stage 1, the history-based repetition/frequency/presence penalties, applied
// before temperature and ahead of the greedy argmax alike. Any parameter whose
// behaviour is still not implemented (logprobs, stop strings/ids) is rejected
// up front by `Sampler::Create` with `Unimplemented`, so no requested knob is
// ever silently ignored — each later ticket removes its guard as it lands the
// stage.

namespace engine::sampling {

// Per-call context for `Sample`: the request's token history, which the
// penalty stage (T03) reads and whose length is the step index the seeded RNG
// (T02) keys on. The engine already owns these vectors, so the sampler stays
// history-stateless and cheap to construct per request.
struct SampleContext {
  // The prompt token ids (prefill input).
  std::span<const std::int32_t> prompt_ids;
  // The tokens generated so far this request, in order. Its size is the current
  // decode step index (0 for the first sampled token).
  std::span<const std::int32_t> generated_ids;
};

// A configured, validated sampler for one request's `SamplingParams`. Holds the
// params and the resolved RNG seed — cheap to construct per sequence and safe
// to reuse across steps of the same request. No mutable RNG stream state: the
// stochastic draw is a pure function of `(seed, step)` (the Philox counter is
// derived from `context.generated_ids.size()`), which is what keeps sampling
// reproducible per `(seed, step)`, independent of batch composition, and
// `Sample` `const`.
class Sampler {
 public:
  // Build a sampler from `params`. Returns `InvalidArgument` if the params fail
  // `ValidateSamplingParams`, or `Unimplemented` (naming the field) for a
  // parameter whose stage is not yet implemented (logprobs, stop conditions).
  // On success the returned sampler's `Sample` is guaranteed to be within the
  // implemented subset. When `params.seed` is `nullopt` a nondeterministic seed
  // is drawn once here and exposed via `seed()`.
  [[nodiscard]] static core::StatusOr<Sampler> Create(SamplingParams params);

  // Select the next token id from one position's `[V]` fp32 logits row.
  //
  // `temperature == 0` is the greedy branch: argmax with a strict `>` scan so
  // the lowest vocab index wins ties (deterministic — two identical requests
  // are bit-identical). Otherwise the stochastic branch runs temperature ->
  // top-k -> top-p -> a Philox categorical draw keyed on the seed and the step
  // index (`context.generated_ids.size()`). Non-finite logits (a `NaN`/`+inf`,
  // i.e. an upstream numerics bug) surface as `Internal` rather than a bogus
  // token; an empty `logits` row is `InvalidArgument`. `const`: the draw
  // carries no mutable state.
  [[nodiscard]] core::StatusOr<std::int32_t> Sample(
      std::span<const float> logits, const SampleContext& context) const;

  // The params this sampler was built from (the engine reads `max_tokens`,
  // stop conditions, etc.).
  [[nodiscard]] const SamplingParams& params() const { return params_; }

  // The RNG seed this sampler samples with: `*params().seed` when the request
  // supplied one, else the nondeterministic seed drawn at `Create` (so the
  // caller — e.g. the M10 API layer — can echo/log the seed actually used).
  [[nodiscard]] std::uint64_t seed() const { return seed_; }

 private:
  Sampler(SamplingParams params, std::uint64_t seed)
      : params_(std::move(params)), seed_(seed) {}

  SamplingParams params_;
  std::uint64_t seed_;
};

}  // namespace engine::sampling
