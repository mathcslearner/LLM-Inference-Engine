#pragma once

#include "core/status.h"
#include "sampling/logprobs.h"
#include "sampling/params.h"

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// The sampling pipeline entry point (M7; design: docs/design/model-execution.md
// §15). `Sampler` turns a single position's raw logits into a chosen token id,
// running the documented stage order: penalties → temperature → top-k → top-p →
// selection (greedy argmax when `temperature == 0`, else a seeded categorical
// draw). It is the single-sequence *reference* sampler; the batched optimized
// sampler (T06) must match it token-for-token given identical RNG counters.
//
// M7-T02 added the stochastic branch (temperature scaling, top-k/top-p
// filtering, a counter-based Philox draw keyed on `(seed, step)`); M7-T03 added
// stage 1, the history-based repetition/frequency/presence penalties, applied
// before temperature and ahead of the greedy argmax alike; M7-T05 adds stage 6,
// logprobs (`SampleWithLogprobs`). Stop conditions (`stop_token_ids`/
// `stop_strings`, M7-T04) are the generation loop's `StopChecker`, not the
// sampler's — `Create` accepts and ignores them. As of T05 every
// `SamplingParams` field is implemented, so `Create` no longer rejects any
// knob with `Unimplemented`: it validates and returns.

namespace engine::sampling {

// The result of a sampling step: the chosen token id, plus (when the request
// asked for logprobs) the per-step logprob information. `logprobs` is
// `std::nullopt` when `params.logprobs == 0` — the token-only path pays nothing
// for the feature. `Sample` returns just the id; `SampleWithLogprobs` returns
// this struct.
struct SampleResult {
  std::int32_t token = 0;
  std::optional<StepLogprobs> logprobs;
};

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
  // `ValidateSamplingParams`. As of M7-T05 every field is implemented, so
  // `Create` no longer returns `Unimplemented`. When `params.seed` is `nullopt`
  // a nondeterministic seed is drawn once here and exposed via `seed()`.
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
  // carries no mutable state. This is the token-only path — logprobs are never
  // computed regardless of `params().logprobs`; use `SampleWithLogprobs` for
  // those.
  [[nodiscard]] core::StatusOr<std::int32_t> Sample(
      std::span<const float> logits, const SampleContext& context) const;

  // Like `Sample`, but the returned `SampleResult` also carries the per-step
  // logprobs when `params().logprobs > 0` (else `SampleResult::logprobs` is
  // `std::nullopt`). The chosen token is identical to what `Sample` returns for
  // the same `(logits, context)` — computing logprobs never perturbs selection
  // (the RNG draw is unchanged). Logprobs are the natural-log softmax of the
  // post-penalty/post-temperature logits, taken before top-k/top-p masking
  // (§15.2 stage 6, docs on `StepLogprobs`). Same error posture as `Sample`.
  [[nodiscard]] core::StatusOr<SampleResult> SampleWithLogprobs(
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

  // The shared core behind `Sample` (`want_logprobs == false`) and
  // `SampleWithLogprobs` (`want_logprobs == params().logprobs > 0`): dispatch
  // on the selection mode and, when asked, attach the per-step logprobs.
  [[nodiscard]] core::StatusOr<SampleResult> SampleWithLogprobsImpl(
      std::span<const float> logits, const SampleContext& context,
      bool want_logprobs) const;

  SamplingParams params_;
  std::uint64_t seed_;
};

namespace detail {

// Reusable per-row scratch for `SampleRow` (M7-T06). Holding these buffers on
// the caller lets the batched sampler run allocation-free in steady state: it
// owns one `RowScratch` per batch slot and reuses it across decode steps
// (grow-on-demand — `std::vector::assign` keeps capacity). The reference
// `Sampler` allocates a throwaway `RowScratch` per call.
struct RowScratch {
  std::vector<float> logits;       // mutable copy of the row (penalties/temp)
  std::vector<double> probs;       // softmax distribution (draw)
  std::vector<double> lp;          // log-softmax (logprobs)
  std::vector<float> exp_scratch;  // fp32 staging for the vector exp
  std::vector<std::int32_t> top_p_order;  // top-p sort order (positives only)
};

// The single per-row sampling pipeline shared by the reference `Sampler` and
// the T06 `BatchedSampler` — one arithmetic implementation, so the batched
// path picks identical tokens by construction (the acceptance criterion).
// Dispatches on `params.temperature`: `== 0` is greedy argmax over the
// (penalized) row, `> 0` is penalties → temperature → top-k → softmax → top-p
// → a Philox inverse-CDF draw keyed on `(seed, step)` where the step index is
// `context.generated_ids.size()`. When `want_logprobs`, the returned
// `SampleResult::logprobs` carries the full-vocabulary log-softmax (§15.2
// stage 6), taken before top-k/top-p masking; the RNG draw is unaffected, so
// the chosen token matches the no-logprobs path. Same error posture as
// `Sampler::Sample` (empty row → InvalidArgument; NaN/+inf → Internal; a bad
// history id → InvalidArgument, inputs untouched). `scratch` supplies the
// working buffers.
[[nodiscard]] core::StatusOr<SampleResult> SampleRow(
    std::span<const float> logits, const SampleContext& context,
    const SamplingParams& params, std::uint64_t seed, bool want_logprobs,
    RowScratch& scratch);

}  // namespace detail

}  // namespace engine::sampling
