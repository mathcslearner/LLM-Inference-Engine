#pragma once

#include "core/status.h"
#include "sampling/logprobs.h"

#include <cstdint>
#include <span>
#include <vector>

// The individual stochastic-sampling stages (M7-T02; design:
// docs/design/model-execution.md §15.2), factored out of `Sampler::Sample` so
// each stage's exact behaviour is unit-testable in isolation (the acceptance
// criterion: "top-k/top-p masks verified exactly") and so the T06 batched
// sampler can be validated against the same reference arithmetic. All stages
// operate on a single position's `[V]` fp32 logits row. The fixed order the
// sampler runs them in is:
//
//   CheckFinite -> ApplyPenalties -> ApplyTemperature -> ApplyTopK -> Softmax
//   -> ApplyTopP -> SelectByCdf
//
// Stage 6 (logprobs, T05) is `LogSoftmax` + `ExtractLogprobs`, run over the
// post-penalty/post-temperature logits *before* `ApplyTopK` masks them (§15.2:
// logprobs describe the full-vocabulary distribution, not the truncated
// nucleus). `ApplyPenalties` (stage 1, T03) runs ahead of the greedy branch
// too, so a penalty can change the argmax.
//
// Numeric contract (documented so the T06 optimized path can match token-for-
// token given the same RNG draw): the exponential is the shared vector
// `kernels::ExpF32` polynomial (M7-T06 promoted it to a public unthreaded
// entry; kernels/exp.h) applied to the fp32 `logits - max`, and the resulting
// weights are summed in `double` in ascending vocab order; `-inf` logits (the
// top-k mask value) map to exactly `0.0` (the polynomial flushes `x < kExpLo`
// to `+0.0`). Both the single-sequence reference `Sampler` and the T06 batched
// sampler call these same functions, so they pick identical tokens by
// construction. Using the shared polynomial (rather than `std::exp`) makes a
// draw bit-identical across libm implementations *for a fixed ISA*; it is still
// Class T across ISAs (scalar vs NEON vs AVX2 differ by ≤2 ulp), so the full
// cross-platform determinism contract remains M17-T04's. `sampling → kernels`
// is a layer-2 → layer-1 edge ADR-002 already permits (as `model → kernels`
// is) — the earlier "sampling cannot link kernels" note this file carried was
// mistaken.

namespace engine::sampling::detail {

// Reject a logits row that cannot yield a valid distribution: any `NaN` or
// `+inf` entry, or an all-`-inf` row, is `Internal` (an upstream numerics bug,
// not a bad request). `-inf` entries are allowed — they are masked-out tokens —
// as long as at least one finite logit remains. An empty row is the caller's
// (`InvalidArgument`) concern and is not checked here.
[[nodiscard]] core::Status CheckFinite(std::span<const float> logits);

// Stage 1 (T03): apply the three OpenAI/vLLM repetition penalties in place,
// over the request's token history, in the fixed order repetition -> frequency
// -> presence (matching vLLM's `apply_penalties`). Documented history choice:
//   - `repetition_penalty` (HF): every token that appears in the prompt *or*
//   the
//     generated output is penalized once — `x < 0 ? x*r : x/r` (a token seen
//     twice is not compounded).
//   - `frequency_penalty` / `presence_penalty` (OpenAI): only *generated*
//   tokens
//     count (a token that occurs solely in the prompt is untouched). Frequency
//     subtracts `f * occurrences`; presence subtracts a flat `p` once.
// An exact no-op — the caller's logits are left bitwise unchanged — when all
// three are at their defaults (`repetition_penalty == 1`, the others `== 0`),
// so the default pipeline is byte-identical to the pre-T03 one. `-inf` logits
// stay
// `-inf` under every operation. A history id outside `[0, logits.size())` is an
// `InvalidArgument` naming the offending index and value, front-loaded so a
// rejected call leaves the logits untouched.
[[nodiscard]] core::Status ApplyPenalties(
    std::span<float> logits, std::span<const std::int32_t> prompt_ids,
    std::span<const std::int32_t> generated_ids, float repetition_penalty,
    float presence_penalty, float frequency_penalty);

// Divide every logit by `temperature` in place. Precondition: `temperature > 0`
// (the `== 0` greedy path never reaches this stage).
void ApplyTemperature(std::span<float> logits, float temperature);

// Top-k filter (HF/vLLM threshold semantics): keep the `k` highest logits by
// masking every logit strictly below the k-th largest value to `-inf`. Ties at
// the threshold are kept, so more than `k` tokens may survive — this is
// deterministic and matches `torch.topk`'s cutoff. No-op when `k <= 0` or
// `k >= logits.size()` (the filter is disabled / keeps everything).
void ApplyTopK(std::span<float> logits, std::int32_t k);

// Numerically stable softmax of `logits` into `probs` (resized to match):
// `probs[v] = exp(logits[v] - max) / Z`, with `-inf` logits mapping to exactly
// `0.0`. The exponential is `kernels::ExpF32` over the fp32 `logits - max`
// (staged through `exp_scratch`, resized to match); `Z` is accumulated in
// `double` in ascending index order. Precondition: at least one finite logit
// (guaranteed by `CheckFinite` + top-k keeping the max). The `exp_scratch`
// overload lets the batched sampler reuse one buffer across steps
// (allocation-free); the two-argument form allocates a local scratch.
void Softmax(std::span<const float> logits, std::vector<double>& probs,
             std::vector<float>& exp_scratch);
void Softmax(std::span<const float> logits, std::vector<double>& probs);

// Top-p (nucleus) filter over a probability vector: keep the smallest set of
// highest-probability tokens whose cumulative probability reaches `top_p`,
// zeroing the rest. Ties are ordered by ascending vocab index; the token that
// crosses the threshold is included, and at least one token is always kept.
// No-op when `top_p >= 1.0` (keeps everything — and avoids any rounding that
// could drop a tail token). Only the positive-probability tokens are sorted
// (zeros can never enter the nucleus), so a post-top-k row sorts just its `k`
// survivors rather than the whole vocabulary; `order` is reusable scratch (the
// batched path passes an owned buffer, the two-argument form allocates one).
void ApplyTopP(std::vector<double>& probs, float top_p,
               std::vector<std::int32_t>& order);
void ApplyTopP(std::vector<double>& probs, float top_p);

// Inverse-CDF categorical draw from a (not necessarily normalized) probability
// vector and a uniform `u` in [0, 1): return the first ascending index whose
// running sum exceeds `u * total`, where `total` is the sum of `probs`. Falls
// back to the last positive-probability index (guards `u` rounding to the top
// of the mass). Precondition: `probs` has at least one positive entry.
[[nodiscard]] std::int32_t SelectByCdf(const std::vector<double>& probs,
                                       double u);

// Stage 6a (T05): natural-log softmax of `logits` into `lp` (resized to match):
// `lp[v] = (logits[v] - max) - log(Σ_u exp(logits[u] - max))`, the log-sum-exp
// accumulated in `double` in ascending index order. A `-inf` logit maps to
// exactly `-inf` (log-prob 0); `max` is finite by precondition (`CheckFinite`),
// so `exp(logits[v] - max)` is well defined. Fed the same post-penalty/
// post-temperature logits the draw uses, but taken before top-k/top-p masking
// so the reported distribution covers the whole vocabulary and `Σ_v exp(lp[v])`
// is `1` (the acceptance criterion). The exponentials use the shared
// `kernels::ExpF32` polynomial (staged through `exp_scratch`), matching the
// draw's softmax; `std::log` computes the normalizer. The `exp_scratch`
// overload is allocation-free for the batched path; the two-argument form
// allocates locally.
void LogSoftmax(std::span<const float> logits, std::vector<double>& lp,
                std::vector<float>& exp_scratch);
void LogSoftmax(std::span<const float> logits, std::vector<double>& lp);

// Stage 6b (T05): assemble a `StepLogprobs` from precomputed log-probs `lp`
// (from `LogSoftmax`), the chosen token id, and the requested count `top_n`.
// `chosen_logprob = lp[chosen]`; `top` holds the `min(top_n, #finite)` largest
// log-probs ordered by descending value with ascending-id tie-break (a partial
// sort — `top_n <= kMaxLogprobs`). `-inf` (masked) entries are excluded from
// `top`. Precondition: `chosen` is in range and finite (selection guarantees
// it), `top_n >= 0`.
[[nodiscard]] StepLogprobs ExtractLogprobs(const std::vector<double>& lp,
                                           std::int32_t chosen,
                                           std::int32_t top_n);

}  // namespace engine::sampling::detail
