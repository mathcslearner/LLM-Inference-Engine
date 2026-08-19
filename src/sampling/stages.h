#pragma once

#include "core/status.h"

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
//   CheckFinite -> ApplyTemperature -> ApplyTopK -> Softmax -> ApplyTopP ->
//   SelectByCdf
//
// (Penalties are stage 1, added in T03; logprobs are stage 6, added in T05.)
//
// Numeric contract (documented so the T06 optimized path can match token-for-
// token given the same RNG draw): softmax is max-subtracted with the
// exponential sum accumulated in `double` in ascending vocab order; `-inf`
// logits (the top-k mask value) map to exactly `0.0`. Sampled *token sequences*
// are therefore reproducible on a given machine but not bit-identical across
// platforms, since `std::exp` differs by ulps between libm implementations —
// the cross-platform determinism contract is M17-T04's, and `sampling` cannot
// link the `kernels` module's shared exp polynomial (ADR-002).

namespace engine::sampling::detail {

// Reject a logits row that cannot yield a valid distribution: any `NaN` or
// `+inf` entry, or an all-`-inf` row, is `Internal` (an upstream numerics bug,
// not a bad request). `-inf` entries are allowed — they are masked-out tokens —
// as long as at least one finite logit remains. An empty row is the caller's
// (`InvalidArgument`) concern and is not checked here.
[[nodiscard]] core::Status CheckFinite(std::span<const float> logits);

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
// `0.0`. `Z` is accumulated in `double` in ascending index order. Precondition:
// at least one finite logit (guaranteed by `CheckFinite` + top-k keeping the
// max).
void Softmax(std::span<const float> logits, std::vector<double>& probs);

// Top-p (nucleus) filter over a probability vector: keep the smallest set of
// highest-probability tokens whose cumulative probability reaches `top_p`,
// zeroing the rest. Ties are ordered by ascending vocab index; the token that
// crosses the threshold is included, and at least one token is always kept.
// No-op when `top_p >= 1.0` (keeps everything — and avoids any rounding that
// could drop a tail token).
void ApplyTopP(std::vector<double>& probs, float top_p);

// Inverse-CDF categorical draw from a (not necessarily normalized) probability
// vector and a uniform `u` in [0, 1): return the first ascending index whose
// running sum exceeds `u * total`, where `total` is the sum of `probs`. Falls
// back to the last positive-probability index (guards `u` rounding to the top
// of the mass). Precondition: `probs` has at least one positive entry.
[[nodiscard]] std::int32_t SelectByCdf(const std::vector<double>& probs,
                                       double u);

}  // namespace engine::sampling::detail
