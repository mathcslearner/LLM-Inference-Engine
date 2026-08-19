#pragma once

#include <cstdint>
#include <vector>

// Per-step logprobs (M7-T05; design: docs/design/model-execution.md §15.2
// stage 6). The sampler can return, alongside the chosen token, the log
// probability it assigned that token and the top-N `(id, logprob)` pairs — the
// data the OpenAI API exposes as `logprobs` / `top_logprobs`.
//
// Documented semantics (the "which logits" choice, §15.2): logprobs are the
// natural-log softmax of the logits the sampler saw **after** the penalty
// (stage 1) and temperature (stage 2) stages but **before** the top-k/top-p
// truncation filters — i.e. the model's full (temperature-scaled) belief over
// the whole vocabulary, not the renormalized truncated nucleus. The greedy
// branch (`temperature == 0`) uses the post-penalty logits with no scaling. So
// the returned distribution always covers the entire vocabulary and its
// probabilities sum to 1; with top-k/top-p active, a reported token's logprob
// can be non-zero even though the filter would never let it be sampled. This
// matches OpenAI's "logprobs describe the model, the filters describe the
// draw" convention. Values are computed in `double` and narrowed to `float`
// (the exposed precision); `sampling` cannot link the `kernels` exp polynomial
// (ADR-002), so cross-platform bit-identity is M17-T04's concern, as for the
// draw itself.

namespace engine::sampling {

// One `(token id, log probability)` pair. `logprob` is a natural log in
// `(-inf, 0]` — masked (`-inf`-logit) tokens are excluded from `top`, so a
// listed entry is always finite.
struct TokenLogprob {
  std::int32_t id = 0;
  float logprob = 0.0F;
};

// The logprob information for one produced token.
struct StepLogprobs {
  // The natural-log probability the sampler assigned the token it returned
  // (`Sample`'s `token`). Always finite (selection never picks a masked token).
  float chosen_logprob = 0.0F;
  // The highest-probability tokens, ordered by descending logprob with ties
  // broken by ascending vocab id. Holds `min(params.logprobs, #finite-logit
  // tokens)` entries — the chosen token is included when it is among them (it
  // always is for greedy). Empty only if the request asked for zero.
  std::vector<TokenLogprob> top;
};

}  // namespace engine::sampling
