#include "sampling/stages.h"

#include "core/status.h"
#include "kernels/exp.h"
#include "sampling/logprobs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// M7-T02: the stochastic-sampling stages. See stages.h for the stage order and
// the numeric contract these implementations honour.

namespace engine::sampling::detail {

core::Status CheckFinite(std::span<const float> logits) {
  bool any_finite = false;
  for (std::size_t v = 0; v < logits.size(); ++v) {
    const float x = logits[v];
    if (std::isnan(x)) {
      return core::InternalError("logits[{}] is NaN (upstream numerics bug)",
                                 v);
    }
    if (std::isinf(x) && x > 0.0F) {
      return core::InternalError("logits[{}] is +inf (upstream numerics bug)",
                                 v);
    }
    if (std::isfinite(x)) {
      any_finite = true;
    }
  }
  if (!any_finite) {
    return core::InternalError("every logit is -inf; no token can be sampled");
  }
  return core::OkStatus();
}

core::Status ApplyPenalties(std::span<float> logits,
                            std::span<const std::int32_t> prompt_ids,
                            std::span<const std::int32_t> generated_ids,
                            float repetition_penalty, float presence_penalty,
                            float frequency_penalty) {
  const bool repetition = repetition_penalty != 1.0F;
  const bool presence = presence_penalty != 0.0F;
  const bool frequency = frequency_penalty != 0.0F;
  if (!repetition && !presence && !frequency) {
    return core::OkStatus();  // exact no-op: logits left bitwise unchanged
  }
  const auto vocab = static_cast<std::int64_t>(logits.size());
  const auto validate = [vocab](std::int32_t id,
                                std::string_view where) -> core::Status {
    if (id < 0 || static_cast<std::int64_t>(id) >= vocab) {
      return core::InvalidArgumentError(
          "{} token id {} out of range for vocab size {}", where, id, vocab);
    }
    return core::OkStatus();
  };

  // Collect the history first (no logit is touched yet) so a bad id leaves the
  // caller's logits untouched. `counts` is over the generated tokens only
  // (frequency/presence); `seen` is prompt ∪ generated (repetition).
  std::unordered_map<std::int32_t, std::int32_t> counts;
  std::unordered_set<std::int32_t> seen;
  for (const std::int32_t id : generated_ids) {
    if (core::Status s = validate(id, "generated"); !s.ok()) {
      return s;
    }
    ++counts[id];
    seen.insert(id);
  }
  for (const std::int32_t id : prompt_ids) {
    if (core::Status s = validate(id, "prompt"); !s.ok()) {
      return s;
    }
    seen.insert(id);
  }

  // Apply in vLLM order: repetition (multiplicative, once per distinct seen
  // token) -> frequency (subtract f * count) -> presence (subtract p once).
  if (repetition) {
    for (const std::int32_t id : seen) {
      float& x = logits[static_cast<std::size_t>(id)];
      x = x < 0.0F ? x * repetition_penalty : x / repetition_penalty;
    }
  }
  if (frequency) {
    for (const auto& [id, count] : counts) {
      logits[static_cast<std::size_t>(id)] -=
          frequency_penalty * static_cast<float>(count);
    }
  }
  if (presence) {
    for (const auto& entry : counts) {  // one flat subtraction per distinct id
      logits[static_cast<std::size_t>(entry.first)] -= presence_penalty;
    }
  }
  return core::OkStatus();
}

void ApplyTemperature(std::span<float> logits, float temperature) {
  for (float& x : logits) {
    x /= temperature;
  }
}

void ApplyTopK(std::span<float> logits, std::int32_t k) {
  const auto vocab = static_cast<std::int64_t>(logits.size());
  if (k <= 0 || static_cast<std::int64_t>(k) >= vocab) {
    return;  // filter disabled or keeps the whole vocabulary
  }
  // The threshold is the k-th largest logit; `nth_element` on a copy places it
  // at index k-1 under descending order without a full sort.
  std::vector<float> scratch(logits.begin(), logits.end());
  const auto kth = scratch.begin() + (static_cast<std::ptrdiff_t>(k) - 1);
  std::nth_element(scratch.begin(), kth, scratch.end(), std::greater<>());
  const float threshold = *kth;
  // Keep everything >= threshold (ties at the boundary survive), mask the rest.
  for (float& x : logits) {
    if (x < threshold) {
      x = -std::numeric_limits<float>::infinity();
    }
  }
}

void Softmax(std::span<const float> logits, std::vector<double>& probs,
             std::vector<float>& exp_scratch) {
  const std::size_t n = logits.size();
  probs.assign(n, 0.0);
  exp_scratch.assign(n, 0.0F);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (const float x : logits) {
    max_logit = std::max(max_logit, x);
  }
  // `max_logit` is finite (a precondition), so `x - max_logit` is `-inf` only
  // for masked (`-inf`) logits, which `ExpF32` flushes to exactly `0.0` (the
  // top-k mask contract). The exponential is the shared vector polynomial so
  // the reference and batched samplers agree bit-for-bit (stages.h).
  for (std::size_t v = 0; v < n; ++v) {
    exp_scratch[v] = logits[v] - max_logit;
  }
  kernels::ExpF32(exp_scratch.data(), exp_scratch.data(),
                  static_cast<std::int64_t>(n));
  double sum = 0.0;  // ascending-index `double` accumulation (stages.h)
  for (std::size_t v = 0; v < n; ++v) {
    const auto w = static_cast<double>(exp_scratch[v]);
    probs[v] = w;
    sum += w;
  }
  const double inv = 1.0 / sum;
  for (double& p : probs) {
    p *= inv;
  }
}

void Softmax(std::span<const float> logits, std::vector<double>& probs) {
  std::vector<float> exp_scratch;
  Softmax(logits, probs, exp_scratch);
}

void ApplyTopP(std::vector<double>& probs, float top_p,
               std::vector<std::int32_t>& order) {
  if (top_p >= 1.0F) {
    return;  // nucleus keeps everything; skip to avoid any rounding drop
  }
  const std::size_t n = probs.size();
  // Sort only the positive-probability tokens: a zero contributes nothing to
  // the cumulative mass and is already zeroed, so it can never enter the
  // nucleus. After top-k this collapses the sort from the whole vocabulary to
  // just the `k` survivors (the rest are exactly `0` from `exp(-inf)`) — the
  // "partial sort, not full sort" win for the common top-k+top-p path. A row
  // with no top-k (every prob positive) simply sorts them all, matching the old
  // full-sort cost exactly — never worse. Bit-identical to sorting the full
  // vocabulary and walking it: the ordering is a *total* order (ties broken by
  // ascending id), so the positive prefix is unique, and the trailing zeros a
  // full sort would append are all zeroed anyway.
  order.clear();
  for (std::size_t v = 0; v < n; ++v) {
    if (probs[v] > 0.0) {
      order.push_back(static_cast<std::int32_t>(v));
    }
  }
  std::ranges::sort(order, [&probs](std::int32_t a, std::int32_t b) {
    const double pa = probs[static_cast<std::size_t>(a)];
    const double pb = probs[static_cast<std::size_t>(b)];
    if (pa != pb) {
      return pa > pb;
    }
    return a < b;
  });
  // Walk the sorted positives accumulating mass; the token that reaches `top_p`
  // is the last one kept. Everything after it (positive but below the nucleus)
  // is zeroed; the already-zero tokens stay zero.
  const auto threshold = static_cast<double>(top_p);
  double cumulative = 0.0;
  std::size_t kept = 0;
  for (; kept < order.size(); ++kept) {
    cumulative += probs[static_cast<std::size_t>(order[kept])];
    if (cumulative >= threshold) {
      ++kept;  // include the crossing token
      break;
    }
  }
  for (std::size_t rank = kept; rank < order.size(); ++rank) {
    probs[static_cast<std::size_t>(order[rank])] = 0.0;
  }
}

void ApplyTopP(std::vector<double>& probs, float top_p) {
  std::vector<std::int32_t> order;
  ApplyTopP(probs, top_p, order);
}

std::int32_t SelectByCdf(const std::vector<double>& probs, double u) {
  double total = 0.0;
  std::int32_t last_positive = -1;
  for (std::size_t v = 0; v < probs.size(); ++v) {
    total += probs[v];
    if (probs[v] > 0.0) {
      last_positive = static_cast<std::int32_t>(v);
    }
  }
  const double target = u * total;
  double cumulative = 0.0;
  for (std::size_t v = 0; v < probs.size(); ++v) {
    cumulative += probs[v];
    if (cumulative > target && probs[v] > 0.0) {
      return static_cast<std::int32_t>(v);
    }
  }
  // `u` rounded to the very top of the mass; fall back to the last kept token.
  return last_positive;
}

void LogSoftmax(std::span<const float> logits, std::vector<double>& lp,
                std::vector<float>& exp_scratch) {
  const std::size_t n = logits.size();
  lp.assign(n, -std::numeric_limits<double>::infinity());
  exp_scratch.assign(n, 0.0F);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (const float x : logits) {
    max_logit = std::max(max_logit, x);
  }
  // `max_logit` is finite (precondition), so `x - max_logit` is `-inf` only for
  // masked (`-inf`) logits, which `ExpF32` flushes to exactly `0.0` — the
  // log-sum-exp is taken over the finite tokens. The exponential matches the
  // draw's softmax (shared vector polynomial, stages.h).
  for (std::size_t v = 0; v < n; ++v) {
    exp_scratch[v] = logits[v] - max_logit;
  }
  kernels::ExpF32(exp_scratch.data(), exp_scratch.data(),
                  static_cast<std::int64_t>(n));
  double sum = 0.0;  // ascending-index `double` accumulation (stages.h)
  for (std::size_t v = 0; v < n; ++v) {
    sum += static_cast<double>(exp_scratch[v]);
  }
  const double log_z = std::log(sum);
  for (std::size_t v = 0; v < n; ++v) {
    // A `-inf` logit stays `-inf` (its shifted value is `-inf`); every finite
    // logit gets `(x - max) - log_z`.
    if (std::isfinite(logits[v])) {
      lp[v] =
          (static_cast<double>(logits[v]) - static_cast<double>(max_logit)) -
          log_z;
    }
  }
}

void LogSoftmax(std::span<const float> logits, std::vector<double>& lp) {
  std::vector<float> exp_scratch;
  LogSoftmax(logits, lp, exp_scratch);
}

StepLogprobs ExtractLogprobs(const std::vector<double>& lp, std::int32_t chosen,
                             std::int32_t top_n) {
  StepLogprobs out;
  out.chosen_logprob = static_cast<float>(lp[static_cast<std::size_t>(chosen)]);

  const auto want = static_cast<std::size_t>(std::max(top_n, 0));
  const std::size_t n = std::min(want, lp.size());
  if (n == 0) {
    return out;
  }
  // Partial-sort the `n` largest log-probs (descending, ascending-id tie-break)
  // without a full sort of the vocabulary. `-inf` entries sort last, so the
  // prefix is filled with finite tokens first.
  std::vector<std::int32_t> order(lp.size());
  std::iota(order.begin(), order.end(), 0);
  const auto by_logprob = [&lp](std::int32_t a, std::int32_t b) {
    const double la = lp[static_cast<std::size_t>(a)];
    const double lb = lp[static_cast<std::size_t>(b)];
    if (la != lb) {
      return la > lb;
    }
    return a < b;
  };
  std::partial_sort(order.begin(),
                    order.begin() + static_cast<std::ptrdiff_t>(n), order.end(),
                    by_logprob);
  out.top.reserve(n);
  for (std::size_t rank = 0; rank < n; ++rank) {
    const std::int32_t id = order[rank];
    const double value = lp[static_cast<std::size_t>(id)];
    if (!std::isfinite(value)) {
      break;  // remaining entries are masked (`-inf`); exclude them
    }
    out.top.push_back(
        TokenLogprob{.id = id, .logprob = static_cast<float>(value)});
  }
  return out;
}

}  // namespace engine::sampling::detail
