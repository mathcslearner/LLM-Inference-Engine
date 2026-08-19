#include "sampling/stages.h"

#include "core/status.h"

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

void Softmax(std::span<const float> logits, std::vector<double>& probs) {
  probs.assign(logits.size(), 0.0);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (const float x : logits) {
    max_logit = std::max(max_logit, x);
  }
  // `max_logit` is finite (a precondition), so `x - max_logit` is `-inf` only
  // for masked (`-inf`) logits, and `exp(-inf) == 0` exactly.
  double sum = 0.0;
  for (std::size_t v = 0; v < logits.size(); ++v) {
    const double w = std::exp(static_cast<double>(logits[v]) -
                              static_cast<double>(max_logit));
    probs[v] = w;
    sum += w;
  }
  const double inv = 1.0 / sum;
  for (double& p : probs) {
    p *= inv;
  }
}

void ApplyTopP(std::vector<double>& probs, float top_p) {
  if (top_p >= 1.0F) {
    return;  // nucleus keeps everything; skip to avoid any rounding drop
  }
  // Order tokens by descending probability, breaking ties by ascending index so
  // the kept set is deterministic.
  std::vector<std::int32_t> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, [&](std::int32_t a, std::int32_t b) {
    if (probs[static_cast<std::size_t>(a)] !=
        probs[static_cast<std::size_t>(b)]) {
      return probs[static_cast<std::size_t>(a)] >
             probs[static_cast<std::size_t>(b)];
    }
    return a < b;
  });
  // Walk the sorted order accumulating mass; the token that reaches `top_p` is
  // the last one kept. Everything after it is zeroed.
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

}  // namespace engine::sampling::detail
