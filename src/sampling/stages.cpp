#include "sampling/stages.h"

#include "core/status.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
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
