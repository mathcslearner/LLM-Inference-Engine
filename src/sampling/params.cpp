#include "sampling/params.h"

#include "core/status.h"

#include <cmath>
#include <cstddef>

namespace engine::sampling {

core::Status ValidateSamplingParams(const SamplingParams& params) {
  if (!std::isfinite(params.temperature) || params.temperature < 0.0F) {
    return core::InvalidArgumentError(
        "temperature must be finite and >= 0, got {}", params.temperature);
  }
  if (params.top_k < 0) {
    return core::InvalidArgumentError("top_k must be >= 0, got {}",
                                      params.top_k);
  }
  if (!std::isfinite(params.top_p) || params.top_p <= 0.0F ||
      params.top_p > 1.0F) {
    return core::InvalidArgumentError(
        "top_p must be finite and in (0, 1], got {}", params.top_p);
  }
  if (!std::isfinite(params.repetition_penalty) ||
      params.repetition_penalty <= 0.0F) {
    return core::InvalidArgumentError(
        "repetition_penalty must be finite and > 0, got {}",
        params.repetition_penalty);
  }
  if (!std::isfinite(params.presence_penalty) ||
      params.presence_penalty < -2.0F || params.presence_penalty > 2.0F) {
    return core::InvalidArgumentError(
        "presence_penalty must be finite and in [-2, 2], got {}",
        params.presence_penalty);
  }
  if (!std::isfinite(params.frequency_penalty) ||
      params.frequency_penalty < -2.0F || params.frequency_penalty > 2.0F) {
    return core::InvalidArgumentError(
        "frequency_penalty must be finite and in [-2, 2], got {}",
        params.frequency_penalty);
  }
  if (params.max_tokens <= 0) {
    return core::InvalidArgumentError("max_tokens must be > 0, got {}",
                                      params.max_tokens);
  }
  for (std::size_t i = 0; i < params.stop_token_ids.size(); ++i) {
    if (params.stop_token_ids[i] < 0) {
      return core::InvalidArgumentError(
          "stop_token_ids[{}] must be >= 0, got {}", i,
          params.stop_token_ids[i]);
    }
  }
  for (std::size_t i = 0; i < params.stop_strings.size(); ++i) {
    if (params.stop_strings[i].empty()) {
      return core::InvalidArgumentError("stop_strings[{}] must not be empty",
                                        i);
    }
  }
  if (params.logprobs < 0 || params.logprobs > kMaxLogprobs) {
    return core::InvalidArgumentError("logprobs must be in [0, {}], got {}",
                                      kMaxLogprobs, params.logprobs);
  }
  return core::OkStatus();
}

}  // namespace engine::sampling
