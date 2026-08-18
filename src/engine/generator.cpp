#include "engine/generator.h"

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Greedy generation loop (M5-T09; design: docs/design/model-execution.md §10).
// Deliberately allocation-light and backend-agnostic: prefill the prompt once
// (kLast), then decode one token per forward, argmax over the final-position
// logits, feeding each produced token back as the next step's single input.

namespace engine::engine {
namespace {

// Argmax over a `[1, V]` (or `[·, V]`) fp32 logits row: scan the last-dimension
// vector with a strict `>` so the lowest vocab index wins ties (deterministic,
// the two-runs-identical criterion). A non-finite maximum (NaN — an upstream
// numerics bug) is surfaced as Internal rather than returned as a bogus token.
[[nodiscard]] core::StatusOr<std::int32_t> ArgmaxLastRow(
    const tensor::Tensor& logits) {
  const int rank = logits.shape().rank();
  const std::int64_t vocab = logits.shape().dim(rank - 1);
  const float* row = logits.data_ptr<float>();
  std::int64_t best = 0;
  float best_val = row[0];
  for (std::int64_t v = 1; v < vocab; ++v) {
    if (row[v] > best_val) {
      best_val = row[v];
      best = v;
    }
  }
  if (std::isnan(best_val)) {
    return core::InternalError("argmax over non-finite logits (max is NaN)");
  }
  return static_cast<std::int32_t>(best);
}

}  // namespace

core::StatusOr<std::vector<std::int32_t>> Generate(
    model::Model& model, kvcache::KvCache& cache,
    std::span<const std::int32_t> prompt_ids, const GenerateOptions& options,
    const TokenCallback& on_token) {
  if (prompt_ids.empty()) {
    return core::InvalidArgumentError("prompt_ids must be non-empty");
  }
  if (options.max_new_tokens <= 0) {
    return core::InvalidArgumentError("max_new_tokens must be > 0, got {}",
                                      options.max_new_tokens);
  }

  const auto prompt_len = static_cast<std::int64_t>(prompt_ids.size());
  const std::int64_t start = cache.length();
  // Worst-case cache occupancy (no early EOS): the prompt (prefill) plus one
  // appended token per decode step. The final produced token is never appended,
  // so the peak length is start + prompt_len + (max_new_tokens - 1). Checked up
  // front — a StatusOr<vector> cannot carry a partial result beside a Status.
  const std::int64_t peak = start + prompt_len + (options.max_new_tokens - 1);
  if (peak > cache.capacity()) {
    return core::ResourceExhaustedError(
        "cache capacity {} too small for prompt {} + {} new tokens from length "
        "{} (needs {})",
        cache.capacity(), prompt_len, options.max_new_tokens, start, peak);
  }

  const auto is_eos = [&](std::int32_t id) {
    return std::ranges::find(options.eos_ids, id) != options.eos_ids.end();
  };

  const auto cap = static_cast<std::size_t>(options.max_new_tokens);
  std::vector<std::int32_t> generated;
  generated.reserve(cap);

  // --- Prefill: whole prompt at absolute positions start..start+T-1, logits
  //     for the last position only. ------------------------------------------
  std::vector<std::int32_t> positions(static_cast<std::size_t>(prompt_len));
  for (std::int64_t t = 0; t < prompt_len; ++t) {
    positions[static_cast<std::size_t>(t)] =
        static_cast<std::int32_t>(start + t);
  }
  {
    const model::ForwardRequest req{
        .token_ids = prompt_ids,
        .positions = positions,
        .cache = &cache,
        .logits_mode = model::LogitsMode::kLast,
    };
    auto logits = model.forward(req);
    if (!logits.ok()) {
      return logits.status();
    }
    auto next = ArgmaxLastRow(*logits);
    if (!next.ok()) {
      return next.status();
    }
    generated.push_back(*next);
    if (on_token) {
      on_token(*next);
    }
  }

  // --- Decode: feed the last produced token back as a single-token forward at
  //     the running cache position, until EOS or the cap. ---------------------
  std::int32_t token = generated.back();
  while (!is_eos(token) && generated.size() < cap) {
    const auto position = static_cast<std::int32_t>(cache.length());
    positions.assign(1, position);  // == start + prompt_len + (step index)
    const std::span<const std::int32_t> step_ids{&token, 1};
    const model::ForwardRequest req{
        .token_ids = step_ids,
        .positions = positions,
        .cache = &cache,
        .logits_mode = model::LogitsMode::kLast,
    };
    auto logits = model.forward(req);
    if (!logits.ok()) {
      return logits.status();
    }
    auto next = ArgmaxLastRow(*logits);
    if (!next.ok()) {
      return next.status();
    }
    token = *next;
    generated.push_back(token);
    if (on_token) {
      on_token(token);
    }
  }

  return generated;
}

}  // namespace engine::engine
