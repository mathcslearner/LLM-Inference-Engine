#include "engine/generator.h"

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"
#include "sampling/params.h"
#include "sampling/sampler.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Generation loop (M5-T09; M7-T01 routes it through the sampler; design:
// docs/design/model-execution.md §10, §15). Deliberately allocation-light and
// backend-agnostic: prefill the prompt once (kLast), then decode one token per
// forward, sampling the next id from the final-position logits and feeding each
// produced token back as the next step's single input.

namespace engine::engine {
namespace {

// The `[1, V]` (or `[·, V]`) logits tensor from `forward` as a flat `[V]` span
// over its last row, for the sampler. `forward` returns a contiguous
// caller-owned buffer (kLast is a single row), so the last `V` elements are the
// final position's logits.
[[nodiscard]] std::span<const float> LastRow(const tensor::Tensor& logits) {
  const int rank = logits.shape().rank();
  const std::int64_t vocab = logits.shape().dim(rank - 1);
  const std::int64_t rows = logits.numel() / vocab;
  const float* base = logits.data_ptr<float>();
  const auto offset = static_cast<std::size_t>((rows - 1) * vocab);
  return std::span<const float>{base + offset, static_cast<std::size_t>(vocab)};
}

}  // namespace

core::StatusOr<std::vector<std::int32_t>> Generate(
    model::Model& model, kvcache::KvCache& cache,
    std::span<const std::int32_t> prompt_ids, const GenerateOptions& options,
    const TokenCallback& on_token) {
  if (prompt_ids.empty()) {
    return core::InvalidArgumentError("prompt_ids must be non-empty");
  }
  // Build the sampler up front: `Create` validates the params (an invalid or
  // not-yet-implemented `sampling` fails here, before any forward) and gives
  // the authoritative `max_tokens` the loop caps on. This is where the old
  // `max_new_tokens <= 0` check now lives (ValidateSamplingParams).
  auto sampler = sampling::Sampler::Create(options.sampling);
  if (!sampler.ok()) {
    return sampler.status();
  }
  const std::int64_t max_tokens = sampler->params().max_tokens;

  const auto prompt_len = static_cast<std::int64_t>(prompt_ids.size());
  const std::int64_t start = cache.length();
  // Worst-case cache occupancy (no early EOS): the prompt (prefill) plus one
  // appended token per decode step. The final produced token is never appended,
  // so the peak length is start + prompt_len + (max_tokens - 1). Checked up
  // front — a StatusOr<vector> cannot carry a partial result beside a Status.
  const std::int64_t peak = start + prompt_len + (max_tokens - 1);
  if (peak > cache.capacity()) {
    return core::ResourceExhaustedError(
        "cache capacity {} too small for prompt {} + {} new tokens from length "
        "{} (needs {})",
        cache.capacity(), prompt_len, max_tokens, start, peak);
  }

  const auto is_eos = [&](std::int32_t id) {
    return std::ranges::find(options.eos_ids, id) != options.eos_ids.end();
  };

  const auto cap = static_cast<std::size_t>(max_tokens);
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
    // Step 0: no tokens generated yet, so the sampler's context history is the
    // prompt alone.
    auto next = sampler->Sample(
        LastRow(*logits), sampling::SampleContext{.prompt_ids = prompt_ids,
                                                  .generated_ids = generated});
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
    // `generated` holds every token produced so far, so its size is this step's
    // index — exactly the context the penalties (T03) and seeded RNG (T02) key
    // on.
    auto next = sampler->Sample(
        LastRow(*logits), sampling::SampleContext{.prompt_ids = prompt_ids,
                                                  .generated_ids = generated});
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
