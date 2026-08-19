#include "engine/generator.h"

#include "core/status.h"
#include "engine/stop.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"
#include "sampling/params.h"
#include "sampling/sampler.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Generation loop (M5-T09; M7-T01 routes it through the sampler; M7-T04 adds
// the stop-condition machinery; design: docs/design/model-execution.md §10,
// §15). Deliberately allocation-light and backend-agnostic: prefill the prompt
// once (kLast), then decode one token per forward, sampling the next id from
// the final-position logits and feeding each produced token back as the next
// step's single input. A `StopChecker` (not the sampler) owns EOS / stop-token
// / stop-string / max_tokens stopping and the incremental detokenization.

namespace engine::engine {
namespace {

// The `[·, V]` logits tensor from `forward` as a flat `[V]` span over its last
// row, for the sampler. `forward` returns a contiguous caller-owned buffer
// (kLast is a single row), so the last `V` elements are the final position's
// logits.
[[nodiscard]] std::span<const float> LastRow(const tensor::Tensor& logits) {
  const int rank = logits.shape().rank();
  const std::int64_t vocab = logits.shape().dim(rank - 1);
  const std::int64_t rows = logits.numel() / vocab;
  const float* base = logits.data_ptr<float>();
  const auto offset = static_cast<std::size_t>((rows - 1) * vocab);
  return std::span<const float>{base + offset, static_cast<std::size_t>(vocab)};
}

}  // namespace

core::StatusOr<GenerateResult> Generate(
    model::Model& model, kvcache::KvCache& cache,
    std::span<const std::int32_t> prompt_ids, const GenerateOptions& options,
    const TokenCallback& on_token) {
  if (prompt_ids.empty()) {
    return core::InvalidArgumentError("prompt_ids must be non-empty");
  }
  // Build the sampler and stop checker up front: `Create` validates the params
  // (an invalid or not-yet-implemented `sampling`, or stop_strings without a
  // tokenizer, fails here before any forward) and gives the authoritative
  // `max_tokens` the loop caps on.
  auto sampler = sampling::Sampler::Create(options.sampling);
  if (!sampler.ok()) {
    return sampler.status();
  }
  auto checker =
      StopChecker::Create(options.sampling, options.eos_ids, options.tokenizer,
                          options.skip_special_tokens);
  if (!checker.ok()) {
    return checker.status();
  }
  const std::int64_t max_tokens = sampler->params().max_tokens;

  const auto prompt_len = static_cast<std::int64_t>(prompt_ids.size());
  const std::int64_t start = cache.length();
  // Worst-case cache occupancy (no early stop): the prompt (prefill) plus one
  // appended token per decode step. The final produced token is never appended,
  // so the peak length is start + prompt_len + (max_tokens - 1). Checked up
  // front — a StatusOr cannot carry a partial result beside a Status.
  const std::int64_t peak = start + prompt_len + (max_tokens - 1);
  if (peak > cache.capacity()) {
    return core::ResourceExhaustedError(
        "cache capacity {} too small for prompt {} + {} new tokens from length "
        "{} (needs {})",
        cache.capacity(), prompt_len, max_tokens, start, peak);
  }

  GenerateResult result;
  result.tokens.reserve(static_cast<std::size_t>(max_tokens));

  // Sample the next token from `logits`, append it to the result, run the stop
  // checker, fold in any end-of-stream residue, fire the callback, and report
  // whether generation should stop. Shared by prefill and each decode step.
  bool finished = false;
  const auto consume = [&](std::span<const float> logits) -> core::Status {
    auto next = sampler->Sample(
        logits, sampling::SampleContext{.prompt_ids = prompt_ids,
                                        .generated_ids = result.tokens});
    if (!next.ok()) {
      return next.status();
    }
    result.tokens.push_back(*next);

    auto step = checker->Observe(
        *next, static_cast<std::int64_t>(result.tokens.size()));
    if (!step.ok()) {
      return step.status();
    }
    std::string delta = std::move(step->text_delta);
    if (step->finished) {
      finished = true;
      result.finish_reason = step->finish_reason;
      result.stop_trigger = step->trigger;
      result.matched_stop = std::move(step->matched_stop);
      delta += checker->Finish();  // release held/residual text with this token
    }
    result.text += delta;
    if (on_token) {
      on_token(TokenEvent{.id = *next, .text = delta});
    }
    return core::OkStatus();
  };

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
    if (core::Status status = consume(LastRow(*logits)); !status.ok()) {
      return status;
    }
  }

  // --- Decode: feed the last produced token back as a single-token forward at
  //     the running cache position, until a stop condition. -------------------
  while (!finished) {
    std::int32_t token = result.tokens.back();
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
    if (core::Status status = consume(LastRow(*logits)); !status.ok()) {
      return status;
    }
  }

  return result;
}

}  // namespace engine::engine
