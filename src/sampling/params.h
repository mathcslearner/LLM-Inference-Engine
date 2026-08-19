#pragma once

#include "core/status.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Sampling & generation controls (M7; design: docs/design/model-execution.md
// §15). `SamplingParams` is the per-request knob bundle the API layer (M10)
// maps OpenAI request fields onto and the engine loop threads through the
// sampler. M7-T01 defines the struct, its defaults, and validation; the
// individual stages (temperature/top-k/top-p in T02, penalties in T03, stop
// conditions in T04, logprobs in T05) fill in the behaviour behind these
// fields. Defaults are the identity configuration: a default-constructed
// `SamplingParams` (temperature 1.0, all filters/penalties off) is the plain
// "sample from the raw softmax" request; `SamplingParams::Greedy(n)` is the
// argmax request the M5/M6 generation path used before this milestone.

namespace engine::sampling {

// OpenAI caps `logprobs` (well, `top_logprobs`) at 20; we mirror that so the
// per-step logprob buffers have a documented bound (used from T05).
inline constexpr std::int32_t kMaxLogprobs = 20;

// Per-request sampling configuration. Field semantics follow the OpenAI / vLLM
// conventions (documented per field); `ValidateSamplingParams` rejects any
// out-of-range combination before generation starts. Every field has a
// meaningful default so a default-constructed value is a valid, if maximally
// permissive, request.
struct SamplingParams {
  // Softmax temperature. `> 0` scales logits by `1/temperature` before the
  // softmax; **exactly 0 means greedy** (argmax), the OpenAI/vLLM convention —
  // no division is performed in that case. Must be finite and `>= 0`.
  float temperature = 1.0F;

  // Top-k filter: keep only the `top_k` highest-logit tokens before sampling.
  // `0` disables the filter (HF convention — keep the full vocabulary). Must be
  // `>= 0`.
  std::int32_t top_k = 0;

  // Top-p (nucleus) filter: keep the smallest set of highest-probability tokens
  // whose cumulative probability is `>= top_p`. `1.0` keeps everything. Must be
  // finite with `0 < top_p <= 1`.
  float top_p = 1.0F;

  // Repetition penalty (HF semantics): logits of previously seen tokens are
  // divided by this factor when positive, multiplied when negative — `1.0` is a
  // no-op. Must be finite and `> 0`.
  float repetition_penalty = 1.0F;

  // Presence penalty (OpenAI semantics): a flat subtraction from the logit of
  // any token that has appeared at least once. `0.0` is a no-op. Range
  // `[-2, 2]` (finite), matching the OpenAI API.
  float presence_penalty = 0.0F;

  // Frequency penalty (OpenAI semantics): subtraction scaled by how many times
  // a token has appeared. `0.0` is a no-op. Range `[-2, 2]` (finite), matching
  // the OpenAI API.
  float frequency_penalty = 0.0F;

  // Per-request RNG seed for reproducible sampling (T02 wires a counter-based
  // Philox stream keyed on this seed and the step index). `nullopt` = an
  // engine-chosen nondeterministic seed. No validation.
  std::optional<std::uint64_t> seed = std::nullopt;

  // Hard cap on generated tokens (prompt excluded); generation also stops early
  // on a stop condition (T04). Must be `> 0`. This is the field the M5/M6
  // `GenerateOptions::max_new_tokens` folded into.
  std::int64_t max_tokens = 0;

  // Stop as soon as any of these token ids is produced (T04). Each id must be
  // `>= 0`; the vocabulary-range check belongs to the forward pass, not here.
  std::vector<std::int32_t> stop_token_ids;

  // Stop as soon as any of these strings appears in the incrementally
  // detokenized output (T04, matched across token boundaries). No element may
  // be empty.
  std::vector<std::string> stop_strings;

  // Number of top logprobs to return per step (T05); `0` disables logprobs.
  // Must be in `[0, kMaxLogprobs]`.
  std::int32_t logprobs = 0;

  // The pre-M7 configuration: argmax with lowest-index tie-break, no filters,
  // no penalties. `temperature == 0` selects the greedy branch of the sampler.
  [[nodiscard]] static SamplingParams Greedy(std::int64_t max_tokens) {
    SamplingParams params;
    params.temperature = 0.0F;
    params.max_tokens = max_tokens;
    return params;
  }
};

// Validate a `SamplingParams`. Returns `InvalidArgument` (naming the offending
// field and value) on the first violated constraint, else `OkStatus()`. This is
// the single validation entry point the sampler (`Sampler::Create`) and the API
// layer share, so the rules live in exactly one place.
[[nodiscard]] core::Status ValidateSamplingParams(const SamplingParams& params);

}  // namespace engine::sampling
