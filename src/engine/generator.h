#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"
#include "sampling/params.h"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// The greedy generation loop (M5-T09; design: docs/design/model-execution.md
// §10). Prefill the prompt in one forward, then autoregressively decode one
// token per step by argmax over the final-position logits, appending K/V into
// the cache. Backend-agnostic: it touches only the abstract `Model` and
// `KvCache` interfaces, so M6-T07 reuses it verbatim against the optimized
// backend, and the M5 golden is agreement with HF `generate(do_sample=False)`.
//
// M5 hard-coded greedy — the smallest thing that makes end-to-end generation
// testable. M7-T01 routes the loop through the `sampling::Sampler` pipeline
// (§15) without changing behaviour: greedy is now the sampler's
// `temperature == 0` branch, and the full pipeline (temperature, top-k/p,
// penalties, logprobs) lands stage by stage across M7 behind the same skeleton.

namespace engine::engine {

// Knobs for `Generate` (§10, §15).
struct GenerateOptions {
  // The per-request sampling configuration (temperature, filters, penalties,
  // max_tokens, …). `max_tokens` caps the generated-token count (prompt
  // excluded) and must be > 0. `SamplingParams::Greedy(n)` reproduces the
  // pre-M7 argmax behaviour.
  sampling::SamplingParams sampling;
  // Stop as soon as any of these ids is produced (the matched id is included in
  // the result). Empty = run to `sampling.max_tokens`. The caller composes this
  // from the tokenizer's `eos_id()` and/or `ModelConfig::eos_token_ids`. This
  // is model-derived end-of-sequence handling, distinct from a request's
  // `SamplingParams::stop_token_ids` (T04).
  std::vector<std::int32_t> eos_ids;
};

// Invoked with each newly produced token id, in order, as it is produced — the
// streaming seam the roadmap names ("hooks for per-token callbacks, streaming
// later"; M10's SSE and cancel-within-one-step build on it). Fires exactly once
// per returned id, including a terminal EOS id, after that id is appended to
// the result and before the next forward. `std::function` for now; a
// non-allocating callback is a measured-perf follow-up if a hot path ever needs
// one.
using TokenCallback = std::function<void(std::int32_t)>;

// Sampled continuation of `prompt_ids`, appending this run's K/V into `cache`
// (which may already hold a prefix — decoding continues from `cache.length()`).
// Returns the generated ids (prompt excluded). The token at each step comes
// from a `sampling::Sampler` built from `options.sampling`; with
// `SamplingParams::Greedy(n)` this is argmax with a lowest-index tie-break, so
// two runs are bit-identical (the determinism criterion).
//
// Stopping: the first `options.eos_ids` match (the EOS id is the last element
// of the result), or `options.sampling.max_tokens` ids.
//
// Errors leave nothing generated and, being front-loaded, `cache` unmodified:
//   * empty `prompt_ids` → InvalidArgument.
//   * an invalid or not-yet-implemented `options.sampling` → InvalidArgument /
//     Unimplemented (from `Sampler::Create`); `max_tokens <= 0` is one such
//     InvalidArgument.
//   * a prompt+continuation that cannot fit `cache.capacity()` →
//     ResourceExhausted. The check is up front (worst case, ignoring early EOS)
//     because a `StatusOr<vector>` cannot return a partial result beside a
//     Status — a caller who wants "generate up to the cache limit" sizes
//     `max_tokens` to fit.
//   * any `model.forward` error (out-of-range id, geometry mismatch, position
//     overflow) propagates unchanged.
[[nodiscard]] core::StatusOr<std::vector<std::int32_t>> Generate(
    model::Model& model, kvcache::KvCache& cache,
    std::span<const std::int32_t> prompt_ids, const GenerateOptions& options,
    const TokenCallback& on_token = {});

}  // namespace engine::engine
