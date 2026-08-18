#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"

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
// M5 hard-codes greedy — the smallest thing that makes end-to-end generation
// testable. The sampling pipeline (temperature, top-k/p, penalties) arrives in
// M7 behind the same loop skeleton (§15).

namespace engine::engine {

// Knobs for `Generate` (§10).
struct GenerateOptions {
  // Hard cap on generated tokens; must be > 0. Generation also stops early on
  // an EOS match. The prompt is excluded from this count.
  std::int64_t max_new_tokens = 0;
  // Stop as soon as any of these ids is produced (the matched id is included in
  // the result). Empty = run to `max_new_tokens`. The caller composes this from
  // the tokenizer's `eos_id()` and/or `ModelConfig::eos_token_ids`.
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

// Greedy (argmax) continuation of `prompt_ids`, appending this run's K/V into
// `cache` (which may already hold a prefix — decoding continues from
// `cache.length()`). Returns the generated ids (prompt excluded).
//
// Greedy tie-break is the lowest vocab index, so two runs are bit-identical
// (the determinism criterion). Stopping: the first `options.eos_ids` match (the
// EOS id is the last element of the result), or `options.max_new_tokens` ids.
//
// Errors leave nothing generated and, being front-loaded, `cache` unmodified:
//   * empty `prompt_ids`, or `max_new_tokens <= 0` → InvalidArgument.
//   * a prompt+continuation that cannot fit `cache.capacity()` →
//     ResourceExhausted. The check is up front (worst case, ignoring early EOS)
//     because a `StatusOr<vector>` cannot return a partial result beside a
//     Status — a caller who wants "generate up to the cache limit" sizes
//     `max_new_tokens` to fit.
//   * any `model.forward` error (out-of-range id, geometry mismatch, position
//     overflow) propagates unchanged.
[[nodiscard]] core::StatusOr<std::vector<std::int32_t>> Generate(
    model::Model& model, kvcache::KvCache& cache,
    std::span<const std::int32_t> prompt_ids, const GenerateOptions& options,
    const TokenCallback& on_token = {});

}  // namespace engine::engine
