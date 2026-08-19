#pragma once

#include "core/status.h"
#include "engine/stop.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"
#include "sampling/logprobs.h"
#include "sampling/params.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The generation loop (M5-T09; design: docs/design/model-execution.md §10).
// Prefill the prompt in one forward, then autoregressively decode one token per
// step by sampling the final-position logits, appending K/V into the cache.
// Backend-agnostic: it touches only the abstract `Model` and `KvCache`
// interfaces, so M6-T07 reuses it verbatim against the optimized backend, and
// the M5 golden is agreement with HF `generate(do_sample=False)`.
//
// M5 hard-coded greedy — the smallest thing that makes end-to-end generation
// testable. M7-T01 routes the loop through the `sampling::Sampler` pipeline
// (§15). M7-T04 adds stop conditions and finish reasons: the model EOS set,
// the request's `stop_token_ids`/`stop_strings` (matched on the incrementally
// detokenized stream, across token boundaries), and `max_tokens` produce a
// `FinishReason`. This machinery is the loop's (a `StopChecker`, §15.2), not
// the sampler's.

namespace engine::engine {

// Knobs for `Generate` (§10, §15).
struct GenerateOptions {
  // The per-request sampling configuration (temperature, filters, penalties,
  // max_tokens, stop conditions, …). `max_tokens` caps the generated-token
  // count (prompt excluded) and must be > 0. `SamplingParams::Greedy(n)`
  // reproduces the pre-M7 argmax behaviour.
  sampling::SamplingParams sampling;
  // Model-derived end-of-sequence ids: stop as soon as any is produced (the
  // matched id is included in the result). Empty = run to a stop condition or
  // `sampling.max_tokens`. The caller composes this from the tokenizer's
  // `eos_id()` and/or `ModelConfig::eos_token_ids`. Distinct from a request's
  // `SamplingParams::stop_token_ids`.
  std::vector<std::int32_t> eos_ids;
  // Tokenizer used to detokenize the produced ids into `GenerateResult::text`
  // and to match `sampling.stop_strings`. Optional: null runs the token-only
  // path (no text, no stop strings) that the fixture tests and the throughput
  // benchmark use. **Required** when `sampling.stop_strings` is non-empty —
  // `Generate` then returns `InvalidArgument`. Must outlive the call.
  const tokenizer::Tokenizer* tokenizer = nullptr;
  // Whether special tokens are dropped from the detokenized text/stop stream
  // (HF `skip_special_tokens`). Ignored when `tokenizer` is null.
  bool skip_special_tokens = true;
};

// One produced token and the text that became safe to emit with it — the
// streaming seam (M10's SSE and cancel-within-one-step build on it). `text` is
// empty when no tokenizer is supplied, or while bytes are held back for an
// as-yet-incomplete UTF-8 sequence or a pending stop-string prefix; the
// concatenation of every event's `text` equals `GenerateResult::text`.
struct TokenEvent {
  std::int32_t id;
  std::string_view text;
  // The per-step logprobs for this token when `sampling.logprobs > 0`, else
  // null (the M10 SSE per-chunk `logprobs` seam). Points into the
  // `GenerateResult::logprobs` entry for this token; valid for the duration of
  // the callback.
  const sampling::StepLogprobs* logprobs = nullptr;
};

// Invoked with each produced token as it is produced, in order — exactly once
// per returned id (including a terminal EOS/stop id), after that id is appended
// to the result and before the next forward. `std::function` for now; a
// non-allocating callback is a measured-perf follow-up if a hot path needs one.
using TokenCallback = std::function<void(const TokenEvent&)>;

// The outcome of a `Generate` call.
struct GenerateResult {
  // The generated ids (prompt excluded). A terminating EOS / stop-token id is
  // the last element; a token that completed a stop string is included too.
  std::vector<std::int32_t> tokens;
  // Why generation stopped: `kStop` (a stop condition) or `kLength`
  // (`max_tokens` reached).
  FinishReason finish_reason = FinishReason::kLength;
  // The specific condition that fired (finer than `finish_reason`).
  StopTrigger stop_trigger = StopTrigger::kNone;
  // The matched stop string, when `stop_trigger == kStopString` (else empty).
  std::string matched_stop;
  // The detokenized generated text, trimmed before a stop-string match. Empty
  // when no tokenizer was supplied.
  std::string text;
  // Per-step logprobs, index-aligned with `tokens`, when `sampling.logprobs >
  // 0`; empty otherwise. `logprobs[i]` describes `tokens[i]` (§15.2 stage 6).
  std::vector<sampling::StepLogprobs> logprobs;
};

// Sampled continuation of `prompt_ids`, appending this run's K/V into `cache`
// (which may already hold a prefix — decoding continues from `cache.length()`).
// The token at each step comes from a `sampling::Sampler` built from
// `options.sampling`; with `SamplingParams::Greedy(n)` this is argmax with a
// lowest-index tie-break, so two runs are bit-identical.
//
// Stopping (checked per token, in this priority): an `options.eos_ids` match, a
// `sampling.stop_token_ids` match, a `sampling.stop_strings` match on the
// detokenized stream, or `sampling.max_tokens`. The first three yield
// `FinishReason::kStop`; the last, `kLength`.
//
// Two error classes with different cache/output guarantees:
//
// *Front-loaded* errors are raised before any forward runs, leave nothing
// generated, and leave `cache` unmodified:
//   * empty `prompt_ids` → InvalidArgument.
//   * an invalid or not-yet-implemented `options.sampling` → InvalidArgument /
//     Unimplemented (from `Sampler::Create`); `max_tokens <= 0` is one such
//     InvalidArgument.
//   * `sampling.stop_strings` set without `options.tokenizer` → InvalidArgument
//     (from `StopChecker::Create`).
//   * a prompt+continuation that cannot fit `cache.capacity()` →
//     ResourceExhausted. The check is up front (worst case, ignoring early
//     stops) because a `StatusOr` cannot return a partial result beside a
//     Status — a caller who wants "generate up to the cache limit" sizes
//     `max_tokens` to fit. For a private cache this bound is exact, so a run
//     that cannot fit is rejected here with the cache untouched.
//
// *Mid-generation* errors surface once decoding is under way (any
// `model.forward` failure — out-of-range id, geometry mismatch, position
// overflow, or a `ResourceExhausted` from a **shared** paged pool drained by
// another sequence after this run's up-front check passed, M8-T08). These
// propagate the `forward` status unchanged and, because a `StatusOr` cannot
// carry a partial result, the already-produced tokens are **not** in the
// returned value — but every one of them was already delivered through
// `on_token` (the callback is the partial-result channel), and the failing
// forward wrote nothing (front-loaded inside `forward`), so `cache` holds a
// consistent prefix of exactly `cache.length()` committed tokens. Generation is
// therefore **resumable**: once capacity frees up, a later
// `Generate(model, cache, {last delivered token}, …)` continues the identical
// greedy trajectory (the KV invariant as a resumability guarantee). The
// sequence's blocks are reclaimed when its cache drops regardless (RAII), so no
// blocks leak on the error path. This is the seam M9 preemption/resume and
// M10 streaming-with-cancel build on.
[[nodiscard]] core::StatusOr<GenerateResult> Generate(
    model::Model& model, kvcache::KvCache& cache,
    std::span<const std::int32_t> prompt_ids, const GenerateOptions& options,
    const TokenCallback& on_token = {});

}  // namespace engine::engine
