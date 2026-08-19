#pragma once

#include "core/status.h"
#include "sampling/params.h"
#include "tokenizer/detokenize.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Stop conditions & finish reasons (M7-T04; design:
// docs/design/model-execution.md §15.2 stop paragraph). The generation loop
// (§10) turns each sampled token into a stop/continue decision plus the
// safe-to-emit text delta. Two pieces live here, both engine-side (the sampler
// stays `core`-only, so nothing tokenizer-flavoured belongs in `sampling`):
//
//   * `StopStringMatcher` — a pure byte-stream matcher that trims the output at
//     the first occurrence of any stop string, buffering just enough of the
//     tail to catch a stop string split across token boundaries. No tokenizer
//     dependency, so it is unit-testable in isolation and reusable by M9's
//     per-request loop.
//   * `StopChecker` — the per-request composite: it detokenizes each id (a
//     `tokenizer::DetokenizerStream`), feeds the text through the matcher, and
//     applies the id-based conditions (the model EOS set, the request's
//     `stop_token_ids`) and `max_tokens`, producing a `FinishReason`.
//
// `engine → tokenizer` is an already-sanctioned edge (ADR-002 layer table /
// module diagram); M7-T04 is the first ticket to use it.

namespace engine::engine {

// The OpenAI-style completion finish reason. `kStop` = a stop condition fired
// (EOS id, a `stop_token_ids` id, or a `stop_strings` match); `kLength` =
// `max_tokens` reached without one. These are the wire strings the M10 API
// layer emits.
enum class FinishReason : std::uint8_t { kStop, kLength };

// The wire string for a finish reason ("stop" / "length").
[[nodiscard]] std::string_view FinishReasonName(FinishReason reason);

// Which specific condition ended generation — finer than `FinishReason` (the
// M10 layer maps this to vLLM's `stop_reason`; tests assert exactly which
// condition fired). `kNone` means generation has not stopped.
enum class StopTrigger : std::uint8_t {
  kNone,         // still running
  kEosId,        // an id in the model EOS set (GenerateOptions::eos_ids)
  kStopTokenId,  // an id in the request's SamplingParams::stop_token_ids
  kStopString,   // a SamplingParams::stop_strings match on the detok stream
  kMaxTokens,    // SamplingParams::max_tokens reached
};

// Streaming stop-string matcher over a byte stream. `Feed` appends a text chunk
// and returns the prefix that is now safe to emit; it holds back the longest
// suffix that could still become (the start of) a stop string, so a stop string
// spanning multiple `Feed` calls is still caught and the text after the match
// is trimmed. Matching is bytewise (the detokenized stream is UTF-8, but a stop
// string is compared as raw bytes — an ill-formed stop is the M10 mapper's
// concern). With no stop strings it is a pure pass-through.
//
// The scan is naive (O(held · Σ|stop|) per feed); stop strings are short and
// few, so an Aho–Corasick automaton is not warranted.
class StopStringMatcher {
 public:
  explicit StopStringMatcher(std::vector<std::string> stops);

  struct FeedResult {
    // Text safe to emit now (everything before a match, minus any held tail).
    std::string emit;
    // Set once a stop string matched; then `emit` is the text strictly before
    // the match and the matcher is spent (later `Feed`s are no-ops).
    std::optional<std::string> matched;
  };

  // Append `chunk` and return the newly emittable prefix (see `FeedResult`).
  [[nodiscard]] FeedResult Feed(std::string_view chunk);

  // Release the held tail at end of stream (no match): it is real output once
  // no further tokens can complete a stop string. Clears the buffer, so a
  // second call returns "".
  [[nodiscard]] std::string Flush();

  [[nodiscard]] bool has_stops() const { return !stops_.empty(); }

 private:
  std::vector<std::string> stops_;
  std::string
      held_;  // buffered tail: a strict prefix of some stop, < max|stop|
  bool matched_ = false;
};

// Per-request stop bookkeeping over a produced-token stream. Construct once per
// request (`Create`), call `Observe` for each sampled token in order, and
// `Finish` once after the loop ends. `Observe` returns the safe-to-emit text
// delta and whether (and why) generation should stop.
class StopChecker {
 public:
  // Build from the request's sampling params, the model EOS set, and an
  // optional tokenizer. A tokenizer is required iff `params.stop_strings` is
  // non-empty (stop strings match on the detokenized stream); passing none with
  // stop strings set is `InvalidArgument`. When `tokenizer` is null no text is
  // produced (`text_delta` is always empty) — the token-only path the tests and
  // the benchmark use. `params` must already have passed
  // `ValidateSamplingParams` (the caller's `Sampler::Create` does this).
  [[nodiscard]] static core::StatusOr<StopChecker> Create(
      const sampling::SamplingParams& params,
      std::span<const std::int32_t> eos_ids,
      const tokenizer::Tokenizer* tokenizer, bool skip_special_tokens);

  struct StepResult {
    std::string text_delta;  // safe-to-emit text from this token ("" if no
                             // tokenizer, or held back for a pending stop)
    bool finished = false;   // generation should stop after this token
    FinishReason finish_reason = FinishReason::kLength;
    StopTrigger trigger = StopTrigger::kNone;
    std::string matched_stop;  // the matched stop string (kStopString only)
  };

  // Observe the token just appended to the output. `produced_count` is the
  // number of tokens produced so far including this one (i.e. the 1-based step)
  // — `max_tokens` fires when it reaches the cap. An out-of-range id surfaces
  // as `InvalidArgument` (from the detokenizer) and leaves the checker usable.
  // Must not be called again once a call has reported `finished`.
  [[nodiscard]] core::StatusOr<StepResult> Observe(std::int32_t id,
                                                   std::int64_t produced_count);

  // Release any text the detokenizer or matcher held back — real output once
  // the stream ends. Returns "" when generation stopped on a stop-string match
  // (the held/residual text is past the match and trimmed). Call once, after
  // the last `Observe`.
  [[nodiscard]] std::string Finish();

 private:
  StopChecker(std::vector<std::int32_t> eos_ids,
              std::vector<std::int32_t> stop_token_ids, std::int64_t max_tokens,
              std::optional<tokenizer::DetokenizerStream> detok,
              StopStringMatcher matcher)
      : eos_ids_(std::move(eos_ids)),
        stop_token_ids_(std::move(stop_token_ids)),
        max_tokens_(max_tokens),
        detok_(std::move(detok)),
        matcher_(std::move(matcher)) {}

  [[nodiscard]] bool IsEos(std::int32_t id) const;
  [[nodiscard]] bool IsStopTokenId(std::int32_t id) const;

  std::vector<std::int32_t> eos_ids_;
  std::vector<std::int32_t> stop_token_ids_;
  std::int64_t max_tokens_;
  std::optional<tokenizer::DetokenizerStream> detok_;
  StopStringMatcher matcher_;
  StopTrigger trigger_ = StopTrigger::kNone;  // set once finished
};

}  // namespace engine::engine
