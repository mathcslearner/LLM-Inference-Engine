#include "engine/stop.h"

#include "core/status.h"
#include "sampling/params.h"
#include "tokenizer/detokenize.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// M7-T04: the stop machinery the generation loop drives. See stop.h for the
// contract; the matcher's streaming trim and the checker's condition priority
// (EOS → stop_token_ids → stop_strings → max_tokens, matching vLLM) are the
// load-bearing behaviours the tests pin.

namespace engine::engine {

std::string_view FinishReasonName(FinishReason reason) {
  switch (reason) {
    case FinishReason::kStop:
      return "stop";
    case FinishReason::kLength:
      return "length";
  }
  return "length";
}

// -------------------------------------------------------- StopStringMatcher --

StopStringMatcher::StopStringMatcher(std::vector<std::string> stops)
    : stops_(std::move(stops)) {}

namespace {

// The longest suffix of `text` that is a strict prefix (length < |stop|) of
// `stop`, i.e. the number of trailing bytes that could still grow into `stop`.
[[nodiscard]] std::size_t LongestPrefixOverlap(std::string_view text,
                                               std::string_view stop) {
  const std::size_t max_len =
      std::min(text.size(), stop.size() - 1);  // full-length would be a match
  for (std::size_t len = max_len; len >= 1; --len) {
    if (text.substr(text.size() - len) == stop.substr(0, len)) {
      return len;
    }
  }
  return 0;
}

}  // namespace

StopStringMatcher::FeedResult StopStringMatcher::Feed(std::string_view chunk) {
  FeedResult result;
  if (matched_ || stops_.empty()) {
    // Spent, or pure pass-through: emit the chunk verbatim (held_ stays empty).
    result.emit.assign(chunk);
    return result;
  }
  held_.append(chunk);

  // Earliest occurrence of any stop string; ties broken by the stop's order in
  // the request list (the first-listed wins), matching a "check each stop in
  // order" scan.
  std::size_t best_pos = std::string::npos;
  const std::string* best_stop = nullptr;
  for (const std::string& stop : stops_) {
    const std::size_t pos = held_.find(stop);
    if (pos != std::string::npos && pos < best_pos) {
      best_pos = pos;
      best_stop = &stop;
    }
  }
  if (best_stop != nullptr) {
    result.emit = held_.substr(0, best_pos);
    result.matched = *best_stop;
    held_.clear();
    matched_ = true;
    return result;
  }

  // No match: hold back the longest tail that could still start a stop string.
  std::size_t hold = 0;
  for (const std::string& stop : stops_) {
    hold = std::max(hold, LongestPrefixOverlap(held_, stop));
  }
  const std::size_t emit_len = held_.size() - hold;
  result.emit = held_.substr(0, emit_len);
  held_.erase(0, emit_len);
  return result;
}

std::string StopStringMatcher::Flush() {
  std::string out = std::move(held_);
  held_.clear();
  return out;
}

// -------------------------------------------------------------- StopChecker --

core::StatusOr<StopChecker> StopChecker::Create(
    const sampling::SamplingParams& params,
    std::span<const std::int32_t> eos_ids,
    const tokenizer::Tokenizer* tokenizer, bool skip_special_tokens) {
  if (!params.stop_strings.empty() && tokenizer == nullptr) {
    return core::InvalidArgumentError(
        "stop_strings require a tokenizer to match on the detokenized stream");
  }
  std::optional<tokenizer::DetokenizerStream> detok;
  if (tokenizer != nullptr) {
    detok.emplace(*tokenizer, skip_special_tokens);
  }
  return StopChecker(std::vector<std::int32_t>(eos_ids.begin(), eos_ids.end()),
                     params.stop_token_ids, params.max_tokens, std::move(detok),
                     StopStringMatcher(params.stop_strings));
}

bool StopChecker::IsEos(std::int32_t id) const {
  return std::ranges::find(eos_ids_, id) != eos_ids_.end();
}

bool StopChecker::IsStopTokenId(std::int32_t id) const {
  return std::ranges::find(stop_token_ids_, id) != stop_token_ids_.end();
}

core::StatusOr<StopChecker::StepResult> StopChecker::Observe(
    std::int32_t id, std::int64_t produced_count) {
  // Detokenize first so the text delta is available regardless of which
  // condition fires. Empty when there is no tokenizer, or for a special token
  // under skip_special_tokens.
  std::string delta;
  if (detok_.has_value()) {
    core::StatusOr<std::string> piece = detok_->push(id);
    if (!piece.ok()) {
      return piece.status();
    }
    delta = *std::move(piece);
  }

  StepResult result;

  // Priority 1 — id-based stops (EOS set, then request stop_token_ids). The
  // token's own text is real output and is NOT trimmed; the matcher's held tail
  // (a stop-string prefix that never completed) is released as real text too.
  StopTrigger id_trigger = StopTrigger::kNone;
  if (IsEos(id)) {
    id_trigger = StopTrigger::kEosId;
  } else if (IsStopTokenId(id)) {
    id_trigger = StopTrigger::kStopTokenId;
  }
  if (id_trigger != StopTrigger::kNone) {
    result.finished = true;
    result.finish_reason = FinishReason::kStop;
    result.trigger = id_trigger;
    result.text_delta = matcher_.Flush();
    result.text_delta += delta;
    trigger_ = id_trigger;
    return result;
  }

  // Priority 2 — stop strings on the detokenized stream (may span tokens).
  if (matcher_.has_stops()) {
    StopStringMatcher::FeedResult fed = matcher_.Feed(delta);
    result.text_delta = std::move(fed.emit);
    if (fed.matched.has_value()) {
      result.finished = true;
      result.finish_reason = FinishReason::kStop;
      result.trigger = StopTrigger::kStopString;
      result.matched_stop = *std::move(fed.matched);
      trigger_ = StopTrigger::kStopString;
      return result;
    }
  } else {
    result.text_delta = std::move(delta);
  }

  // Priority 3 — max_tokens. The matcher's held tail (if any) is released in
  // Finish(), which the loop calls immediately after this finishing step.
  if (produced_count >= max_tokens_) {
    result.finished = true;
    result.finish_reason = FinishReason::kLength;
    result.trigger = StopTrigger::kMaxTokens;
    trigger_ = StopTrigger::kMaxTokens;
  }
  return result;
}

std::string StopChecker::Finish() {
  // On a stop-string match everything after the match (the matcher's held tail
  // is already cleared, plus the detokenizer's trailing residue) is trimmed.
  if (trigger_ == StopTrigger::kStopString) {
    return {};
  }
  std::string out = matcher_.Flush();
  if (detok_.has_value()) {
    out += detok_->finish();
  }
  return out;
}

}  // namespace engine::engine
