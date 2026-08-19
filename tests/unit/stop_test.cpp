#include "engine/stop.h"

#include "core/status.h"
#include "sampling/params.h"
#include "tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// M7-T04: the stop machinery (engine/stop.{h,cpp}). Two layers:
//   * StopStringMatcher — the pure byte-stream trim: a match in one chunk, a
//     match split across two/three chunks (the ticket's headline case), the
//     hold-back that keeps a pending prefix out of the output until it either
//     completes or is broken, overlap, earliest/longest tie-breaks, UTF-8, and
//     Flush.
//   * StopChecker — the composite over a real (synthetic) tokenizer: the
//     EOS/stop_token/stop_string/max_tokens priority, exact max_tokens,
//     finish_reason/trigger per case, trailing-text trimming, held-prefix
//     release on a non-stop-string finish, and deltas concatenating to the full
//     text. `Create`'s stop_strings-without-tokenizer rejection.

namespace {

using engine::core::IsInvalidArgument;
using engine::engine::FinishReason;
using engine::engine::FinishReasonName;
using engine::engine::StopChecker;
using engine::engine::StopStringMatcher;
using engine::engine::StopTrigger;
using engine::sampling::SamplingParams;
using engine::tokenizer::Tokenizer;

// ------------------------------------------------ StopStringMatcher ------

StopStringMatcher Matcher(std::vector<std::string> stops) {
  return StopStringMatcher(std::move(stops));
}

TEST(StopStringMatcherTest, NoStopsIsPassThrough) {
  StopStringMatcher m = Matcher({});
  EXPECT_FALSE(m.has_stops());
  const StopStringMatcher::FeedResult r = m.Feed("anything at all");
  EXPECT_EQ(r.emit, "anything at all");
  EXPECT_FALSE(r.matched.has_value());
  EXPECT_EQ(m.Flush(), "");
}

TEST(StopStringMatcherTest, MatchInSingleChunkTrimsTrailingText) {
  StopStringMatcher m = Matcher({"STOP"});
  const StopStringMatcher::FeedResult r = m.Feed("hiSTOPthere");
  EXPECT_EQ(r.emit, "hi");
  ASSERT_TRUE(r.matched.has_value());
  EXPECT_EQ(r.matched.value_or(""), "STOP");
}

TEST(StopStringMatcherTest, MatchAcrossTwoChunksIsCaughtAndTrailingTrimmed) {
  StopStringMatcher m = Matcher({"STOP"});
  // The stop string is split "ST" | "OPthere": the first chunk emits only the
  // text before the pending prefix; the match lands in the second, and the
  // trailing "there" is trimmed.
  const StopStringMatcher::FeedResult a = m.Feed("hiST");
  EXPECT_EQ(a.emit, "hi");
  EXPECT_FALSE(a.matched.has_value());
  const StopStringMatcher::FeedResult b = m.Feed("OPthere");
  EXPECT_EQ(b.emit, "");
  ASSERT_TRUE(b.matched.has_value());
  EXPECT_EQ(b.matched.value_or(""), "STOP");
}

TEST(StopStringMatcherTest, MatchAcrossThreeChunks) {
  StopStringMatcher m = Matcher({"STOP"});
  EXPECT_EQ(m.Feed("S").emit, "");
  EXPECT_EQ(m.Feed("T").emit, "");
  const StopStringMatcher::FeedResult r = m.Feed("OPx");
  EXPECT_EQ(r.emit, "");
  ASSERT_TRUE(r.matched.has_value());
  EXPECT_EQ(r.matched.value_or(""), "STOP");
}

TEST(StopStringMatcherTest, HeldPrefixReleasedWhenBroken) {
  StopStringMatcher m = Matcher({"STOP"});
  EXPECT_EQ(m.Feed("ST").emit, "");  // held: could still become STOP
  const StopStringMatcher::FeedResult r = m.Feed("X");  // ...but "STX" cannot
  EXPECT_EQ(r.emit, "STX");
  EXPECT_FALSE(r.matched.has_value());
}

TEST(StopStringMatcherTest, OverlappingRepeats) {
  StopStringMatcher m = Matcher({"aab"});
  const StopStringMatcher::FeedResult r = m.Feed("aaab");
  EXPECT_EQ(r.emit, "a");  // first occurrence of "aab" starts at index 1
  ASSERT_TRUE(r.matched.has_value());
  EXPECT_EQ(r.matched.value_or(""), "aab");
}

TEST(StopStringMatcherTest, EarliestOccurrenceWins) {
  StopStringMatcher m = Matcher({"world", "lo"});
  const StopStringMatcher::FeedResult r = m.Feed("hello world");
  EXPECT_EQ(r.emit, "hel");  // "lo" at index 3 beats "world" at index 6
  ASSERT_TRUE(r.matched.has_value());
  EXPECT_EQ(r.matched.value_or(""), "lo");
}

TEST(StopStringMatcherTest, TieAtSamePositionBrokenByListOrder) {
  StopStringMatcher m = Matcher({"ab", "abc"});
  const StopStringMatcher::FeedResult r = m.Feed("abc");
  EXPECT_EQ(r.emit, "");
  ASSERT_TRUE(r.matched.has_value());
  EXPECT_EQ(r.matched.value_or(""),
            "ab");  // first-listed wins the tie at index 0
}

TEST(StopStringMatcherTest, Utf8StopSplitAcrossChunks) {
  // "🛑" is F0 9F 9B 91; split it across two feeds.
  StopStringMatcher m = Matcher({"🛑"});
  const StopStringMatcher::FeedResult a = m.Feed(std::string("go\xF0\x9F"));
  EXPECT_EQ(a.emit, "go");
  EXPECT_FALSE(a.matched.has_value());
  const StopStringMatcher::FeedResult b =
      m.Feed(std::string("\x9B\x91"
                         "end"));
  EXPECT_EQ(b.emit, "");
  ASSERT_TRUE(b.matched.has_value());
  EXPECT_EQ(b.matched.value_or(""), "🛑");
}

TEST(StopStringMatcherTest, FlushReleasesHeldTail) {
  StopStringMatcher m = Matcher({"STOP"});
  EXPECT_EQ(m.Feed("aST").emit, "a");  // "ST" held
  EXPECT_EQ(m.Flush(), "ST");          // released as real output at end
  EXPECT_EQ(m.Flush(), "");            // idempotent
}

TEST(StopStringMatcherTest, NonPrefixStreamIsNotHeldBack) {
  // A stream that never forms a stop prefix is emitted in full — the held
  // buffer never grows unboundedly.
  StopStringMatcher m = Matcher({"STOP"});
  std::string total;
  std::string expected;
  for (int i = 0; i < 100; ++i) {
    total += m.Feed("xyz").emit;
    expected += "xyz";
  }
  total += m.Flush();
  EXPECT_EQ(total, expected);     // emitted in full
  EXPECT_EQ(total.size(), 300U);  // nothing held back
}

// ------------------------------------------------------- StopChecker -----

// A minimal byte-level BPE tokenizer (the Qwen-2 Split pattern), vocab
// {a:0, b:1, ab:2} plus a special added token <s>=3 and a non-special <tool>=4.
// token_bytes: 0->"a", 1->"b", 2->"ab", 3->"<s>", 4->"<tool>".
constexpr std::string_view kMiniJson = R"({
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [
    {"id": 3, "content": "<s>", "special": true},
    {"id": 4, "content": "<tool>", "special": false}],
  "normalizer": null,
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex":
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
     "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false,
     "trim_offsets": true, "use_regex": false}]},
  "post_processor": null, "decoder": null,
  "model": {"type": "BPE", "ignore_merges": false,
    "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": []}})";

Tokenizer MiniTokenizer() {
  auto tok = Tokenizer::from_json(kMiniJson);
  EXPECT_TRUE(tok.ok()) << tok.status().ToString();
  return *std::move(tok);
}

SamplingParams Params(std::int64_t max_tokens) {
  SamplingParams p;
  p.max_tokens = max_tokens;
  return p;
}

// Drive a checker over a token stream, collecting the concatenated text and the
// finishing step. Stops feeding once a step reports finished.
struct DriveResult {
  std::string text;
  bool finished = false;
  FinishReason finish_reason = FinishReason::kLength;
  StopTrigger trigger = StopTrigger::kNone;
  std::string matched_stop;
  int consumed = 0;  // tokens actually observed
};

DriveResult Drive(StopChecker& checker, const std::vector<std::int32_t>& ids) {
  DriveResult out;
  std::int64_t count = 0;
  for (const std::int32_t id : ids) {
    ++count;
    auto step = checker.Observe(id, count);
    EXPECT_TRUE(step.ok()) << step.status().ToString();
    out.text += step->text_delta;
    ++out.consumed;
    if (step->finished) {
      out.finished = true;
      out.finish_reason = step->finish_reason;
      out.trigger = step->trigger;
      out.matched_stop = step->matched_stop;
      break;
    }
  }
  out.text += checker.Finish();
  return out;
}

TEST(StopCheckerTest, FinishReasonNames) {
  EXPECT_EQ(FinishReasonName(FinishReason::kStop), "stop");
  EXPECT_EQ(FinishReasonName(FinishReason::kLength), "length");
}

TEST(StopCheckerTest, StopStringsRequireTokenizer) {
  SamplingParams p = Params(8);
  p.stop_strings = {"ab"};
  const auto checker = StopChecker::Create(p, {}, /*tokenizer=*/nullptr, true);
  ASSERT_FALSE(checker.ok());
  EXPECT_TRUE(IsInvalidArgument(checker.status()));
}

TEST(StopCheckerTest, MaxTokensExactAndLength) {
  const Tokenizer tok = MiniTokenizer();
  auto checker = StopChecker::Create(Params(3), {}, &tok, true);
  ASSERT_TRUE(checker.ok());
  // Four tokens available but max_tokens is 3: stops exactly at the third.
  const DriveResult r = Drive(*checker, {0, 1, 0, 1});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.consumed, 3);
  EXPECT_EQ(r.trigger, StopTrigger::kMaxTokens);
  EXPECT_EQ(r.finish_reason, FinishReason::kLength);
  EXPECT_EQ(r.text, "aba");  // a b a
}

TEST(StopCheckerTest, StopStringAcrossTokensTrimsTrailingText) {
  const Tokenizer tok = MiniTokenizer();
  SamplingParams p = Params(10);
  p.stop_strings = {"ab"};
  auto checker = StopChecker::Create(p, {}, &tok, true);
  ASSERT_TRUE(checker.ok());
  // b, a, b -> text "bab"; the "ab" (spanning tokens 2-3) ends generation, and
  // its own bytes plus anything after are trimmed, leaving "b".
  const DriveResult r = Drive(*checker, {1, 0, 1});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.consumed, 3);
  EXPECT_EQ(r.trigger, StopTrigger::kStopString);
  EXPECT_EQ(r.finish_reason, FinishReason::kStop);
  EXPECT_EQ(r.matched_stop, "ab");
  EXPECT_EQ(r.text, "b");
}

TEST(StopCheckerTest, StopStringWithinSingleToken) {
  const Tokenizer tok = MiniTokenizer();
  SamplingParams p = Params(10);
  p.stop_strings = {"b"};
  auto checker = StopChecker::Create(p, {}, &tok, true);
  ASSERT_TRUE(checker.ok());
  // Token 2 detokenizes to "ab"; the stop "b" matches mid-token, keeping "a".
  const DriveResult r = Drive(*checker, {2, 0});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.consumed, 1);
  EXPECT_EQ(r.trigger, StopTrigger::kStopString);
  EXPECT_EQ(r.text, "a");
}

TEST(StopCheckerTest, EosIdStopsAndIsStopReason) {
  const Tokenizer tok = MiniTokenizer();
  const std::vector<std::int32_t> eos = {3};  // <s>, special
  auto checker = StopChecker::Create(Params(10), eos, &tok, true);
  ASSERT_TRUE(checker.ok());
  const DriveResult r = Drive(*checker, {0, 3, 1});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.consumed, 2);
  EXPECT_EQ(r.trigger, StopTrigger::kEosId);
  EXPECT_EQ(r.finish_reason, FinishReason::kStop);
  EXPECT_EQ(r.text, "a");  // the special EOS contributes no text
}

TEST(StopCheckerTest, StopTokenIdStopsInclusiveOfItsText) {
  const Tokenizer tok = MiniTokenizer();
  SamplingParams p = Params(10);
  p.stop_token_ids = {1};  // "b" — an id-based stop is NOT trimmed
  auto checker = StopChecker::Create(p, {}, &tok, true);
  ASSERT_TRUE(checker.ok());
  const DriveResult r = Drive(*checker, {0, 1, 0});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.consumed, 2);
  EXPECT_EQ(r.trigger, StopTrigger::kStopTokenId);
  EXPECT_EQ(r.text, "ab");  // both "a" and the stop token's "b" are kept
}

TEST(StopCheckerTest, EosBeatsStopStringOnTheSameToken) {
  const Tokenizer tok = MiniTokenizer();
  SamplingParams p = Params(10);
  p.stop_strings = {"ab"};
  const std::vector<std::int32_t> eos = {1};  // "b" also completes "ab"
  auto checker = StopChecker::Create(p, eos, &tok, true);
  ASSERT_TRUE(checker.ok());
  // a, b: "b" is both EOS and completes the stop string "ab". EOS wins
  // (priority), so the trigger is kEosId and the "b" text is kept (not
  // trimmed).
  const DriveResult r = Drive(*checker, {0, 1});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.trigger, StopTrigger::kEosId);
  EXPECT_EQ(r.text, "ab");
}

TEST(StopCheckerTest, MaxTokensReleasesHeldStopPrefix) {
  const Tokenizer tok = MiniTokenizer();
  SamplingParams p = Params(2);
  p.stop_strings = {"abX"};  // never completes; "a" gets held mid-stream
  auto checker = StopChecker::Create(p, {}, &tok, true);
  ASSERT_TRUE(checker.ok());
  // b, a: "a" is held as a prefix of "abX", then max_tokens fires; the held
  // "a" is real output, released by Finish().
  const DriveResult r = Drive(*checker, {1, 0});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.trigger, StopTrigger::kMaxTokens);
  EXPECT_EQ(r.text, "ba");
}

TEST(StopCheckerTest, NoTokenizerProducesNoText) {
  auto checker =
      StopChecker::Create(Params(3), {}, /*tokenizer=*/nullptr, true);
  ASSERT_TRUE(checker.ok());
  const DriveResult r = Drive(*checker, {0, 1, 2, 4});
  EXPECT_TRUE(r.finished);
  EXPECT_EQ(r.trigger, StopTrigger::kMaxTokens);
  EXPECT_EQ(r.text, "");  // token-only path
}

}  // namespace
