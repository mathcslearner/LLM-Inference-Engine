#include "runtime/channel.h"

#include "engine/stop.h"
#include "runtime/request.h"
#include "sampling/logprobs.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// OutputChannel tests (M9-T02; design: docs/design/scheduler-runtime.md §4).
// The per-request SPSC queue: ordered delivery across a producer thread and a
// consumer thread, blocking `Next` waking on push and on close, non-blocking
// `TryNext`, close-once under a finish + late-cancel race, and buffered items
// draining after close. Pure threading over portable C++ — no model, kernel, or
// ISA dependency, so no SCALAR_PASS. `runtime` label.
//
// Optionals are dereferenced via `.value()` (a checked access) rather than
// `operator*`/`operator->`, satisfying bugprone-unchecked-optional-access — the
// repo idiom (e.g. stop_test).

namespace {

using engine::engine::FinishReason;
using engine::engine::StopTrigger;
using engine::runtime::FinishInfo;
using engine::runtime::OutputChannel;
using engine::runtime::OutputItem;
using engine::runtime::SeqState;

[[nodiscard]] FinishInfo FinishedInfo() {
  return FinishInfo{.terminal = SeqState::kFinished,
                    .finish_reason = FinishReason::kStop,
                    .stop_trigger = StopTrigger::kEosId,
                    .matched_stop = "",
                    .error = {}};
}

[[nodiscard]] OutputItem Tok(std::int32_t id) {
  return OutputItem{
      .token_id = id, .text_delta = std::to_string(id), .logprobs = {}};
}

// Dereference an optional the analyzer can prove engaged (the explicit-guard
// idiom bugprone-unchecked-optional-access requires — gtest's ASSERT_TRUE is
// not modeled as a control-flow guard; cf. batched_sampler_test).
template <typename T>
[[nodiscard]] T Require(std::optional<T> opt) {
  if (!opt.has_value()) {
    ADD_FAILURE() << "expected an engaged optional";
    return T{};
  }
  return std::move(opt).value();
}

TEST(OutputChannelTest, TryNextOnEmptyOpenIsNullopt) {
  OutputChannel ch;
  EXPECT_FALSE(ch.closed());
  EXPECT_FALSE(ch.finish().has_value());
  EXPECT_EQ(ch.size(), 0U);
  EXPECT_FALSE(ch.TryNext().has_value());
}

TEST(OutputChannelTest, PushThenTryNextFifo) {
  OutputChannel ch;
  ch.Push(Tok(10));
  ch.Push(Tok(20));
  ch.Push(Tok(30));
  EXPECT_EQ(ch.size(), 3U);

  const OutputItem a = Require(ch.TryNext());
  const OutputItem b = Require(ch.TryNext());
  const OutputItem c = Require(ch.TryNext());
  EXPECT_EQ(a.token_id, 10);
  EXPECT_EQ(b.token_id, 20);
  EXPECT_EQ(c.token_id, 30);
  EXPECT_FALSE(ch.TryNext().has_value());
}

TEST(OutputChannelTest, LogprobsPassthrough) {
  OutputChannel ch;
  OutputItem item{.token_id = 7, .text_delta = "x", .logprobs = {}};
  item.logprobs = engine::sampling::StepLogprobs{};
  item.logprobs.value().chosen_logprob = -1.5F;
  item.logprobs.value().top.push_back({.id = 7, .logprob = -1.5F});
  ch.Push(std::move(item));

  const OutputItem got = Require(ch.TryNext());
  const engine::sampling::StepLogprobs lp = Require(got.logprobs);
  EXPECT_FLOAT_EQ(lp.chosen_logprob, -1.5F);
  ASSERT_EQ(lp.top.size(), 1U);
  EXPECT_EQ(lp.top[0].id, 7);
}

TEST(OutputChannelTest, FinishSetOnlyAfterClose) {
  OutputChannel ch;
  EXPECT_FALSE(ch.finish().has_value());
  EXPECT_TRUE(ch.Close(FinishedInfo()));
  const FinishInfo fin = Require(ch.finish());
  EXPECT_EQ(fin.terminal, SeqState::kFinished);
  EXPECT_EQ(fin.finish_reason, FinishReason::kStop);
  EXPECT_TRUE(ch.closed());
}

TEST(OutputChannelTest, CloseOnceFirstWins) {
  OutputChannel ch;
  EXPECT_TRUE(ch.Close(FinishedInfo()));

  const FinishInfo cancel{.terminal = SeqState::kCancelled,
                          .finish_reason = FinishReason::kLength,
                          .stop_trigger = StopTrigger::kNone,
                          .matched_stop = "",
                          .error = {}};
  EXPECT_FALSE(ch.Close(cancel));  // late cancel is a no-op.

  const FinishInfo fin = Require(ch.finish());
  EXPECT_EQ(fin.terminal, SeqState::kFinished);  // first close held.
}

TEST(OutputChannelTest, BufferedItemsDrainAfterClose) {
  OutputChannel ch;
  ch.Push(Tok(1));
  ch.Push(Tok(2));
  ch.Push(Tok(3));
  ch.Close(FinishedInfo());

  // Both the blocking and polling readers see all three, then nullopt.
  const OutputItem a = Require(ch.Next());
  const OutputItem b = Require(ch.Next());
  EXPECT_EQ(a.token_id, 1);
  EXPECT_EQ(b.token_id, 2);

  const OutputItem c = Require(ch.TryNext());
  EXPECT_EQ(c.token_id, 3);

  EXPECT_FALSE(ch.Next().has_value());  // closed and drained.
  EXPECT_FALSE(ch.TryNext().has_value());
}

TEST(OutputChannelTest, NextOnClosedEmptyReturnsNullopt) {
  OutputChannel ch;
  ch.Close(FinishedInfo());
  EXPECT_FALSE(ch.Next().has_value());
}

TEST(OutputChannelDeathTest, PushAfterCloseChecks) {
  OutputChannel ch;
  ch.Close(FinishedInfo());
  EXPECT_DEATH(ch.Push(Tok(1)), "closed");
}

// Blocking Next wakes when the producer pushes after the consumer has blocked.
TEST(OutputChannelTest, NextBlocksUntilPush) {
  OutputChannel ch;
  std::atomic<bool> consumer_started{false};
  std::optional<OutputItem> got;

  std::thread consumer([&] {
    consumer_started.store(true);
    got = ch.Next();  // blocks until the push below.
  });

  while (!consumer_started.load()) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ch.Push(Tok(42));
  consumer.join();

  EXPECT_EQ(Require(got).token_id, 42);
}

// Blocking Next wakes when the producer closes with an empty queue.
TEST(OutputChannelTest, NextWakesOnClose) {
  OutputChannel ch;
  std::atomic<bool> consumer_started{false};
  std::optional<OutputItem> got;

  std::thread consumer([&] {
    consumer_started.store(true);
    got = ch.Next();  // blocks until the close below.
  });

  while (!consumer_started.load()) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ch.Close(FinishedInfo());
  consumer.join();

  EXPECT_FALSE(got.has_value());
  EXPECT_EQ(Require(ch.finish()).terminal, SeqState::kFinished);
}

// Ordered delivery across a producer thread and a consumer thread: the consumer
// sees exactly the produced ids, in order, then the terminal FinishInfo (the
// M9-T02 acceptance test).
TEST(OutputChannelTest, OrderedDeliveryAcrossThreads) {
  OutputChannel ch;
  constexpr std::int32_t kN = 10000;

  std::thread producer([&] {
    for (std::int32_t i = 0; i < kN; ++i) {
      ch.Push(Tok(i));
    }
    ch.Close(FinishedInfo());
  });

  std::vector<std::int32_t> received;
  received.reserve(kN);
  while (std::optional<OutputItem> item = ch.Next()) {
    received.push_back(Require(std::move(item)).token_id);
  }
  producer.join();

  ASSERT_EQ(received.size(), static_cast<std::size_t>(kN));
  for (std::int32_t i = 0; i < kN; ++i) {
    ASSERT_EQ(received[static_cast<std::size_t>(i)], i);
  }
  EXPECT_EQ(Require(ch.finish()).terminal, SeqState::kFinished);
}

}  // namespace
