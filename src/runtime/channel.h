#pragma once

#include "runtime/request.h"
#include "sampling/logprobs.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

// Per-request output channel (M9-T02; design: docs/design/scheduler-runtime.md
// §4). Each `Sequence` owns one `OutputChannel` — a thread-safe queue the
// single engine thread writes and one client thread reads. The forward pass
// dominates a step by orders of magnitude, so the queue lock is never contended
// on the hot path; a lock-free MPSC is a measured follow-up (§14), not v1.

namespace engine::runtime {

// One produced token's worth of output — mirrors the single-sequence
// `engine::TokenEvent` (generator.h) by value (it crosses a thread boundary, so
// the text delta is owned, not a view). The concatenation of every item's
// `text_delta` equals the sequence's full detokenized text (the M7-T04
// streaming invariant M10 relies on).
struct OutputItem {
  std::int32_t token_id = 0;
  std::string text_delta;  // safe-to-emit bytes (may be empty)
  std::optional<sampling::StepLogprobs> logprobs;  // when params.logprobs > 0
};

// A single-producer / single-consumer queue with blocking and polling
// consumption, closed exactly once with a `FinishInfo`. Unbounded: the producer
// (the engine thread) must never block on a slow client, which would stall
// every other sequence in the batch (§4). A bounded channel with backpressure
// is M10's decision, made where the socket write rate is visible.
class OutputChannel {
 public:
  OutputChannel() = default;
  OutputChannel(const OutputChannel&) = delete;
  OutputChannel& operator=(const OutputChannel&) = delete;
  OutputChannel(OutputChannel&&) = delete;
  OutputChannel& operator=(OutputChannel&&) = delete;
  ~OutputChannel() = default;

  // --- producer (engine thread) -------------------------------------------

  // Enqueue one produced token and wake a blocked consumer. CHECK-fails if the
  // channel is already closed (pushing after close is a programmer error — the
  // loop checks `closed()` before producing on a cancelled sequence, §11.1).
  void Push(OutputItem item);

  // Close the channel with its finish descriptor. Idempotent-guarded: the first
  // call records `info`, closes, and wakes all waiters, returning `true`; any
  // later call is a no-op returning `false` (the finish-then-late-cancel race,
  // §4). The recorded `info` is the winning first close.
  bool Close(FinishInfo info);

  // --- consumer (client thread) -------------------------------------------

  // Block until an item is available or the channel closes. Returns the next
  // buffered item, or `nullopt` once the channel is closed and drained.
  [[nodiscard]] std::optional<OutputItem> Next();

  // Non-blocking poll: the next buffered item if one is queued, else `nullopt`
  // (whether open-and-empty or closed-and-drained — distinguish via `closed()`
  // / `finish()`). The seam M10's SSE loop uses to interleave disconnect
  // checks.
  [[nodiscard]] std::optional<OutputItem> TryNext();

  // --- status --------------------------------------------------------------

  [[nodiscard]] bool closed() const;
  // The finish descriptor, set once the channel is closed (else `nullopt`).
  [[nodiscard]] std::optional<FinishInfo> finish() const;
  // Number of items currently buffered (for tests).
  [[nodiscard]] std::size_t size() const;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<OutputItem> queue_;
  bool closed_ = false;
  std::optional<FinishInfo> finish_;
};

}  // namespace engine::runtime
