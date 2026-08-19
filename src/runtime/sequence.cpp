#include "core/check.h"
#include "core/status.h"
#include "engine/stop.h"
#include "kvcache/block_pool.h"
#include "kvcache/paged_cache.h"
#include "runtime/channel.h"
#include "runtime/request.h"
#include "sampling/sampler.h"

#include <memory>
#include <string_view>
#include <utility>

namespace engine::runtime {

std::string_view SeqStateName(SeqState state) {
  switch (state) {
    case SeqState::kWaiting:
      return "waiting";
    case SeqState::kRunning:
      return "running";
    case SeqState::kPreempted:
      return "preempted";
    case SeqState::kFinished:
      return "finished";
    case SeqState::kCancelled:
      return "cancelled";
    case SeqState::kFailed:
      return "failed";
  }
  return "unknown";
}

core::StatusOr<Sequence> Sequence::Create(const Request& request,
                                          kvcache::BlockPool* pool) {
  CHECK(pool != nullptr, "Sequence::Create requires a non-null BlockPool");

  // Front-load the per-sequence sampler and stop bookkeeping: a malformed
  // request (bad params, or stop-strings without a tokenizer) fails here, not
  // mid-forward. Resolving the sampler also fixes the RNG seed once.
  auto sampler = sampling::Sampler::Create(request.params);
  if (!sampler.ok()) {
    return sampler.status();
  }
  auto stop =
      StopChecker::Create(request.params, request.eos_ids, request.tokenizer,
                          request.skip_special_tokens);
  if (!stop.ok()) {
    return stop.status();
  }

  auto cache = std::make_unique<kvcache::PagedKvCache>(pool);
  auto channel = std::make_shared<OutputChannel>();
  return Sequence(request, std::move(cache), *std::move(sampler),
                  *std::move(stop), std::move(channel));
}

Sequence::Sequence(const Request& request,
                   std::unique_ptr<kvcache::PagedKvCache> cache,
                   sampling::Sampler sampler, StopChecker stop,
                   std::shared_ptr<OutputChannel> channel)
    : request_(&request),
      cache_(std::move(cache)),
      sampler_(std::move(sampler)),
      stop_(std::move(stop)),
      channel_(std::move(channel)) {}

Sequence::Sequence(Sequence&&) noexcept = default;
Sequence::~Sequence() = default;

void Sequence::Transition(SeqState next) {
  CHECK(IsLegalTransition(state_, next), "illegal transition: {} -> {}",
        SeqStateName(state_), SeqStateName(next));
  state_ = next;
}

FinishInfo Sequence::MakeFinishInfo() const {
  CHECK(is_terminal(), "MakeFinishInfo in non-terminal state {}",
        SeqStateName(state_));
  return FinishInfo{.terminal = state_,
                    .finish_reason = finish_reason_,
                    .stop_trigger = stop_trigger_,
                    .matched_stop = matched_stop_,
                    .error = error_};
}

}  // namespace engine::runtime
