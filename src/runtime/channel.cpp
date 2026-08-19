#include "runtime/channel.h"

#include "core/check.h"

#include <mutex>
#include <optional>
#include <utility>

namespace engine::runtime {

void OutputChannel::Push(OutputItem item) {
  {
    const std::lock_guard<std::mutex> lock(mu_);
    CHECK(!closed_, "Push on a closed OutputChannel");
    queue_.push_back(std::move(item));
  }
  cv_.notify_one();
}

bool OutputChannel::Close(FinishInfo info) {
  {
    const std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      return false;  // already closed — the first close wins (§4).
    }
    closed_ = true;
    finish_ = std::move(info);
  }
  cv_.notify_all();
  return true;
}

std::optional<OutputItem> OutputChannel::Next() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return !queue_.empty() || closed_; });
  if (queue_.empty()) {
    return std::nullopt;  // closed and drained.
  }
  OutputItem item = std::move(queue_.front());
  queue_.pop_front();
  return item;
}

std::optional<OutputItem> OutputChannel::TryNext() {
  const std::lock_guard<std::mutex> lock(mu_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  OutputItem item = std::move(queue_.front());
  queue_.pop_front();
  return item;
}

bool OutputChannel::closed() const {
  const std::lock_guard<std::mutex> lock(mu_);
  return closed_;
}

std::optional<FinishInfo> OutputChannel::finish() const {
  const std::lock_guard<std::mutex> lock(mu_);
  return finish_;
}

std::size_t OutputChannel::size() const {
  const std::lock_guard<std::mutex> lock(mu_);
  return queue_.size();
}

}  // namespace engine::runtime
