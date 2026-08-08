#include "parallel/parallel_for.h"

#include "core/check.h"
#include "parallel/thread_pool.h"

#include <algorithm>
#include <cstdint>
#include <functional>

namespace engine::parallel {

namespace detail {

void RunChunks(ThreadPool& pool, std::int64_t num_chunks,
               const std::function<void(std::int64_t)>& chunk_fn) {
  // noexcept frame: an escaping body exception terminates here rather than
  // unwinding into the worker loop (design §3.4).
  pool.Run(num_chunks, [&chunk_fn](std::int64_t chunk) noexcept {
    const RegionGuard guard;
    chunk_fn(chunk);
  });
}

}  // namespace detail

void parallel_for(
    ThreadPool& pool, std::int64_t n, std::int64_t grain,
    FunctionRef<void(std::int64_t begin, std::int64_t end)> body) {
  CHECK(n >= 0, "parallel_for: n must be >= 0 (got {})", n);
  CHECK(grain >= 1, "parallel_for: grain must be >= 1 (got {})", grain);
  CHECK(!detail::InsideParallelRegion(), "parallel regions must not nest");
  if (n == 0) {
    return;
  }
  const std::int64_t num_chunks = detail::CeilDiv(n, grain);
  if (num_chunks <= 1) {
    const detail::RegionGuard guard;
    // noexcept frame: an escaping exception terminates on the inline path
    // exactly as it does on the pooled path (design §3.4).
    const auto invoke = [&]() noexcept { body(0, n); };
    invoke();
    return;
  }
  detail::RunChunks(pool, num_chunks, [&](std::int64_t chunk) {
    const std::int64_t begin = chunk * grain;
    body(begin, std::min(begin + grain, n));
  });
}

}  // namespace engine::parallel
