#pragma once

#include "core/check.h"
#include "parallel/thread_pool.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

// Deterministic parallel_for / parallel_reduce (M3-T04; design:
// docs/design/cpu-backend.md §3.2–§3.4).
//
// Partitioning is the determinism foundation and is shared by both entry
// points: `num_chunks = ceil(n / grain)`, chunk c covers
// [c·grain, min((c+1)·grain, n)). Chunks depend only on (n, grain) — thread
// count never enters the formula — so chunk boundaries, and therefore
// parallel_reduce's partials and fold order, are reproducible on any machine
// at any thread count. `grain` is a per-call-site constant chosen by the
// kernel (a named constant, never computed from pool size or load) and
// doubles as the serial threshold: when num_chunks <= 1 the body runs inline
// on the caller and the pool is never touched.
//
// Contract for bodies (design §3.4): no exceptions (an escaping exception
// terminates — enforced by invoking bodies through noexcept frames on both
// the inline and pooled paths, so the failure mode does not depend on n vs
// grain; all recoverable-error checking happens before entering a region,
// inside there is only arithmetic and CHECK), and no nesting — a body must
// not call parallel_for/parallel_reduce (CHECK-enforced; nesting means a
// kernel called a kernel, which the layering already forbids).

namespace engine::parallel {

// Non-owning callable reference (design §3.2). v1 spells it as a reference
// to std::function: invocation cost is per *chunk*, not per element, so this
// is not hot; a real non-allocating FunctionRef is a measured-perf follow-up.
template <typename Sig>
using FunctionRef = const std::function<Sig>&;

namespace detail {

// RAII for the thread-local region flag (declared in thread_pool.h so
// ThreadPool::Run can also check it). Bodies never throw (§3.4), so plain
// set/clear is safe.
class RegionGuard {
 public:
  RegionGuard() { SetInsideParallelRegion(true); }
  ~RegionGuard() { SetInsideParallelRegion(false); }
  RegionGuard(const RegionGuard&) = delete;
  RegionGuard& operator=(const RegionGuard&) = delete;
};

[[nodiscard]] constexpr std::int64_t CeilDiv(std::int64_t n, std::int64_t d) {
  return (n / d) + (n % d != 0 ? 1 : 0);
}

// Shared pool-region primitive: runs chunk_fn(c) for c in [0, num_chunks)
// via pool.Run, with the nesting flag set around each chunk execution.
void RunChunks(ThreadPool& pool, std::int64_t num_chunks,
               const std::function<void(std::int64_t)>& chunk_fn);

}  // namespace detail

// Runs body(begin, end) over disjoint sub-ranges covering [0, n), partitioned
// per the rules above. Blocks until every sub-range completed; when the pool
// is used, the calling thread waits and executes no chunks (design §3.2
// rule 5). body must not throw and must write only to locations disjoint
// from other sub-ranges' writes. n >= 0 and grain >= 1 (CHECK). n == 0
// returns immediately without invoking body.
void parallel_for(ThreadPool& pool, std::int64_t n, std::int64_t grain,
                  FunctionRef<void(std::int64_t begin, std::int64_t end)> body);

// Deterministic reduction (design §3.3). partial(begin, end) computes one
// chunk's partial; combine(a, b) folds two partials. Partials land in a
// num_chunks-sized array indexed by chunk id (which worker computed one is
// irrelevant), then the *calling thread* folds the array in a fixed pairwise
// halving tree: s[i] = combine(s[i], s[i + half]) for i < half, the odd
// element carrying to the next pass. The fold order is a pure function of
// num_chunks, hence of (n, grain) — results are bit-identical at any thread
// count. combine need not be associative in exact arithmetic (fp add is
// not); determinism comes from the fixed order.
//
// Returns identity when n == 0 (identity is never combined otherwise). T
// must be default-constructible and copyable. Partial slots are padded to a
// cache line to avoid false sharing — a performance detail with no effect on
// values. When passing lambdas, name T explicitly
// (`parallel_reduce<float>(...)`): it is not deducible through FunctionRef.
template <typename T>
T parallel_reduce(ThreadPool& pool, std::int64_t n, std::int64_t grain,
                  T identity,
                  FunctionRef<T(std::int64_t begin, std::int64_t end)> partial,
                  FunctionRef<T(T a, T b)> combine) {
  CHECK(n >= 0, "parallel_reduce: n must be >= 0 (got {})", n);
  CHECK(grain >= 1, "parallel_reduce: grain must be >= 1 (got {})", grain);
  CHECK(!detail::InsideParallelRegion(), "parallel regions must not nest");
  if (n == 0) {
    return identity;
  }
  const std::int64_t num_chunks = detail::CeilDiv(n, grain);
  if (num_chunks <= 1) {
    const detail::RegionGuard guard;
    // noexcept frame: an escaping exception terminates on the inline path
    // exactly as it does on the pooled path (design §3.4).
    const auto invoke = [&]() noexcept -> T { return partial(0, n); };
    return invoke();
  }
  struct alignas(64) Slot {
    T value;
  };
  std::vector<Slot> slots;
  try {
    slots.resize(static_cast<std::size_t>(num_chunks));
  } catch (const std::bad_alloc&) {
    // ADR-003: no exceptions across the module boundary. A partial-slot
    // allocation this size failing has no recoverable handling at this
    // layer, so convert to CHECK (same stance as ThreadPool's constructor).
    CHECK(false,
          "parallel_reduce: failed to allocate {} partial slots ({} bytes)",
          num_chunks, num_chunks * static_cast<std::int64_t>(sizeof(Slot)));
  }
  detail::RunChunks(pool, num_chunks, [&](std::int64_t chunk) {
    const std::int64_t begin = chunk * grain;
    const std::int64_t end = std::min(begin + grain, n);
    slots[static_cast<std::size_t>(chunk)].value = partial(begin, end);
  });
  // noexcept frame: combine is a body too — same termination contract.
  const auto fold = [&]() noexcept -> T {
    std::int64_t count = num_chunks;
    while (count > 1) {
      const std::int64_t half = count / 2;
      for (std::int64_t i = 0; i < half; ++i) {
        T& low = slots[static_cast<std::size_t>(i)].value;
        low = combine(low, slots[static_cast<std::size_t>(i + half)].value);
      }
      if (count % 2 == 1) {
        // The odd element (index 2·half) carries into the next pass.
        slots[static_cast<std::size_t>(half)].value =
            slots[static_cast<std::size_t>(count - 1)].value;
      }
      count = half + count % 2;
    }
    return slots[0].value;
  };
  return fold();
}

}  // namespace engine::parallel
