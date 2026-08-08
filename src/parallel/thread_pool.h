#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Thread pool (M3-T04; design: docs/design/cpu-backend.md §3).
//
// `parallel` is the only module in the engine that creates compute threads
// (design §3.4). Kernels never touch ThreadPool directly — they express
// parallelism exclusively through parallel_for / parallel_reduce
// (parallel/parallel_for.h), which take a pool and schedule index ranges
// over it. This module knows nothing of tensors.

namespace engine::parallel {

// One persistent, fixed-size worker pool. Workers park on a condition
// variable between regions; there is no busy-spinning (design §3.1). The
// pool runs one region at a time: concurrent submissions from different
// caller threads are serialized internally.
class ThreadPool {
 public:
  // Spawns `num_threads` workers (CHECK: >= 1). A failed thread spawn is a
  // CHECK failure — a process that cannot create threads at startup has no
  // meaningful degraded mode (design §3.4).
  explicit ThreadPool(int num_threads);

  // Joins all workers. Must not run concurrently with an active region.
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  [[nodiscard]] int num_threads() const { return num_threads_; }

  // Implementation seam for parallel_for / parallel_reduce — the only
  // intended callers (kernels use those free functions, never Run). Executes
  // chunk_fn(c) for every chunk id c in [0, num_chunks) with the static
  // round-robin assignment from design §3.2: worker w of T runs chunks
  // w, w+T, w+2T, … Blocks until every chunk completed; the calling thread
  // executes no chunks (design §3.2 rule 5). chunk_fn must not throw.
  // Calling Run from inside a parallel region is a CHECK failure (M3 audit):
  // without the check, a worker calling Run would deadlock silently on the
  // pool's region serialization.
  void Run(std::int64_t num_chunks,
           const std::function<void(std::int64_t)>& chunk_fn);

 private:
  void WorkerLoop(int worker_index);

  const int num_threads_;

  // Serializes callers: one region in flight at a time.
  std::mutex run_mutex_;

  // Guards everything below; work_cv_ wakes workers (new region or stop),
  // done_cv_ wakes the caller when the last worker finishes.
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::uint64_t region_generation_ = 0;
  std::int64_t region_num_chunks_ = 0;
  const std::function<void(std::int64_t)>* region_chunk_fn_ = nullptr;
  int workers_remaining_ = 0;
  bool stopping_ = false;

  std::vector<std::thread> workers_;
};

// The process-wide pool used by kernels: lazily constructed function-local
// static, sized by detail::DefaultPoolSize() on first use.
ThreadPool& DefaultPool();

// Physical (not logical) cores: macOS via sysctl("hw.physicalcpu"); Linux
// via sysfs topology (counting unique cores, so SMT siblings collapse);
// fallback std::thread::hardware_concurrency(). Never returns < 1. SMT
// siblings contend for the memory bandwidth and FMA ports our kernels are
// bound on, so the default pool deliberately ignores them (design §3.1).
[[nodiscard]] int physical_core_count();

namespace detail {

// The size DefaultPool() is constructed with: ENGINE_NUM_THREADS if set
// (numeric values below 1 clamp to 1; a non-numeric value is fatal at
// startup with an actionable message — same stance as ENGINE_FORCE_ISA,
// design §4.3; empty counts as unset), else physical_core_count(). Exposed
// for tests; DefaultPool() reads it exactly once.
[[nodiscard]] int DefaultPoolSize();

// Thread-local "currently executing a region body" flag (design §3.4). Set
// on workers while executing a chunk and on the caller for the inline path
// (parallel_for.h's RegionGuard); backs the nesting CHECK in
// parallel_for / parallel_reduce and the self-use CHECK in ThreadPool::Run.
[[nodiscard]] bool InsideParallelRegion();
void SetInsideParallelRegion(bool inside);

}  // namespace detail

}  // namespace engine::parallel
