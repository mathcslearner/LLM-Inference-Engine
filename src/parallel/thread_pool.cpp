#include "parallel/thread_pool.h"

#include "core/check.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <system_error>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#include <set>
#include <string>
#include <utility>
#endif

namespace engine::parallel {

ThreadPool::ThreadPool(int num_threads) : num_threads_(num_threads) {
  CHECK(num_threads >= 1, "ThreadPool requires num_threads >= 1 (got {})",
        num_threads);
  workers_.reserve(static_cast<std::size_t>(num_threads));
  // std::thread's constructor throws std::system_error when the OS cannot
  // spawn a thread. ADR-003 forbids exceptions across module boundaries, and
  // pool lifecycle errors are CHECK by design (§3.4), so convert here.
  try {
    for (int w = 0; w < num_threads; ++w) {
      workers_.emplace_back([this, w] { WorkerLoop(w); });
    }
  } catch (const std::system_error& error) {
    CHECK(false, "ThreadPool failed to spawn a worker thread: {}",
          error.what());
  }
}

ThreadPool::~ThreadPool() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  work_cv_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::Run(std::int64_t num_chunks,
                     const std::function<void(std::int64_t)>& chunk_fn) {
  CHECK(num_chunks >= 1, "Run requires num_chunks >= 1 (got {})", num_chunks);
  const std::lock_guard<std::mutex> run_lock(run_mutex_);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    region_num_chunks_ = num_chunks;
    region_chunk_fn_ = &chunk_fn;
    workers_remaining_ = num_threads_;
    ++region_generation_;
  }
  work_cv_.notify_all();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return workers_remaining_ == 0; });
    region_chunk_fn_ = nullptr;
  }
}

void ThreadPool::WorkerLoop(int worker_index) {
  std::uint64_t seen_generation = 0;
  for (;;) {
    const std::function<void(std::int64_t)>* chunk_fn = nullptr;
    std::int64_t num_chunks = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [&] {
        return stopping_ || region_generation_ != seen_generation;
      });
      if (stopping_) {
        return;
      }
      seen_generation = region_generation_;
      chunk_fn = region_chunk_fn_;
      num_chunks = region_num_chunks_;
    }
    // Static round-robin assignment (design §3.2 rule 3): worker w of T
    // executes chunks w, w+T, w+2T, … No stealing, no dynamic queues.
    for (std::int64_t chunk = worker_index; chunk < num_chunks;
         chunk += num_threads_) {
      (*chunk_fn)(chunk);
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (--workers_remaining_ == 0) {
        done_cv_.notify_all();
      }
    }
  }
}

ThreadPool& DefaultPool() {
  static ThreadPool pool(detail::DefaultPoolSize());
  return pool;
}

#if defined(__linux__)
namespace {

// Counts unique (package, core) pairs from sysfs so SMT siblings collapse to
// one. Returns 0 when the topology tree is absent (containers, exotic
// kernels); the caller falls back.
int LinuxPhysicalCoreCount() {
  std::set<std::pair<int, int>> cores;
  for (int cpu = 0;; ++cpu) {
    const std::string base =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream package_file(base + "physical_package_id");
    std::ifstream core_file(base + "core_id");
    int package = -1;
    int core = -1;
    if (!(package_file >> package) || !(core_file >> core)) {
      break;
    }
    cores.emplace(package, core);
  }
  return static_cast<int>(cores.size());
}

}  // namespace
#endif

int physical_core_count() {
#if defined(__APPLE__)
  int count = 0;
  std::size_t size = sizeof(count);
  if (::sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0 &&
      count >= 1) {
    return count;
  }
#elif defined(__linux__)
  const int count = LinuxPhysicalCoreCount();
  if (count >= 1) {
    return count;
  }
#endif
  const unsigned int logical = std::thread::hardware_concurrency();
  return logical >= 1U ? static_cast<int>(logical) : 1;
}

namespace detail {

int DefaultPoolSize() {
  const char* env = std::getenv("ENGINE_NUM_THREADS");
  if (env == nullptr || *env == '\0') {
    return physical_core_count();
  }
  char* end = nullptr;
  errno = 0;
  const auto value = static_cast<std::int64_t>(std::strtol(env, &end, 10));
  // Silently ignoring developer/test configuration is the failure mode
  // ENGINE_FORCE_ISA's design rejects (§4.3); apply the same stance here.
  CHECK(end != env && *end == '\0' && errno == 0,
        "ENGINE_NUM_THREADS must be an integer thread count, got \"{}\"", env);
  return static_cast<int>(
      std::clamp<std::int64_t>(value, 1, std::numeric_limits<int>::max()));
}

}  // namespace detail

}  // namespace engine::parallel
