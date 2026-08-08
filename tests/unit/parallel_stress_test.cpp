#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

// M3-T04 stress tests — the primary TSAN target (design §8.2/§9): thousands
// of small regions back-to-back, the region-handoff pattern kernels will
// actually produce, plus concurrent callers hammering one pool. Kept in its
// own TU so the slow tests stay isolated from the fast parallel suite.

namespace engine::parallel {
namespace {

constexpr std::int64_t kN = 256;
constexpr std::int64_t kGrain = 16;  // 16 chunks per region
constexpr std::int64_t kExpectedSum = kN * (kN - 1) / 2;

[[nodiscard]] std::int64_t RangeSum(std::int64_t begin, std::int64_t end) {
  std::int64_t acc = 0;
  for (std::int64_t i = begin; i < end; ++i) {
    acc += i;
  }
  return acc;
}

TEST(ParallelStressTest, ManySmallRegionsBackToBack) {
  ThreadPool pool(4);
  for (int iteration = 0; iteration < 2000; ++iteration) {
    std::atomic<std::int64_t> for_sum{0};
    parallel_for(pool, kN, kGrain, [&](std::int64_t begin, std::int64_t end) {
      for_sum.fetch_add(RangeSum(begin, end));
    });
    ASSERT_EQ(for_sum.load(), kExpectedSum) << "iteration " << iteration;

    const auto reduce_sum = parallel_reduce<std::int64_t>(
        pool, kN, kGrain, 0,
        [](std::int64_t begin, std::int64_t end) {
          return RangeSum(begin, end);
        },
        [](std::int64_t a, std::int64_t b) { return a + b; });
    ASSERT_EQ(reduce_sum, kExpectedSum) << "iteration " << iteration;
  }
}

TEST(ParallelStressTest, ConcurrentCallersAreSerializedSafely) {
  ThreadPool pool(4);
  std::atomic<int> failures{0};
  std::vector<std::thread> callers;
  callers.reserve(4);
  for (int caller = 0; caller < 4; ++caller) {
    callers.emplace_back([&pool, &failures] {
      for (int iteration = 0; iteration < 250; ++iteration) {
        const auto sum = parallel_reduce<std::int64_t>(
            pool, kN, kGrain, 0,
            [](std::int64_t begin, std::int64_t end) {
              return RangeSum(begin, end);
            },
            [](std::int64_t a, std::int64_t b) { return a + b; });
        if (sum != kExpectedSum) {
          failures.fetch_add(1);
        }
      }
    });
  }
  for (std::thread& caller : callers) {
    caller.join();
  }
  EXPECT_EQ(failures.load(), 0);
}

}  // namespace
}  // namespace engine::parallel
