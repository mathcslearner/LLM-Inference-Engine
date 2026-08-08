#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// M3-T04 tests (design: docs/design/cpu-backend.md §3, §9): range/chunk edge
// cases, the parallel_reduce determinism guarantee, pool lifecycle, and the
// nesting CHECK. The many-small-regions stress test lives in
// parallel_stress_test.cpp (the TSAN target).

namespace engine::parallel {
namespace {

[[nodiscard]] std::uint32_t Bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

// Mixed magnitudes so fp fold order visibly changes the sum (design §3.3):
// 1e30 + 1 - 1e30 is 0 or 1 depending on association.
[[nodiscard]] std::vector<float> AdversarialData(std::int64_t n) {
  static constexpr float kPattern[] = {1e30F, 1.0F,  -1e30F, 1e-8F,
                                       0.25F, -3.5F, 1e20F,  -1e20F};
  std::vector<float> data(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = kPattern[i % 8];
  }
  return data;
}

// Independent re-implementation of the specified reduction: left-to-right
// partial per chunk, then the fixed pairwise halving fold with the odd
// element carrying. Written in a deliberately different shape from the
// production in-place fold (a fresh vector per round instead of in-place
// halving over one buffer), so an aliasing or carry-order bug in either
// implementation breaks the agreement instead of being replicated (M3
// audit: a line-for-line copy of the fold would only pin serial-vs-parallel
// self-consistency, not the tree spec).
[[nodiscard]] float ReferenceTreeSum(const std::vector<float>& data,
                                     std::int64_t grain) {
  const auto n = static_cast<std::int64_t>(data.size());
  std::vector<float> level;
  for (std::int64_t begin = 0; begin < n; begin += grain) {
    const std::int64_t end = std::min(begin + grain, n);
    float acc = 0.0F;
    for (std::int64_t i = begin; i < end; ++i) {
      acc += data[static_cast<std::size_t>(i)];
    }
    level.push_back(acc);
  }
  while (level.size() > 1) {
    const std::size_t half = level.size() / 2;
    std::vector<float> next;
    next.reserve(half + (level.size() % 2));
    for (std::size_t i = 0; i < half; ++i) {
      next.push_back(level[i] + level[i + half]);
    }
    if (level.size() % 2 == 1) {
      // The odd element carries into the next round at index `half`.
      next.push_back(level.back());
    }
    level = std::move(next);
  }
  return level[0];
}

[[nodiscard]] float PooledSum(ThreadPool& pool, const std::vector<float>& data,
                              std::int64_t grain) {
  return parallel_reduce<float>(
      pool, static_cast<std::int64_t>(data.size()), grain, 0.0F,
      [&data](std::int64_t begin, std::int64_t end) {
        float acc = 0.0F;
        for (std::int64_t i = begin; i < end; ++i) {
          acc += data[static_cast<std::size_t>(i)];
        }
        return acc;
      },
      [](float a, float b) { return a + b; });
}

TEST(PhysicalCoreCountTest, IsAtLeastOne) {
  EXPECT_GE(physical_core_count(), 1);
}

TEST(ThreadPoolTest, ReportsRequestedThreadCount) {
  const ThreadPool pool(3);
  EXPECT_EQ(pool.num_threads(), 3);
}

TEST(ThreadPoolTest, DefaultPoolIsASingletonWithAtLeastOneThread) {
  EXPECT_EQ(&DefaultPool(), &DefaultPool());
  EXPECT_GE(DefaultPool().num_threads(), 1);
}

TEST(ThreadPoolTest, RestartAndShutdownAreClean) {
  // Even iterations run a region before destruction; odd iterations destroy
  // a pool that never ran one.
  for (int iteration = 0; iteration < 10; ++iteration) {
    for (const int threads : {1, 2, 8}) {
      ThreadPool pool(threads);
      if (iteration % 2 == 0) {
        std::atomic<int> calls{0};
        parallel_for(pool, 64, 4, [&](std::int64_t begin, std::int64_t end) {
          calls.fetch_add(static_cast<int>(end - begin));
        });
        ASSERT_EQ(calls.load(), 64);
      }
    }
  }
}

TEST(ThreadPoolDeathTest, ZeroThreadsDies) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH({ const ThreadPool pool(0); }, "num_threads >= 1");
}

TEST(ParallelForTest, EmptyRangeNeverInvokesBody) {
  ThreadPool pool(2);
  int calls = 0;
  parallel_for(pool, 0, 16, [&](std::int64_t, std::int64_t) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(ParallelForTest, SingleElementRange) {
  ThreadPool pool(2);
  std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
  parallel_for(pool, 1, 16, [&](std::int64_t begin, std::int64_t end) {
    ranges.emplace_back(begin, end);
  });
  ASSERT_EQ(ranges.size(), 1U);
  EXPECT_EQ(ranges[0], (std::pair<std::int64_t, std::int64_t>{0, 1}));
}

TEST(ParallelForTest, RangeBelowGrainRunsInlineOnCaller) {
  ThreadPool pool(4);
  std::vector<std::thread::id> ids;
  std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
  parallel_for(pool, 10, 100, [&](std::int64_t begin, std::int64_t end) {
    ids.push_back(std::this_thread::get_id());
    ranges.emplace_back(begin, end);
  });
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(ids[0], std::this_thread::get_id());
  EXPECT_EQ(ranges[0], (std::pair<std::int64_t, std::int64_t>{0, 10}));
}

TEST(ParallelForTest, CallerExecutesNoChunksWhenPooled) {
  ThreadPool pool(2);
  std::mutex mutex;
  std::set<std::thread::id> ids;
  parallel_for(pool, 100, 10, [&](std::int64_t, std::int64_t) {
    const std::lock_guard<std::mutex> lock(mutex);
    ids.insert(std::this_thread::get_id());
  });
  EXPECT_EQ(ids.count(std::this_thread::get_id()), 0U);
}

TEST(ParallelForTest, ChunkBoundsMatchThePartitioningFormula) {
  ThreadPool pool(3);
  // (n, grain) covering: exact multiple, remainder tail, grain == 1,
  // n < num_threads * grain, and a single-chunk inline case.
  const std::vector<std::pair<std::int64_t, std::int64_t>> combos = {
      {100, 10}, {101, 10}, {7, 1}, {5, 2}, {4, 4}};
  for (const auto& [n, grain] : combos) {
    std::mutex mutex;
    std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
    parallel_for(pool, n, grain, [&](std::int64_t begin, std::int64_t end) {
      const std::lock_guard<std::mutex> lock(mutex);
      ranges.emplace_back(begin, end);
    });
    std::vector<std::pair<std::int64_t, std::int64_t>> expected;
    for (std::int64_t begin = 0; begin < n; begin += grain) {
      expected.emplace_back(begin, std::min(begin + grain, n));
    }
    std::ranges::sort(ranges);
    EXPECT_EQ(ranges, expected) << "n=" << n << " grain=" << grain;
  }
}

TEST(ParallelForTest, EveryIndexVisitedExactlyOnce) {
  ThreadPool pool(8);
  for (const auto& [n, grain] :
       std::vector<std::pair<std::int64_t, std::int64_t>>{
           {1, 1}, {17, 3}, {4096, 64}, {100'000, 4096}}) {
    std::vector<std::atomic<int>> visits(static_cast<std::size_t>(n));
    parallel_for(pool, n, grain, [&](std::int64_t begin, std::int64_t end) {
      for (std::int64_t i = begin; i < end; ++i) {
        visits[static_cast<std::size_t>(i)].fetch_add(1);
      }
    });
    for (std::int64_t i = 0; i < n; ++i) {
      ASSERT_EQ(visits[static_cast<std::size_t>(i)].load(), 1)
          << "index " << i << " (n=" << n << " grain=" << grain << ")";
    }
  }
}

TEST(ParallelReduceTest, EmptyRangeReturnsIdentity) {
  ThreadPool pool(2);
  const int result = parallel_reduce<int>(
      pool, 0, 16, -42, [](std::int64_t, std::int64_t) { return 7; },
      [](int a, int b) { return a + b; });
  EXPECT_EQ(result, -42);
}

TEST(ParallelReduceTest, SingleChunkRunsInline) {
  ThreadPool pool(2);
  std::vector<std::thread::id> ids;
  const auto result = parallel_reduce<std::int64_t>(
      pool, 10, 100, 0,
      [&](std::int64_t begin, std::int64_t end) {
        ids.push_back(std::this_thread::get_id());
        return end - begin;
      },
      [](std::int64_t a, std::int64_t b) { return a + b; });
  EXPECT_EQ(result, 10);
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(ids[0], std::this_thread::get_id());
}

TEST(ParallelReduceTest, LargeIntegerSumIsExact) {
  ThreadPool pool(8);
  const std::int64_t n = 1'000'000;
  for (const std::int64_t grain : {1'000'000L, 4096L, 4095L, 1L}) {
    const auto total = parallel_reduce<std::int64_t>(
        pool, n, grain, 0,
        [](std::int64_t begin, std::int64_t end) {
          std::int64_t acc = 0;
          for (std::int64_t i = begin; i < end; ++i) {
            acc += i;
          }
          return acc;
        },
        [](std::int64_t a, std::int64_t b) { return a + b; });
    EXPECT_EQ(total, n * (n - 1) / 2) << "grain=" << grain;
  }
}

TEST(ParallelReduceTest, MaxReduction) {
  ThreadPool pool(4);
  const std::int64_t n = 10'000;
  std::vector<int> data(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    data[static_cast<std::size_t>(i)] = static_cast<int>((i * 7919) % 104729);
  }
  const int expected = *std::ranges::max_element(data);
  const int result = parallel_reduce<int>(
      pool, n, 128, std::numeric_limits<int>::min(),
      [&data](std::int64_t begin, std::int64_t end) {
        int acc = std::numeric_limits<int>::min();
        for (std::int64_t i = begin; i < end; ++i) {
          acc = std::max(acc, data[static_cast<std::size_t>(i)]);
        }
        return acc;
      },
      [](int a, int b) { return std::max(a, b); });
  EXPECT_EQ(result, expected);
}

// The M3-T04 acceptance test (design §3.3): adversarial mixed-magnitude fp32
// input, thread counts {1, 2, 8}, bit-identical results — and identical to
// the same fixed tree computed serially. Grains cover single-chunk, odd
// chunk counts (the odd-carry fold path), and power-of-two chunk counts.
TEST(ParallelReduceTest, BitIdenticalAcrossThreadCounts) {
  const std::int64_t n = 10'000;
  const std::vector<float> data = AdversarialData(n);
  for (const std::int64_t grain : {1L, 7L, 64L, 333L, 1024L, 4096L, 20'000L}) {
    const float reference = ReferenceTreeSum(data, grain);
    for (const int threads : {1, 2, 8}) {
      ThreadPool pool(threads);
      const float result = PooledSum(pool, data, grain);
      EXPECT_EQ(Bits(result), Bits(reference))
          << "grain=" << grain << " threads=" << threads;
    }
  }
}

TEST(ParallelReduceTest, BitIdenticalAcrossRepeatedRuns) {
  const std::vector<float> data = AdversarialData(10'000);
  ThreadPool pool(8);
  const float first = PooledSum(pool, data, 64);
  for (int run = 0; run < 20; ++run) {
    ASSERT_EQ(Bits(PooledSum(pool, data, 64)), Bits(first)) << "run " << run;
  }
}

TEST(ParallelNestingDeathTest, NestedParallelForOnInlinePathDies) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ThreadPool pool(2);
  EXPECT_DEATH(parallel_for(pool, 4, 100,
                            [&](std::int64_t, std::int64_t) {
                              parallel_for(pool, 4, 100,
                                           [](std::int64_t, std::int64_t) {});
                            }),
               "must not nest");
}

TEST(ParallelNestingDeathTest, NestedParallelForFromWorkerDies) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ThreadPool pool(2);
  EXPECT_DEATH(parallel_for(pool, 100, 10,
                            [&](std::int64_t, std::int64_t) {
                              parallel_for(pool, 100, 10,
                                           [](std::int64_t, std::int64_t) {});
                            }),
               "must not nest");
}

TEST(ParallelNestingDeathTest, NestedParallelReduceDies) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ThreadPool pool(2);
  EXPECT_DEATH(parallel_for(pool, 100, 10,
                            [&](std::int64_t, std::int64_t) {
                              (void)parallel_reduce<int>(
                                  pool, 100, 10, 0,
                                  [](std::int64_t, std::int64_t) { return 0; },
                                  [](int a, int b) { return a + b; });
                            }),
               "must not nest");
}

TEST(ParallelNestingDeathTest, DirectRunFromInsideRegionDies) {
  // ThreadPool::Run is a public seam; called from inside a region it would
  // deadlock on the pool's serialization. The CHECK converts that silent
  // hang into a loud failure (M3 audit).
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ThreadPool pool(2);
  EXPECT_DEATH(parallel_for(pool, 100, 10,
                            [&](std::int64_t, std::int64_t) {
                              pool.Run(1, [](std::int64_t) {});
                            }),
               "inside a parallel region");
}

TEST(ParallelBodyExceptionDeathTest, ThrowingBodyTerminatesOnInlinePath) {
  // Design §3.4: an escaping body exception terminates — on the inline path
  // too, not only when a worker thread unwinds. The catch below is reachable
  // only if the exception propagated out of parallel_for; exiting cleanly
  // there makes the death expectation fail (M3 audit).
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ThreadPool pool(2);
  EXPECT_DEATH(
      {
        try {
          parallel_for(pool, 4, 100, [](std::int64_t, std::int64_t) {
            throw std::runtime_error("body threw");
          });
        } catch (...) {
          std::_Exit(0);
        }
      },
      "");
}

TEST(DefaultPoolSizeTest, ParsesAndClampsEnvironmentOverride) {
  ::setenv("ENGINE_NUM_THREADS", "3", 1);
  EXPECT_EQ(detail::DefaultPoolSize(), 3);
  ::setenv("ENGINE_NUM_THREADS", "0", 1);
  EXPECT_EQ(detail::DefaultPoolSize(), 1);
  ::setenv("ENGINE_NUM_THREADS", "-5", 1);
  EXPECT_EQ(detail::DefaultPoolSize(), 1);
  ::setenv("ENGINE_NUM_THREADS", "", 1);  // empty counts as unset
  EXPECT_EQ(detail::DefaultPoolSize(), physical_core_count());
  ::unsetenv("ENGINE_NUM_THREADS");
  EXPECT_EQ(detail::DefaultPoolSize(), physical_core_count());
}

TEST(DefaultPoolSizeDeathTest, NonNumericValueDies) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        ::setenv("ENGINE_NUM_THREADS", "lots", 1);
        (void)detail::DefaultPoolSize();
      },
      "ENGINE_NUM_THREADS");
}

}  // namespace
}  // namespace engine::parallel
