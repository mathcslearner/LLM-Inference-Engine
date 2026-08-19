#include "core/status.h"
#include "parallel/thread_pool.h"
#include "sampling/batched_sampler.h"
#include "sampling/params.h"
#include "sampling/sampler.h"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Batched-sampling microbenchmark (M7-T06). Drives `BatchedSampler::Sample`
// over a synthetic `[batch, vocab]` logits block — the sampling hot path in
// isolation, no model load (like `llama-bench`'s prompt-length-only mode) — and
// reports median ms per step alongside the equivalent serial reference loop, so
// the speedup from `parallel_for` across sequences is visible. The acceptance
// target is advisory: batched sampling for 64 sequences × 128k vocab under
// 5 ms on the dev machine (recorded in benchmarks/BASELINES.md; CLAUDE.md: no
// perf claim without a benchmark delta). A tiny smoke run is registered with
// CTest (benchmarks/CMakeLists.txt) so CI exercises arg parsing and every
// config; real numbers are taken by hand on a quiesced machine.
//
// Usage: bench_sampling [--batch B] [--vocab V] [--runs N] [--threads T]
//                       [--config greedy|temp|topk_topp|topp|logprobs] [--seed
//                       S]
// `--threads T` sets ENGINE_NUM_THREADS before the pool is first sized.

namespace {

using engine::core::Status;
using engine::parallel::DefaultPool;
using engine::sampling::BatchedSampler;
using engine::sampling::BatchRow;
using engine::sampling::Sampler;
using engine::sampling::SampleResult;
using engine::sampling::SamplingParams;

struct Args {
  std::int64_t batch = 64;
  std::int64_t vocab = 128LL * 1024;
  int runs = 20;
  int threads = 0;  // 0 → leave ENGINE_NUM_THREADS untouched.
  std::string config = "temp";
  std::uint64_t seed = 1234;
};

[[nodiscard]] SamplingParams ConfigParams(std::string_view name,
                                          std::uint64_t seed) {
  SamplingParams p;
  p.seed = seed;
  p.max_tokens = 8;
  if (name == "greedy") {
    p = SamplingParams::Greedy(8);
  } else if (name == "temp") {
    p.temperature = 1.0F;
  } else if (name == "topk_topp") {
    p.temperature = 1.0F;
    p.top_k = 50;
    p.top_p = 0.9F;
  } else if (name == "topp") {
    p.temperature = 1.0F;
    p.top_p = 0.95F;
  } else if (name == "logprobs") {
    p.temperature = 1.0F;
    p.logprobs = 5;
  }
  return p;
}

// Deterministic pseudo-random logit (no RNG dependency, reproducible block).
[[nodiscard]] float LogitAt(std::int64_t row, std::int64_t col,
                            std::uint64_t seed) {
  std::uint64_t s = (static_cast<std::uint64_t>(row) * 0x9E3779B97F4A7C15ULL) ^
                    (static_cast<std::uint64_t>(col) * 0xD1B54A32D192ED03ULL) ^
                    seed;
  s ^= s >> 33U;
  s *= 0xFF51AFD7ED558CCDULL;
  s ^= s >> 33U;
  const auto unit =
      static_cast<float>(s >> 40U) / static_cast<float>(1U << 24U);
  return (unit * 16.0F) - 8.0F;  // ~[-8, 8)
}

[[nodiscard]] double MedianMs(std::vector<double>& samples) {
  std::ranges::sort(samples);
  const std::size_t n = samples.size();
  return n % 2 == 1 ? samples[n / 2]
                    : 0.5 * (samples[(n / 2) - 1] + samples[n / 2]);
}

int Run(const Args& args) {
  const auto batch = static_cast<std::size_t>(args.batch);
  const auto vocab = static_cast<std::size_t>(args.vocab);

  std::vector<float> block(batch * vocab);
  for (std::int64_t r = 0; r < args.batch; ++r) {
    for (std::int64_t c = 0; c < args.vocab; ++c) {
      block[(static_cast<std::size_t>(r) * vocab) +
            static_cast<std::size_t>(c)] = LogitAt(r, c, args.seed);
    }
  }

  // Per-row samplers (all the same config here; the batched path supports
  // heterogeneous params — this measures the common case).
  std::vector<Sampler> samplers;
  samplers.reserve(batch);
  std::vector<BatchRow> rows(batch);
  for (std::size_t r = 0; r < batch; ++r) {
    auto s = Sampler::Create(ConfigParams(args.config, args.seed + r));
    if (!s.ok()) {
      fmt::print("sampler create failed: {}\n", s.status().ToString());
      return 1;
    }
    samplers.push_back(*std::move(s));
    rows[r] = BatchRow{.sampler = &samplers[r], .context = {}};
  }

  BatchedSampler batched(DefaultPool());
  std::vector<SampleResult> out(batch);

  // Warmup (first call sizes the scratch and the pool).
  if (const Status st = batched.Sample(block, args.vocab, rows, out);
      !st.ok()) {
    fmt::print("batched sample failed: {}\n", st.ToString());
    return 1;
  }

  std::vector<double> batched_ms;
  batched_ms.reserve(static_cast<std::size_t>(args.runs));
  for (int i = 0; i < args.runs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)batched.Sample(block, args.vocab, rows, out);
    const auto t1 = std::chrono::steady_clock::now();
    batched_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // Serial reference loop for the speedup context number.
  std::vector<double> serial_ms;
  serial_ms.reserve(static_cast<std::size_t>(args.runs));
  for (int i = 0; i < args.runs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < batch; ++r) {
      const std::span<const float> row(block.data() + (r * vocab), vocab);
      (void)samplers[r].SampleWithLogprobs(row, rows[r].context);
    }
    const auto t1 = std::chrono::steady_clock::now();
    serial_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  const double bmed = MedianMs(batched_ms);
  const double smed = MedianMs(serial_ms);
  const int nthreads = DefaultPool().num_threads();
  fmt::print(
      "config={} batch={} vocab={} threads={}\n"
      "  batched  median {:.3f} ms/step\n"
      "  serial   median {:.3f} ms/step  (reference loop)\n"
      "  speedup  {:.2f}x   verdict {} (advisory <= 5 ms)\n",
      args.config, args.batch, args.vocab, nthreads, bmed, smed, smed / bmed,
      bmed <= 5.0 ? "PASS" : "OVER");
  return 0;
}

[[nodiscard]] bool ParseArgs(std::span<const std::string> argv, Args& out) {
  // Every recognized flag takes exactly one value, so consume it up front and
  // dispatch on assignment only (keeps the branch factor low).
  for (std::size_t i = 0; i < argv.size(); ++i) {
    const std::string& a = argv[i];
    if (i + 1 >= argv.size()) {
      fmt::print("missing value for {}\n", a);
      return false;
    }
    const std::string& v = argv[++i];
    if (a == "--batch") {
      out.batch = std::stoll(v);
    } else if (a == "--vocab") {
      out.vocab = std::stoll(v);
    } else if (a == "--runs") {
      out.runs = std::stoi(v);
    } else if (a == "--threads") {
      out.threads = std::stoi(v);
    } else if (a == "--config") {
      out.config = v;
    } else if (a == "--seed") {
      out.seed = std::stoull(v);
    } else {
      fmt::print("unknown flag '{}'\n", a);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  Args parsed;
  if (!ParseArgs(args, parsed)) {
    return 2;
  }
  // Set the thread count before DefaultPool() is first sized.
  if (parsed.threads > 0) {
    ::setenv("ENGINE_NUM_THREADS", std::to_string(parsed.threads).c_str(),
             /*overwrite=*/1);
  }
  return Run(parsed);
}
