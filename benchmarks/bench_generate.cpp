#include "core/status.h"
#include "engine/backend.h"
#include "engine/generator.h"
#include "kernels/dispatch.h"
#include "kvcache/simple_cache.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/registry.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#endif

// End-to-end generation benchmark (M6-T08; design: optimized-cpu-execution.md
// §10, "Benchmark obligations"). Measures prefill tokens/sec and decode
// tokens/sec for a given model, prompt length, and thread count, and prints a
// small report plus a run-to-run stability verdict (the ±5% acceptance
// criterion). Not registered with CTest for its perf numbers — those are run by
// hand on a quiesced machine and recorded in BASELINES.md (CLAUDE.md: no perf
// claim without a benchmark delta). A tiny smoke invocation IS registered
// (benchmarks/CMakeLists.txt) so CI exercises the harness and both backends.
//
// Timing seam: the real greedy `Generate` loop drives the model, and its
// per-token callback timestamps split the run into prefill (loop start → first
// token) and decode (per-step deltas between successive tokens). This measures
// exactly what a caller gets end-to-end (the final-position argmax included),
// and it is backend-agnostic — the reference and optimized backends run through
// the identical path.
//
// Prompt: synthetic random token ids in [0, vocab) of the requested length
// (like llama.cpp's `llama-bench -p N`). Model-agnostic, exact length, no
// tokenizer dependency — the token *values* do not affect the compute, only the
// sequence length does.
//
// Threads are process-wide (parallel::DefaultPool is sized once from
// ENGINE_NUM_THREADS on first use). `--threads N` sets that env var *before*
// the model loads (weight packing already touches the pool), so a thread sweep
// is `for t in 1 2 4 8; do bench_generate --threads $t ...; done`.
//
// Usage:
//   bench_generate --model DIR [--backend optimized|reference]
//                  [--threads N] [--prompt-len P] [--new-tokens N]
//                  [--runs R] [--seed S] [--markdown]

namespace {

using Clock = std::chrono::steady_clock;

using engine::core::Status;
using engine::engine::Backend;
using engine::engine::BackendName;
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::engine::ParseBackend;
using engine::kvcache::CacheGeometry;
using engine::kvcache::SimpleKvCache;
using engine::model::BuildModel;
using engine::model::BuildOptions;
using engine::model::load_model;
using engine::model::Model;

template <typename T>
[[nodiscard]] T Unwrap(engine::core::StatusOr<T> value) {
  if (!value.ok()) {
    fmt::print(stderr, "fatal: {}\n", value.status().ToString());
    std::exit(1);
  }
  return *std::move(value);
}

void Check(const Status& status) {
  if (!status.ok()) {
    fmt::print(stderr, "fatal: {}\n", status.ToString());
    std::exit(1);
  }
}

struct Args {
  std::string model_dir;
  Backend backend = Backend::kOptimized;
  std::int64_t prompt_len = 128;
  std::int64_t new_tokens = 128;
  int runs = 5;
  std::uint32_t seed = 1234;
  int threads = 0;  // 0 → leave ENGINE_NUM_THREADS untouched (env / default).
  bool markdown = false;
};

[[nodiscard]] Status ParseArgs(std::span<const std::string> argv, Args& out) {
  for (std::size_t i = 0; i < argv.size(); ++i) {
    const std::string& a = argv[i];
    const auto next = [&](std::string& dst) -> Status {
      if (i + 1 >= argv.size()) {
        return engine::core::InvalidArgumentError("missing value for {}", a);
      }
      dst = argv[++i];
      return engine::core::OkStatus();
    };
    if (a == "--model") {
      RETURN_IF_ERROR(next(out.model_dir));
    } else if (a == "--backend") {
      std::string b;
      RETURN_IF_ERROR(next(b));
      ASSIGN_OR_RETURN(out.backend, ParseBackend(b));
    } else if (a == "--prompt-len") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.prompt_len = std::stoll(n);
    } else if (a == "--new-tokens") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.new_tokens = std::stoll(n);
    } else if (a == "--runs") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.runs = std::stoi(n);
    } else if (a == "--seed") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.seed = static_cast<std::uint32_t>(std::stoul(n));
    } else if (a == "--threads") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.threads = std::stoi(n);
    } else if (a == "--markdown") {
      out.markdown = true;
    } else {
      return engine::core::InvalidArgumentError("unknown flag '{}'", a);
    }
  }
  if (out.model_dir.empty()) {
    return engine::core::InvalidArgumentError("--model DIR is required");
  }
  if (out.prompt_len < 1) {
    return engine::core::InvalidArgumentError("--prompt-len must be >= 1");
  }
  if (out.new_tokens < 1) {
    return engine::core::InvalidArgumentError("--new-tokens must be >= 1");
  }
  if (out.runs < 1) {
    return engine::core::InvalidArgumentError("--runs must be >= 1");
  }
  return engine::core::OkStatus();
}

// A one-line CPU model string for the fingerprint, best-effort per platform.
[[nodiscard]] std::string CpuBrand() {
#if defined(__APPLE__)
  char buf[256];
  std::size_t len = sizeof(buf);
  if (::sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
    return {buf, len > 0 ? len - 1 : 0};
  }
#elif defined(__linux__)
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    const auto colon = line.find(':');
    if (colon != std::string::npos && line.rfind("model name", 0) == 0) {
      std::string value = line.substr(colon + 1);
      const auto first = value.find_first_not_of(" \t");
      return first == std::string::npos ? value : value.substr(first);
    }
  }
#endif
  return "unknown";
}

// Percentile of a sorted (ascending) sample, nearest-rank.
[[nodiscard]] double Percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) {
    return 0.0;
  }
  const auto n = static_cast<double>(sorted.size());
  auto idx = static_cast<std::size_t>(std::ceil(p / 100.0 * n)) - 1;
  idx = std::min(idx, sorted.size() - 1);
  return sorted[idx];
}

[[nodiscard]] double Median(std::vector<double> xs) {
  std::ranges::sort(xs);
  return Percentile(xs, 50.0);
}

[[nodiscard]] double Mean(const std::vector<double>& xs) {
  if (xs.empty()) {
    return 0.0;
  }
  const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
  return sum / static_cast<double>(xs.size());
}

// Max |x - median| / median, as a percentage — the run-to-run stability
// metric the ±5% acceptance criterion is stated against.
[[nodiscard]] double MaxRelDeviationPct(const std::vector<double>& xs) {
  const double med = Median(xs);
  if (med == 0.0) {
    return 0.0;
  }
  double worst = 0.0;
  for (const double x : xs) {
    worst = std::max(worst, std::abs(x - med) / med);
  }
  return worst * 100.0;
}

struct RunResult {
  double prefill_tok_s = 0.0;  // prompt_len / prefill time (one sample per run)
  double decode_tok_s = 0.0;   // 1000 / median decode-step latency this run
};

// One generation pass over a fresh cache: returns prefill throughput and the
// steady-state decode rate, and appends every decode-step latency (ms) to
// `step_ms`.
//
// Decode uses the *median* step latency, not the whole-window average: on a
// non-quiesced machine a single background hiccup inflates one step out of
// hundreds (see the p90≪max gap in the report), which is not representative of
// steady-state decode. The median is outlier-robust and is what makes the
// per-run number repeatable within the ±5% target. Prefill is a single forward
// per run, so its headline is best-of-N across runs (matching the peak-of-N
// convention of the sibling kernel benches).
[[nodiscard]] RunResult DoRun(Model& model, SimpleKvCache& cache,
                              std::span<const std::int32_t> prompt_ids,
                              std::int64_t new_tokens,
                              std::vector<double>& step_ms) {
  cache.reset();
  // Empty eos set → exactly `new_tokens` decode steps, so the measured decode
  // window is fixed regardless of what the random logits argmax to.
  const GenerateOptions options{.max_new_tokens = new_tokens, .eos_ids = {}};

  std::vector<Clock::time_point> stamps;
  stamps.reserve(static_cast<std::size_t>(new_tokens));
  const auto on_token = [&](std::int32_t /*id*/) {
    stamps.push_back(Clock::now());
  };

  const Clock::time_point t0 = Clock::now();
  const std::vector<std::int32_t> generated =
      Unwrap(Generate(model, cache, prompt_ids, options, on_token));
  if (stamps.empty()) {
    fmt::print(stderr, "fatal: generation produced no tokens\n");
    std::exit(1);
  }

  const double prefill_s =
      std::chrono::duration<double>(stamps.front() - t0).count();
  RunResult r;
  r.prefill_tok_s = prefill_s > 0.0
                        ? static_cast<double>(prompt_ids.size()) / prefill_s
                        : 0.0;

  std::vector<double> run_steps;
  run_steps.reserve(stamps.size());
  for (std::size_t i = 1; i < stamps.size(); ++i) {
    const double ms =
        std::chrono::duration<double, std::milli>(stamps[i] - stamps[i - 1])
            .count();
    run_steps.push_back(ms);
    step_ms.push_back(ms);
  }
  const double median_step_ms = Median(run_steps);
  r.decode_tok_s = median_step_ms > 0.0 ? 1000.0 / median_step_ms : 0.0;
  (void)generated;
  return r;
}

[[nodiscard]] Status RunBench(const Args& args) {
  const Clock::time_point t_load0 = Clock::now();
  ASSIGN_OR_RETURN(auto loaded, load_model(args.model_dir));
  ASSIGN_OR_RETURN(
      std::unique_ptr<Model> model,
      BuildModel(std::move(loaded), BuildOptions{.backend = args.backend}));
  const Clock::time_point t_load1 = Clock::now();
  const double load_ms =
      std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

  const engine::model::ModelConfig& cfg = model->config();
  const std::int64_t total_positions = args.prompt_len + args.new_tokens;
  if (cfg.max_position_embeddings > 0 &&
      total_positions > cfg.max_position_embeddings) {
    return engine::core::InvalidArgumentError(
        "prompt-len + new-tokens = {} exceeds the model's "
        "max_position_embeddings = {}",
        total_positions, cfg.max_position_embeddings);
  }

  // Synthetic prompt: random ids in [0, vocab). Values are irrelevant to the
  // compute; only the length matters.
  std::mt19937 rng(args.seed);
  std::uniform_int_distribution<std::int32_t> dist(
      0, static_cast<std::int32_t>(cfg.vocab_size - 1));
  std::vector<std::int32_t> prompt_ids(
      static_cast<std::size_t>(args.prompt_len));
  for (std::int32_t& id : prompt_ids) {
    id = dist(rng);
  }

  const CacheGeometry geom = model->cache_geometry();
  ASSIGN_OR_RETURN(SimpleKvCache cache,
                   SimpleKvCache::Create(geom, total_positions));

  // Fingerprint.
  fmt::print("== bench_generate ==\n");
  fmt::print("host:     {} ({} physical cores)\n", CpuBrand(),
             engine::parallel::physical_core_count());
  fmt::print("threads:  {} (pool)\n",
             engine::parallel::DefaultPool().num_threads());
  fmt::print("isa:      {}\n",
             engine::kernels::IsaName(engine::kernels::SelectedIsa()));
  fmt::print("backend:  {}\n", BackendName(args.backend));
  fmt::print("model:    {}\n", args.model_dir);
  fmt::print("  arch={} layers={} hidden={} heads={} kv_heads={} head_dim={}\n",
             cfg.architecture_name, cfg.num_layers, cfg.hidden_size,
             cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
  fmt::print("  vocab={} dtype={}\n", cfg.vocab_size,
             engine::tensor::to_string(cfg.torch_dtype));
  fmt::print("load:     {:.0f} ms\n", load_ms);
  fmt::print("workload: prompt-len={} new-tokens={} runs={} seed={}\n\n",
             args.prompt_len, args.new_tokens, args.runs, args.seed);

  // One unrecorded warmup pass (first-touch faults, workspace grow-on-demand).
  std::vector<double> warmup_steps;
  (void)DoRun(*model, cache, prompt_ids, args.new_tokens, warmup_steps);

  std::vector<double> prefill_series;
  std::vector<double> decode_series;
  std::vector<double> all_step_ms;
  prefill_series.reserve(static_cast<std::size_t>(args.runs));
  decode_series.reserve(static_cast<std::size_t>(args.runs));

  fmt::print("{:>4}  {:>14}  {:>14}\n", "run", "prefill tok/s", "decode tok/s");
  for (int r = 0; r < args.runs; ++r) {
    const RunResult res =
        DoRun(*model, cache, prompt_ids, args.new_tokens, all_step_ms);
    prefill_series.push_back(res.prefill_tok_s);
    decode_series.push_back(res.decode_tok_s);
    fmt::print("{:>4}  {:>14.2f}  {:>14.2f}\n", r + 1, res.prefill_tok_s,
               res.decode_tok_s);
  }

  std::ranges::sort(all_step_ms);
  // Prefill headline is best-of-N (peak), like the sibling kernel benches;
  // decode headline is the median per-run steady-state rate. Stability is the
  // run-to-run spread of each series, and the verdict is stated against the
  // decode series — the repeatable steady-state metric the ±5% criterion
  // targets. Prefill's single-forward-per-run spread is reported for context.
  const double prefill_best = *std::ranges::max_element(prefill_series);
  const double prefill_stability = MaxRelDeviationPct(prefill_series);
  const double decode_median = Median(decode_series);
  const double decode_stability = MaxRelDeviationPct(decode_series);

  fmt::print("\n-- summary --\n");
  fmt::print(
      "prefill tok/s: best {:.2f}  median {:.2f}  min {:.2f}  max {:.2f}"
      "  (±{:.1f}% run-to-run)\n",
      prefill_best, Median(prefill_series),
      *std::ranges::min_element(prefill_series), prefill_best,
      prefill_stability);
  fmt::print(
      "decode  tok/s: median {:.2f}  mean {:.2f}  min {:.2f}  max {:.2f}"
      "  (±{:.1f}% run-to-run)\n",
      decode_median, Mean(decode_series),
      *std::ranges::min_element(decode_series),
      *std::ranges::max_element(decode_series), decode_stability);
  fmt::print("decode  step ms: p50 {:.3f}  p90 {:.3f}  max {:.3f}\n",
             Percentile(all_step_ms, 50.0), Percentile(all_step_ms, 90.0),
             all_step_ms.empty() ? 0.0 : all_step_ms.back());

  fmt::print("stability: decode ±{:.1f}% across {} runs — {} (±5% target)\n",
             decode_stability, args.runs,
             decode_stability <= 5.0 ? "PASS" : "OVER");

  if (args.markdown) {
    fmt::print(
        "\n| backend | threads | prompt-len | new-tokens | prefill tok/s"
        " (best) | decode tok/s (median) | decode ±% |\n");
    fmt::print("|---|---:|---:|---:|---:|---:|---:|\n");
    fmt::print("| {} | {} | {} | {} | {:.1f} | {:.1f} | ±{:.1f}% |\n",
               BackendName(args.backend),
               engine::parallel::DefaultPool().num_threads(), args.prompt_len,
               args.new_tokens, prefill_best, decode_median, decode_stability);
  }
  return engine::core::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> rest;
  rest.reserve(static_cast<std::size_t>(argc));
  for (int i = 1; i < argc; ++i) {
    rest.emplace_back(argv[i]);
  }
  Args args;
  const Status parsed = ParseArgs(rest, args);
  if (!parsed.ok()) {
    fmt::print(stderr, "error: {}\n", parsed.message());
    fmt::print(stderr,
               "usage: bench_generate --model DIR [--backend "
               "optimized|reference] [--threads N] [--prompt-len P] "
               "[--new-tokens N] [--runs R] [--seed S] [--markdown]\n");
    return 2;
  }

  // Pin the pool size before anything touches DefaultPool (weight packing in
  // load_model does). Overwrites any inherited ENGINE_NUM_THREADS so the
  // reported thread count matches the flag; without the flag, the inherited
  // env (or physical-core default) stands.
  if (args.threads > 0) {
    const std::string value = std::to_string(args.threads);
    ::setenv("ENGINE_NUM_THREADS", value.c_str(), /*overwrite=*/1);
  }

  Check(RunBench(args));
  return 0;
}
