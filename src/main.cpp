#include "core/status.h"
#include "engine/backend.h"
#include "engine/generator.h"
#include "kvcache/kv_cache.h"
#include "kvcache/simple_cache.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/registry.h"
#include "sampling/logprobs.h"
#include "sampling/params.h"
#include "tensor/dtype.h"
#include "tokenizer/tokenizer.h"

#include <fmt/base.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The engine entry point. `engine` prints the version; `engine generate`
// loads a model and greedily generates text — the M6-T07 real-model driver
// (design: docs/design/optimized-cpu-execution.md §10 acceptance). M9 replaces
// this with the real server binary + config system (ROADMAP M10-T08); until
// then this is the hand-run harness for loading a ~1B checkpoint and printing
// coherent output on the optimized backend.

namespace {

using engine::core::Status;
using engine::engine::Backend;
using engine::engine::Generate;
using engine::engine::GenerateOptions;
using engine::engine::ParseBackend;
using engine::kvcache::CacheGeometry;
using engine::kvcache::SimpleKvCache;
using engine::model::BuildModel;
using engine::model::BuildOptions;
using engine::model::load_model;
using engine::model::Model;
using engine::sampling::SamplingParams;
using engine::tokenizer::Tokenizer;

struct Args {
  std::string model_dir;
  std::string prompt = "Hello";
  Backend backend = Backend::kOptimized;
  std::int64_t max_new_tokens = 64;
  std::int64_t cache_capacity = 0;  // 0 → prompt + max_new_tokens
  bool add_bos = true;
  // Sampling controls (0 temperature ⇒ greedy, the default).
  float temperature = 0.0F;
  std::int32_t top_k = 0;
  float top_p = 1.0F;
  float repetition_penalty = 1.0F;
  float presence_penalty = 0.0F;
  float frequency_penalty = 0.0F;
  std::optional<std::uint64_t> seed;
  std::vector<std::string> stop_strings;
  std::vector<std::int32_t> stop_token_ids;
  std::int32_t logprobs = 0;
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
    } else if (a == "--prompt") {
      RETURN_IF_ERROR(next(out.prompt));
    } else if (a == "--backend") {
      std::string b;
      RETURN_IF_ERROR(next(b));
      ASSIGN_OR_RETURN(out.backend, ParseBackend(b));
    } else if (a == "--max-new-tokens") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.max_new_tokens = std::stoll(n);
    } else if (a == "--cache-capacity") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.cache_capacity = std::stoll(n);
    } else if (a == "--no-bos") {
      out.add_bos = false;
    } else if (a == "--temperature") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.temperature = std::stof(n);
    } else if (a == "--top-k") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.top_k = static_cast<std::int32_t>(std::stol(n));
    } else if (a == "--top-p") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.top_p = std::stof(n);
    } else if (a == "--repetition-penalty") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.repetition_penalty = std::stof(n);
    } else if (a == "--presence-penalty") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.presence_penalty = std::stof(n);
    } else if (a == "--frequency-penalty") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.frequency_penalty = std::stof(n);
    } else if (a == "--seed") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.seed = static_cast<std::uint64_t>(std::stoull(n));
    } else if (a == "--stop") {
      std::string s;
      RETURN_IF_ERROR(next(s));
      out.stop_strings.push_back(s);
    } else if (a == "--stop-token-id") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.stop_token_ids.push_back(static_cast<std::int32_t>(std::stol(n)));
    } else if (a == "--logprobs") {
      std::string n;
      RETURN_IF_ERROR(next(n));
      out.logprobs = static_cast<std::int32_t>(std::stol(n));
    } else {
      return engine::core::InvalidArgumentError("unknown flag '{}'", a);
    }
  }
  if (out.model_dir.empty()) {
    return engine::core::InvalidArgumentError("--model DIR is required");
  }
  return engine::core::OkStatus();
}

[[nodiscard]] Status RunGenerate(const Args& args) {
  const auto t_load0 = std::chrono::steady_clock::now();
  ASSIGN_OR_RETURN(auto loaded, load_model(args.model_dir));
  ASSIGN_OR_RETURN(
      std::unique_ptr<Model> model,
      BuildModel(std::move(loaded), BuildOptions{.backend = args.backend}));
  const auto t_load1 = std::chrono::steady_clock::now();

  ASSIGN_OR_RETURN(const Tokenizer tokenizer,
                   Tokenizer::from_file(std::filesystem::path(args.model_dir) /
                                        "tokenizer.json"));
  ASSIGN_OR_RETURN(std::vector<std::int32_t> prompt_ids,
                   tokenizer.encode(args.prompt, args.add_bos));
  if (prompt_ids.empty()) {
    return engine::core::InvalidArgumentError("prompt encoded to zero tokens");
  }

  const std::int64_t capacity =
      args.cache_capacity > 0
          ? args.cache_capacity
          : static_cast<std::int64_t>(prompt_ids.size()) + args.max_new_tokens;
  const CacheGeometry geom = model->cache_geometry();
  ASSIGN_OR_RETURN(SimpleKvCache cache, SimpleKvCache::Create(geom, capacity));

  SamplingParams sampling;
  sampling.max_tokens = args.max_new_tokens;
  sampling.temperature = args.temperature;
  sampling.top_k = args.top_k;
  sampling.top_p = args.top_p;
  sampling.repetition_penalty = args.repetition_penalty;
  sampling.presence_penalty = args.presence_penalty;
  sampling.frequency_penalty = args.frequency_penalty;
  sampling.seed = args.seed;
  sampling.stop_strings = args.stop_strings;
  sampling.stop_token_ids = args.stop_token_ids;
  sampling.logprobs = args.logprobs;
  const GenerateOptions options{.sampling = sampling,
                                .eos_ids = model->config().eos_token_ids,
                                .tokenizer = &tokenizer,
                                .skip_special_tokens = true};

  fmt::print("model:   {}\n", args.model_dir);
  fmt::print("backend: {}\n", engine::engine::BackendName(args.backend));
  fmt::print(
      "load:    {} ms\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(t_load1 - t_load0)
          .count());
  fmt::print("prompt:  {} ({} tokens)\n", args.prompt, prompt_ids.size());
  fmt::print("----\n{}", args.prompt);
  std::fflush(stdout);

  std::int64_t emitted = 0;
  const auto t_gen0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t_first_token;
  const auto on_token = [&](const engine::engine::TokenEvent& ev) {
    if (emitted == 0) {
      t_first_token = std::chrono::steady_clock::now();
    }
    ++emitted;
    if (!ev.text.empty()) {
      fmt::print("{}", ev.text);
      std::fflush(stdout);
    }
  };

  ASSIGN_OR_RETURN(const engine::engine::GenerateResult result,
                   Generate(*model, cache, prompt_ids, options, on_token));
  const std::vector<std::int32_t>& generated = result.tokens;
  const auto t_gen1 = std::chrono::steady_clock::now();

  const double prefill_ms =
      static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                              t_first_token - t_gen0)
                              .count()) /
      1000.0;
  const double decode_ms =
      static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                              t_gen1 - t_first_token)
                              .count()) /
      1000.0;
  const auto decode_steps =
      static_cast<double>(generated.empty() ? 0 : generated.size() - 1);
  fmt::print("\n----\n");
  fmt::print("generated {} tokens (finish_reason: {})\n", generated.size(),
             engine::engine::FinishReasonName(result.finish_reason));
  fmt::print("prefill: {:.1f} ms ({} prompt tokens, {:.1f} tok/s)\n",
             prefill_ms, prompt_ids.size(),
             prefill_ms > 0 ? static_cast<double>(prompt_ids.size()) /
                                  (prefill_ms / 1000.0)
                            : 0.0);
  if (decode_steps > 0 && decode_ms > 0) {
    fmt::print("decode:  {:.1f} ms ({:.1f} tok/s)\n", decode_ms,
               decode_steps / (decode_ms / 1000.0));
  }

  // Per-step logprobs (M7-T05), when requested: the chosen token's logprob and
  // the top-N alternatives, decoded for readability (single-id decode is
  // best-effort — a byte-level BPE fragment may render as U+FFFD).
  if (args.logprobs > 0 && !result.logprobs.empty()) {
    const auto render = [&](std::int32_t id) -> std::string {
      auto text = tokenizer.decode(std::span<const std::int32_t>(&id, 1),
                                   /*skip_special_tokens=*/false);
      return text.ok() ? *text : std::string{};
    };
    fmt::print("----\nlogprobs (top {}):\n", args.logprobs);
    for (std::size_t i = 0; i < result.logprobs.size(); ++i) {
      const engine::sampling::StepLogprobs& lp = result.logprobs[i];
      fmt::print("  [{}] id={} {:.4f} '{}'\n", i, generated[i],
                 lp.chosen_logprob, render(generated[i]));
      for (const engine::sampling::TokenLogprob& t : lp.top) {
        fmt::print("        id={:<6} {:.4f} '{}'\n", t.id, t.logprob,
                   render(t.id));
      }
    }
  }
  return engine::core::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string_view(argv[1]) == "generate") {
    std::vector<std::string> rest;
    rest.reserve(static_cast<std::size_t>(argc));
    for (int i = 2; i < argc; ++i) {
      rest.emplace_back(argv[i]);
    }
    Args args;
    const Status parsed = ParseArgs(rest, args);
    if (!parsed.ok()) {
      fmt::print(stderr, "error: {}\n", parsed.message());
      fmt::print(stderr,
                 "usage: engine generate --model DIR [--prompt STR] "
                 "[--backend reference|optimized] [--max-new-tokens N] "
                 "[--cache-capacity N] [--no-bos] [--temperature T] "
                 "[--top-k K] [--top-p P] [--repetition-penalty R] "
                 "[--presence-penalty P] [--frequency-penalty F] [--seed S] "
                 "[--stop STR]... [--stop-token-id N]... [--logprobs N]\n");
      return 2;
    }
    const Status status = RunGenerate(args);
    if (!status.ok()) {
      fmt::print(stderr, "error: {}\n", status.ToString());
      return 1;
    }
    return 0;
  }

  fmt::print("llm-inference-engine {}\n", ENGINE_VERSION);
  return 0;
}
