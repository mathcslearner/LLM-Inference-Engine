#include "runtime/engine.h"

#include "core/check.h"
#include "core/status.h"
#include "engine/stop.h"
#include "kvcache/block_pool.h"
#include "model/model.h"
#include "runtime/channel.h"
#include "runtime/request.h"
#include "sampling/logprobs.h"
#include "sampling/sampler.h"
#include "scheduler/scheduler.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// Engine loop & submission queue (M9-T03; design:
// docs/design/scheduler-runtime.md §5, §9.1). The single engine thread owns all
// mutable state (the request map, the block pool, every sequence and its cache,
// the samplers); client threads only enqueue commands and drain their own
// channel. `Step()` is the §9.1 loop with the T03 placeholders (FCFS admission,
// per-sequence execution) the later tickets substitute in place.

namespace engine::runtime {

// The runtime and the (core-only) scheduler each declare their own `RequestId`
// alias rather than share a header across the layer boundary (§2); they must
// name the same integer type so ids pass through the scheduler unchanged.
static_assert(std::is_same_v<RequestId, scheduler::RequestId>,
              "runtime and scheduler RequestId must be the same type");

namespace {

// The `[·, V]` logits from `forward` as a flat `[V]` span over its final row —
// the sampler input. `forward` returns a contiguous caller-owned buffer, so the
// last `V` elements are the final position's logits. (Same helper as
// `engine::Generate`; kLast forwards return a single row.)
[[nodiscard]] std::span<const float> LastRow(const tensor::Tensor& logits) {
  const int rank = logits.shape().rank();
  const std::int64_t vocab = logits.shape().dim(rank - 1);
  const std::int64_t rows = logits.numel() / vocab;
  const float* base = logits.data_ptr<float>();
  const auto offset = static_cast<std::size_t>((rows - 1) * vocab);
  return std::span<const float>{base + offset, static_cast<std::size_t>(vocab)};
}

}  // namespace

Engine::Engine(model::Model& model, kvcache::BlockPool& pool,
               EngineConfig config, std::int64_t max_model_len)
    : model_(model),
      pool_(pool),
      config_(config),
      max_model_len_(max_model_len) {}

core::StatusOr<std::unique_ptr<Engine>> Engine::Create(model::Model& model,
                                                       kvcache::BlockPool& pool,
                                                       EngineConfig config) {
  if (config.max_num_seqs <= 0) {
    return core::InvalidArgumentError("max_num_seqs must be > 0 (got {})",
                                      config.max_num_seqs);
  }
  if (config.max_num_batched_tokens <= 0) {
    return core::InvalidArgumentError(
        "max_num_batched_tokens must be > 0 (got {})",
        config.max_num_batched_tokens);
  }
  if (config.max_model_len < 0) {
    return core::InvalidArgumentError("max_model_len must be >= 0 (got {})",
                                      config.max_model_len);
  }
  // The sequences' caches draw from `pool`, and `model.forward` validates each
  // append against its own geometry; a mismatch would fail every forward, so
  // reject it up front with a clear message.
  const kvcache::CacheGeometry& pg = pool.geometry();
  const kvcache::CacheGeometry mg = model.cache_geometry();
  if (pg.num_layers != mg.num_layers || pg.num_kv_heads != mg.num_kv_heads ||
      pg.head_dim != mg.head_dim || pg.dtype != mg.dtype) {
    return core::InvalidArgumentError(
        "block pool geometry (L={}, Hkv={}, d={}) does not match model "
        "(L={}, Hkv={}, d={})",
        pg.num_layers, pg.num_kv_heads, pg.head_dim, mg.num_layers,
        mg.num_kv_heads, mg.head_dim);
  }
  // Resolve the length ceiling: explicit config, else the model's positional
  // limit. (`Submit` also bounds by the pool's token capacity.)
  std::int64_t max_model_len = config.max_model_len;
  if (max_model_len == 0) {
    max_model_len = model.config().max_position_embeddings;
  }
  // `Engine` is deliberately non-movable (it owns a thread, a mutex, and a map
  // of sequences that borrow into it), so the movable-temporary factory idiom
  // `make_unique<T>(T(...))` cannot apply — the private constructor is reached
  // through `new` inside a `unique_ptr`. The analyzer mis-models the ownership
  // transfer through the returned `unique_ptr` as a leak (it is not — the
  // pointer is owned and moved out).
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return std::unique_ptr<Engine>(
      new Engine(model, pool, config, max_model_len));
}

Engine::~Engine() { Stop(); }

// --- client-thread API -----------------------------------------------------

core::StatusOr<RequestHandle> Engine::Submit(Request request) {
  if (request.prompt_ids.empty()) {
    return core::InvalidArgumentError("prompt_ids must be non-empty");
  }
  const auto prompt_len = static_cast<std::int64_t>(request.prompt_ids.size());
  if (prompt_len > config_.max_num_batched_tokens) {
    return core::InvalidArgumentError(
        "prompt length {} exceeds max_num_batched_tokens {} (chunked prefill "
        "is not implemented until M12-T06)",
        prompt_len, config_.max_num_batched_tokens);
  }
  // Worst-case peak length (no early stop): prompt + one appended token per
  // decode step; the final produced token is never appended, so the peak is
  // `prompt_len + max_tokens - 1` (matching `Generate`). Bound it by both the
  // model's length ceiling and the pool's token capacity — a request that can
  // never fit even an empty pool is rejected here, cache untouched.
  const std::int64_t max_tokens = request.params.max_tokens;
  const std::int64_t peak = prompt_len + (max_tokens > 0 ? max_tokens - 1 : 0);
  if (max_model_len_ > 0 && peak > max_model_len_) {
    return core::ResourceExhaustedError(
        "prompt {} + {} new tokens exceeds max_model_len {} (peak {})",
        prompt_len, max_tokens, max_model_len_, peak);
  }
  const std::int64_t pool_capacity = pool_.num_blocks() * pool_.block_size();
  if (peak > pool_capacity) {
    return core::ResourceExhaustedError(
        "prompt {} + {} new tokens exceeds pool token capacity {} (peak {})",
        prompt_len, max_tokens, pool_capacity, peak);
  }

  // All fallible per-sequence work (sampler/stop validation) happens before the
  // id is assigned, so a rejected submit consumes no id and enqueues nothing.
  auto entry = std::make_unique<Entry>(std::move(request));
  auto seq = Sequence::Create(entry->request, &pool_);
  if (!seq.ok()) {
    return seq.status();
  }
  entry->seq = std::make_unique<Sequence>(*std::move(seq));
  auto channel = entry->seq->channel();

  RequestId id = 0;
  {
    const std::lock_guard<std::mutex> lock(cmd_mu_);
    if (stop_requested_) {
      return core::FailedPreconditionError("engine is stopped");
    }
    id = next_id_++;
    entry->request.id = id;
    entry->request.arrival_index = next_arrival_++;
    commands_.emplace_back(SubmitCmd{.entry = std::move(entry)});
  }
  cmd_cv_.notify_one();
  return RequestHandle(id, std::move(channel));
}

void Engine::Cancel(RequestId id) {
  {
    const std::lock_guard<std::mutex> lock(cmd_mu_);
    commands_.emplace_back(CancelCmd{.id = id});
  }
  cmd_cv_.notify_one();
}

// --- loop control ----------------------------------------------------------

void Engine::Start() {
  CHECK(!thread_.joinable(), "Engine::Start called on a running engine");
  thread_ = std::thread([this] { Run(); });
}

void Engine::Stop() {
  {
    const std::lock_guard<std::mutex> lock(cmd_mu_);
    stop_requested_ = true;
  }
  cmd_cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  // The engine thread is gone (or was never started); this thread is now the
  // sole owner of the engine state. Close every live channel and drop every
  // sequence (RAII frees blocks).
  Shutdown();
}

void Engine::Run() {
  for (;;) {
    {
      const std::lock_guard<std::mutex> lock(cmd_mu_);
      if (stop_requested_) {
        return;
      }
    }
    if (Step()) {
      continue;  // there may be more work — keep stepping without waiting.
    }
    // Idle: nothing waiting or running. Block until a command arrives or stop.
    std::unique_lock<std::mutex> lock(cmd_mu_);
    cmd_cv_.wait(lock,
                 [this] { return !commands_.empty() || stop_requested_; });
    if (stop_requested_) {
      return;
    }
  }
}

bool Engine::Step() {
  DrainCommands();    // 1. absorb submits/cancels at the step boundary
  RetireCancelled();  // 2. drop cancelled sequences (any state), free blocks
  if (waiting_.empty() && running_.empty()) {
    return false;  // idle
  }

  // 3. Schedule: the scheduler names which sequences to prefill (admit), decode
  //    (advance one token), and preempt (evict this step). It reads the current
  //    waiting_/running_ and one pool snapshot; it never mutates engine state.
  const scheduler::SchedulerOutput out = ScheduleStep();

  // 4. Apply preemptions before any forward, so their blocks are free this step
  //    for the decode/prefill that needed them (design §9.1 step 4 / §10.1).
  //    Evict-and-recompute: drop the cache (RAII frees every block), keep the
  //    generated ids / sampler / stop state, and requeue at the head of
  //    waiting_ ahead of never-started requests — resume re-prefills
  //    prompt ++ generated (§10.2).
  for (const RequestId id : out.preempt) {
    Entry& entry = *requests_.at(id);
    entry.seq->ReleaseCache();
    entry.seq->Transition(SeqState::kPreempted);
    std::erase(running_, id);
    waiting_.push_front(id);
  }

  // 5. Prefill pass over the admitted sequences (design §9.1 step 5). A newly
  //    admitted sequence samples its first token from this forward and joins
  //    the decode batch next step; a resumed (preempted) sequence re-prefills
  //    its whole progress and samples the next token, bit-identical to an
  //    uninterrupted run (§10.2, the KV invariant).
  for (const auto& p : out.prefill) {
    Entry& entry = *requests_.at(p.id);
    std::erase(waiting_, p.id);
    entry.seq->EnsureCache(&pool_);
    entry.seq->Transition(SeqState::kRunning);
    running_.push_back(p.id);
    ExecuteAndDeliver(entry, /*prefill=*/true);
  }

  // 6. Decode pass over the scheduler's decode set. Skip any that finished
  //    during their own prefill earlier this step (a sequence is never both
  //    admitted and decoded in one step — prefill produces its first token,
  //    decode advances the already-running set, design §7).
  for (const RequestId id : out.decode) {
    Entry& entry = *requests_.at(id);
    if (entry.seq->state() == SeqState::kRunning) {
      ExecuteAndDeliver(entry, /*prefill=*/false);
    }
  }

  RetireTerminal();  // 7. close + erase finished / failed sequences
  return true;
}

// --- engine-thread internals -----------------------------------------------

void Engine::DrainCommands() {
  std::vector<std::variant<SubmitCmd, CancelCmd>> commands;
  {
    const std::lock_guard<std::mutex> lock(cmd_mu_);
    commands.swap(commands_);
  }
  for (auto& cmd : commands) {
    if (std::holds_alternative<SubmitCmd>(cmd)) {
      auto entry = std::move(std::get<SubmitCmd>(cmd).entry);
      const RequestId id = entry->request.id;
      waiting_.push_back(id);
      requests_.emplace(id, std::move(entry));
    } else {
      const RequestId id = std::get<CancelCmd>(cmd).id;
      auto it = requests_.find(id);
      if (it != requests_.end()) {
        it->second->cancel_requested = true;
      }
      // An unknown id (never submitted, or already retired) is a no-op.
    }
  }
}

void Engine::RetireCancelled() {
  // T03 runs execution per sequence, so there is no in-flight batch at a step
  // boundary: a cancel requested for a sequence in *any* live state is honored
  // now (WAITING/PREEMPTED had no forward this step; a RUNNING sequence's last
  // forward already completed on the previous step). The batched loop (M9-T08)
  // moves RUNNING-cancel handling to the end of the step, where a cancel that
  // lands during the forward is applied at the next boundary (§11.1).
  std::vector<RequestId> retired;
  for (auto& [id, entry] : requests_) {
    if (entry->cancel_requested && !entry->seq->is_terminal()) {
      CloseCancelled(*entry);
      retired.push_back(id);
    }
  }
  for (const RequestId id : retired) {
    std::erase(waiting_, id);
    std::erase(running_, id);
    requests_.erase(id);  // RAII: the sequence's cache returns its blocks
  }
}

scheduler::SchedulerOutput Engine::ScheduleStep() {
  // Fill the scheduler's plain descriptors (design §6.1) from the engine-thread
  // state and one pool snapshot. The scheduler sees no Sequence, cache, or pool
  // type — only these structs — so the whole policy is table-testable (§2).
  std::vector<scheduler::SeqDesc> waiting_descs;
  waiting_descs.reserve(waiting_.size());
  for (const RequestId id : waiting_) {
    const Sequence& seq = *requests_.at(id)->seq;
    const auto prompt_len =
        static_cast<std::int64_t>(seq.request().prompt_ids.size());
    // Admission demand is this step's prefill length: the prompt for a fresh
    // WAITING sequence, or prompt ++ generated for a PREEMPTED one being
    // resumed (§10.2). `num_generated()` is 0 for a fresh sequence, so this is
    // uniform.
    waiting_descs.push_back(
        {.id = id,
         .arrival_index = seq.request().arrival_index,
         .num_computed_tokens = 0,
         .num_prompt_tokens = prompt_len + seq.num_generated(),
         .blocks_held = 0});
  }

  std::vector<scheduler::SeqDesc> running_descs;
  running_descs.reserve(running_.size());
  for (const RequestId id : running_) {
    Sequence& seq = *requests_.at(id)->seq;
    const kvcache::PagedKvCache* cache = seq.cache();
    // A RUNNING sequence always holds a cache (ensured at admission). Read the
    // committed length and blocks held straight from it — the ground truth the
    // decode-demand and preemption arithmetic key on.
    running_descs.push_back({.id = id,
                             .arrival_index = seq.request().arrival_index,
                             .num_computed_tokens = cache->length(),
                             .num_prompt_tokens = 0,
                             .blocks_held = cache->block_table().num_blocks()});
  }

  const scheduler::ScheduleInputs inputs{
      .waiting = waiting_descs,
      .running = running_descs,
      .pool = {.free_blocks = pool_.free_blocks(),
               .total_blocks = pool_.num_blocks(),
               .block_size = pool_.block_size()},
      .config = {.max_num_seqs = config_.max_num_seqs,
                 .max_num_batched_tokens = config_.max_num_batched_tokens},
  };
  return scheduler::Scheduler{}.Schedule(inputs);
}

void Engine::ExecuteAndDeliver(Entry& entry, bool prefill) {
  Sequence& seq = *entry.seq;
  kvcache::KvCache& cache = *seq.cache();

  // Build this step's forward inputs. Prefill runs the whole prompt at absolute
  // positions `0..T-1`; decode runs the single last-produced token at the
  // running cache position (`== num_computed_tokens`). Both request kLast — one
  // logits row to sample from. This is exactly what `Generate` builds.
  std::vector<std::int32_t> positions;
  std::vector<std::int32_t> prefill_tokens;
  std::span<const std::int32_t> token_ids;
  std::int32_t decode_token = 0;
  if (prefill) {
    // Prefill covers the whole context to re-materialize: the prompt for a
    // fresh admission, or prompt ++ generated when resuming a preempted
    // sequence into a fresh cache (design §10.2). The re-prefilled prefix
    // reproduces the same K/V bit-for-bit (the KV invariant), so the next
    // sampled token is identical to an uninterrupted run. A fresh sequence has
    // empty `generated_ids()`, so this reduces to prefilling the prompt.
    const auto& prompt = seq.request().prompt_ids;
    const auto& generated = seq.generated_ids();
    prefill_tokens.reserve(prompt.size() + generated.size());
    prefill_tokens.insert(prefill_tokens.end(), prompt.begin(), prompt.end());
    prefill_tokens.insert(prefill_tokens.end(), generated.begin(),
                          generated.end());
    positions.resize(prefill_tokens.size());
    for (std::size_t t = 0; t < prefill_tokens.size(); ++t) {
      positions[t] = static_cast<std::int32_t>(t);
    }
    token_ids = std::span<const std::int32_t>{prefill_tokens};
  } else {
    decode_token = seq.generated_ids().back();
    positions.assign(1, static_cast<std::int32_t>(cache.length()));
    token_ids = std::span<const std::int32_t>{&decode_token, 1};
  }

  const model::ForwardRequest req{
      .token_ids = token_ids,
      .positions = positions,
      .cache = &cache,
      .logits_mode = model::LogitsMode::kLast,
  };
  auto logits = model_.forward(req);
  if (!logits.ok()) {
    // A per-request forward fault (bad id, position overflow, or this
    // sequence's own append exhausting the pool) fails only this sequence, not
    // the loop (ADR-003; §11.2). Graceful preemption of a decode exhaustion is
    // M9-T09; in T03 it surfaces as a per-request failure.
    seq.set_error(logits.status());
    seq.Transition(SeqState::kFailed);
    return;
  }

  // Sample before appending: the sampler's penalty history keys on the tokens
  // generated *so far* (this token excluded), exactly as `Generate` orders it.
  auto next = seq.sampler().SampleWithLogprobs(
      LastRow(*logits),
      sampling::SampleContext{.prompt_ids = seq.request().prompt_ids,
                              .generated_ids = seq.generated_ids()});
  if (!next.ok()) {
    seq.set_error(next.status());
    seq.Transition(SeqState::kFailed);
    return;
  }
  seq.AppendGenerated(next->token);

  auto step = seq.stop().Observe(next->token, seq.num_generated());
  if (!step.ok()) {
    seq.set_error(step.status());
    seq.Transition(SeqState::kFailed);
    return;
  }

  std::string delta = std::move(step->text_delta);
  const bool finished = step->finished;
  if (finished) {
    delta += seq.stop().Finish();  // release held/residual text with this token
    seq.set_finish(step->finish_reason, step->trigger,
                   std::move(step->matched_stop));
  }

  OutputItem item{.token_id = next->token,
                  .text_delta = std::move(delta),
                  .logprobs = std::move(next->logprobs)};
  seq.channel()->Push(std::move(item));

  if (finished) {
    seq.Transition(SeqState::kFinished);  // retired in step 7
  }
}

void Engine::RetireTerminal() {
  std::vector<RequestId> retired;
  for (const RequestId id : running_) {
    Entry& entry = *requests_.at(id);
    if (entry.seq->is_terminal()) {
      entry.seq->channel()->Close(entry.seq->MakeFinishInfo());
      retired.push_back(id);
    }
  }
  for (const RequestId id : retired) {
    std::erase(running_, id);
    requests_.erase(id);  // RAII frees the sequence's blocks
  }
}

void Engine::CloseCancelled(Entry& entry) {
  entry.seq->Transition(SeqState::kCancelled);
  FinishInfo info;
  info.terminal = SeqState::kCancelled;
  entry.seq->channel()->Close(std::move(info));
}

void Engine::Shutdown() {
  // Drain any commands the engine thread never processed (submits that arrived
  // after the stop flag, or between the last step and the join): close their
  // channels so no handle blocks forever.
  std::vector<std::variant<SubmitCmd, CancelCmd>> commands;
  {
    const std::lock_guard<std::mutex> lock(cmd_mu_);
    commands.swap(commands_);
  }
  for (auto& cmd : commands) {
    if (std::holds_alternative<SubmitCmd>(cmd)) {
      auto entry = std::move(std::get<SubmitCmd>(cmd).entry);
      // Never admitted: still WAITING with an empty cache. Cancel + close.
      CloseCancelled(*entry);
    }
  }
  // Close every live (non-terminal) sequence and drop it (frees blocks).
  for (auto& [id, entry] : requests_) {
    if (!entry->seq->is_terminal()) {
      CloseCancelled(*entry);
    }
  }
  requests_.clear();
  waiting_.clear();
  running_.clear();
}

// --- RequestHandle ---------------------------------------------------------

std::optional<OutputItem> RequestHandle::next_token() {
  CHECK(channel_ != nullptr, "next_token on an invalid RequestHandle");
  return channel_->Next();
}

std::optional<OutputItem> RequestHandle::try_next_token() {
  CHECK(channel_ != nullptr, "try_next_token on an invalid RequestHandle");
  return channel_->TryNext();
}

FinishInfo RequestHandle::await_completion() {
  CHECK(channel_ != nullptr, "await_completion on an invalid RequestHandle");
  return channel_->AwaitFinish();
}

}  // namespace engine::runtime
