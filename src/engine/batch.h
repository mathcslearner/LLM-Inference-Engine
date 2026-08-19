#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "memory/allocator.h"
#include "model/model.h"
#include "sampling/batched_sampler.h"
#include "sampling/sampler.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Batch assembly (M9-T05; design: docs/design/scheduler-runtime.md §8.1/§8.2).
// Flattens the per-step scheduled work into the staged inputs a batched forward
// (M9-T07) consumes: concatenated token ids, per-token absolute positions,
// the `[B+1]` `cu_seqlens` prefix sums, the per-sequence caches, the
// per-request sampling metadata, and — for a decode step — the `[B,
// max_blocks]` block-table tensor and per-sequence lengths.
//
// One `BatchAssembler` is owned by the engine loop and reused across steps: the
// staging vectors keep their capacity and the block-table storage grows to a
// high-water mark, so a steady-state step allocates nothing (the same
// grow-on-demand discipline as `Workspace`/`BatchedSampler`; this closes
// optimized-cpu-execution.md §6.3's "pre-size staging from the batch-token
// budget" note — the staging *is* sized by the batch budget).
//
// Layering (ADR-002). `engine` is a layer-3 module; batch assembly touches
// tensors and the `model::ForwardRequest` struct, so it belongs here beside
// `Generate`, **not** in the tensor-free `scheduler` (which stays a `core`-only
// decision leaf, ADR-002 rule 4 — it never sees a `Sequence`, cache, or
// tensor). The assembler consumes plain `BatchSeqInput` descriptors the runtime
// fills from a `SchedulerOutput` + its `Sequence`s, so no `runtime`/`scheduler`
// type crosses into `engine` (the §8.2 shorthand
// `AssembleBatch(SchedulerOutput, sequences)` is realized as this
// descriptor-struct input). It links only already-present lower layers —
// `model`, `kvcache`, `sampling`, `tensor`, `memory` — so there is no new ADR
// edge.

namespace engine::engine {

// One scheduled sequence's contribution to a batch (design §8.2). Filled by the
// runtime per sequence each step; the assembler copies the token ids into
// staging and borrows the rest, so the referents (the sequence's cache,
// sampler, prompt/generated vectors) must outlive the forward + sample that
// consume the assembled batch — which is exactly the step-loop order (§9.1).
struct BatchSeqInput {
  // The tokens this sequence feeds this step. Prefill: the whole context to
  // (re)materialize — the prompt, or prompt ++ generated on resume of a
  // preempted sequence (§10.2). Decode: the single last-sampled token. Must be
  // non-empty; for a decode assembly it must be exactly one token.
  std::span<const std::int32_t> token_ids;
  // This sequence's KV cache (non-null). Positions are derived from its
  // `length()`, so a fresh prefill cache yields positions [0, T) and a decode
  // cache of length L yields position L (§8.2).
  kvcache::KvCache* cache = nullptr;
  // The configured sampler (params + resolved seed; borrowed). Feeds the
  // batched sampler after the forward (§8.4).
  const sampling::Sampler* sampler = nullptr;
  // The sampler's per-row context (prompt ids + generated-so-far; spans
  // borrowed). Its `generated_ids.size()` is the step index the RNG keys on.
  // The `{}` default (empty spans) keeps designated initializers that omit it
  // warning-clean.
  sampling::SampleContext context{};
};

// The flattened batch inputs (design §8.2).
// `token_ids`/`positions`/`cu_seqlens` are always populated;
// `block_table`/`seq_lens` only for a decode assembly (a prefill reads K/V
// through `caches[b]->view(layer)`, so it needs neither).
struct BatchInputs {
  std::vector<std::int32_t> token_ids;    // [Σ T_b], flattened batch-major
  std::vector<std::int32_t> positions;    // [Σ T_b], per-token absolute pos
  std::vector<std::int32_t> cu_seqlens;   // [B+1], prefix sums of T_b
  std::vector<kvcache::KvCache*> caches;  // [B], one cache per sequence
  std::vector<sampling::BatchRow> sample_rows;  // [B], sampler + context
  // Decode only: [B, max_blocks] int32, row b = caches[b]'s block table padded
  // with −1 to max_blocks = max_b ⌈len_b/bs⌉ (paged-kv-cache.md §9.4).
  // Undefined for a prefill assembly or an empty batch.
  tensor::Tensor block_table;
  std::vector<std::int32_t> seq_lens;  // decode only: [B] cache lengths

  [[nodiscard]] std::int64_t num_seqs() const {
    return static_cast<std::int64_t>(caches.size());
  }
  [[nodiscard]] std::int64_t num_tokens() const {
    return static_cast<std::int64_t>(token_ids.size());
  }

  // The batched `model::ForwardRequest` over these inputs (design §8.1): the
  // spans point into this struct's vectors, `cu_seqlens`/`caches` are set (the
  // batched path), and the single-sequence `cache` is left null. The referents
  // must outlive the returned request. `hook` is optional
  // (debug/observability).
  [[nodiscard]] model::ForwardRequest MakeForwardRequest(
      model::LogitsMode mode = model::LogitsMode::kLast,
      model::ActivationHook* hook = nullptr) const;
};

// Assembles a `BatchInputs` from per-sequence descriptors in one pass, reusing
// its staging across steps (allocation-free once the high-water batch reached).
// Not thread-safe — one per engine loop, called once per pass per step.
class BatchAssembler {
 public:
  // `allocator` backs the block-table tensor storage; null → the process
  // default CPU allocator. Borrowed; must outlive the assembler.
  explicit BatchAssembler(memory::Allocator* allocator = nullptr);

  // Assemble a prefill batch: `token_ids`/`positions`/`cu_seqlens`/`caches`/
  // `sample_rows` populated; `block_table` left undefined and `seq_lens`
  // cleared (prefill attention reads K/V via `view(layer)`). Positions for
  // sequence b are `cache->length() + t` for t in [0, T_b). `seqs.empty()` is a
  // no-op success (an empty pass is legal, §7). A null cache/sampler or empty
  // `token_ids` is `InvalidArgument` naming the sequence index; on error the
  // staging is left cleared.
  [[nodiscard]] core::Status AssemblePrefill(
      std::span<const BatchSeqInput> seqs);

  // Assemble a decode batch: as above plus `seq_lens[b] = cache->length()` and
  // the `[B, max_blocks]` `block_table` from each cache's `paged_view(0)`. Each
  // sequence must contribute exactly one token (T_b == 1); a non-paged cache's
  // `paged_view` `Unimplemented` propagates. Same error posture as
  // `AssemblePrefill`.
  [[nodiscard]] core::Status AssembleDecode(
      std::span<const BatchSeqInput> seqs);

  [[nodiscard]] const BatchInputs& inputs() const { return inputs_; }

  // Total bytes currently reserved by the staging (vector capacities +
  // block-table storage capacity) — the metric the allocation-free-after-
  // warm-up test watches for stability.
  [[nodiscard]] std::size_t staging_bytes() const;

 private:
  // Clears the staging (keeping capacity) and fills the fields common to both
  // passes: token_ids, positions, cu_seqlens, caches, sample_rows. Validates
  // each descriptor (non-null cache/sampler, non-empty tokens; and T_b == 1
  // when `decode`). Returns the flattened token count on success.
  [[nodiscard]] core::Status Flatten(std::span<const BatchSeqInput> seqs,
                                     bool decode);

  // Grow `block_table_storage_` to hold at least `count` int32 (high-water),
  // returning a writable pointer to its base.
  [[nodiscard]] core::StatusOr<std::int32_t*> EnsureBlockTable(
      std::int64_t count);

  memory::Allocator* allocator_;
  BatchInputs inputs_;
  tensor::Tensor block_table_storage_;     // 1-D int32, grown on demand
  std::int64_t block_table_capacity_ = 0;  // elements in block_table_storage_
};

}  // namespace engine::engine
