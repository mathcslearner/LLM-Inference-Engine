#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/model.h"
#include "sampling/batched_sampler.h"
#include "sampling/sampler.h"

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
// staging vectors keep their capacity across steps, so a steady-state step
// allocates nothing (the same grow-on-demand discipline as
// `Workspace`/`BatchedSampler`; this closes optimized-cpu-execution.md §6.3's
// "pre-size staging from the batch-token budget" note — the staging *is* sized
// by the batch budget).
//
// As built (M9-T07): the assembler carries **no** block-table tensor or
// per-sequence lengths. The original §8.2 sketch had a decode assembly build a
// `[B, max_blocks]` block-table tensor + `seq_lens` for the batched decode
// kernel — but block growth happens *inside* the forward (per-sequence
// `append`, §8.3), so a pre-forward snapshot is stale the moment a
// boundary-crossing token allocates a new block. The batched decode kernel
// therefore self-sources each sequence's block table + length through
// `paged_view(layer)` *after* the append (paged-kv-cache.md §9.4 as-built,
// scheduler-runtime.md §8.4 as-built), and the assembly is purely the flattened
// token/position/cu_seqlens/caches/sample_rows bundle for both passes.
//
// Layering (ADR-002). `engine` is a layer-3 module; batch assembly touches the
// `model::ForwardRequest` struct, so it belongs here beside `Generate`, **not**
// in the tensor-free `scheduler` (which stays a `core`-only decision leaf,
// ADR-002 rule 4 — it never sees a `Sequence` or cache). The assembler consumes
// plain `BatchSeqInput` descriptors the runtime fills from a `SchedulerOutput`
// + its `Sequence`s, so no `runtime`/`scheduler` type crosses into `engine`
// (the §8.2 shorthand `AssembleBatch(SchedulerOutput, sequences)` is realized
// as this descriptor-struct input). It links only already-present lower layers
// — `model`, `kvcache`, `sampling` — so there is no new ADR edge.

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

// The flattened batch inputs (design §8.2). Identical for prefill and decode:
// the batched forward slices K/V per sequence by `cu_seqlens` and, for decode,
// self-sources each sequence's block table + length from `paged_view(layer)`
// after the append (§8.4 as-built) — so no block-table tensor or lengths are
// staged here.
struct BatchInputs {
  std::vector<std::int32_t> token_ids;    // [Σ T_b], flattened batch-major
  std::vector<std::int32_t> positions;    // [Σ T_b], per-token absolute pos
  std::vector<std::int32_t> cu_seqlens;   // [B+1], prefix sums of T_b
  std::vector<kvcache::KvCache*> caches;  // [B], one cache per sequence
  std::vector<sampling::BatchRow> sample_rows;  // [B], sampler + context

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
  BatchAssembler() = default;

  // Assemble a prefill batch: `token_ids`/`positions`/`cu_seqlens`/`caches`/
  // `sample_rows` populated. Positions for sequence b are `cache->length() + t`
  // for t in [0, T_b). `seqs.empty()` is a no-op success (an empty pass is
  // legal, §7). A null cache/sampler or empty `token_ids` is `InvalidArgument`
  // naming the sequence index; on error the staging is left cleared.
  [[nodiscard]] core::Status AssemblePrefill(
      std::span<const BatchSeqInput> seqs);

  // Assemble a decode batch: as `AssemblePrefill`, but each sequence must
  // contribute exactly one token (T_b == 1). The batched decode kernel
  // self-sources block tables + lengths post-append (§8.4 as-built), so nothing
  // paged is read here. Same error posture as `AssemblePrefill`.
  [[nodiscard]] core::Status AssembleDecode(
      std::span<const BatchSeqInput> seqs);

  [[nodiscard]] const BatchInputs& inputs() const { return inputs_; }

  // Total bytes currently reserved by the staging vectors (capacities) — the
  // metric the allocation-free-after-warm-up test watches for stability.
  [[nodiscard]] std::size_t staging_bytes() const;

 private:
  // Clears the staging (keeping capacity) and fills the fields common to both
  // passes: token_ids, positions, cu_seqlens, caches, sample_rows. Validates
  // each descriptor (non-null cache/sampler, non-empty tokens; and T_b == 1
  // when `decode`).
  [[nodiscard]] core::Status Flatten(std::span<const BatchSeqInput> seqs,
                                     bool decode);

  BatchInputs inputs_;
};

}  // namespace engine::engine
