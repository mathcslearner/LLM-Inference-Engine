#pragma once

#include "core/status.h"
#include "parallel/thread_pool.h"
#include "sampling/sampler.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Batched sampling (M7-T06; design: docs/design/model-execution.md §15.6). The
// hot path for a batch of decoding sequences: sample one token per sequence
// from a `[batch, vocab]` logits block, in parallel across sequences. Each
// sequence carries its own `SamplingParams` (via its `Sampler`, which also
// holds the resolved Philox seed) and token history, so the batch may mix
// greedy and stochastic requests, different temperatures, penalties, top-k/p,
// and logprob counts freely.
//
// The batched sampler runs the *same* per-row pipeline as the single-sequence
// reference `Sampler` — literally `detail::SampleRow` (sampler.h) — so it picks
// **identical tokens given identical RNG counters** (the acceptance criterion),
// bit-for-bit, tie-breaks included. The optimization is threading across the
// batch (`parallel::parallel_for`) plus reusing per-sequence scratch buffers
// across decode steps, so a steady-state batched step allocates nothing. The
// per-row arithmetic (vectorized softmax via `kernels::ExpF32`, `nth_element`
// top-k, `partial_sort` logprobs) already lives in the shared stages.

namespace engine::sampling {

// One sequence's request within a batch: the configured `Sampler` (params +
// resolved seed; borrowed, must outlive the call) and this sequence's token
// history. `want_logprobs` follows `sampler->params().logprobs > 0`.
struct BatchRow {
  const Sampler* sampler = nullptr;
  SampleContext context;
};

// Samples a batch of sequences in parallel. Reusable across decode steps: it
// owns the per-sequence scratch and grows it on demand (allocation-free once
// the high-water batch size / vocab is reached). Not thread-safe against
// concurrent `Sample` calls on the same instance (it mutates the shared
// scratch); construct one per engine step-loop, call it once per step.
class BatchedSampler {
 public:
  // `pool` threads the batch; defaults to the process-wide pool. Borrowed.
  explicit BatchedSampler(parallel::ThreadPool& pool = parallel::DefaultPool());

  // Sample one token per row. `logits` is the `[batch, vocab]` row-major block
  // (`rows.size()` rows of `vocab` fp32 each); `out` and `row_status` (both
  // size `rows.size()`) receive, index-aligned with `rows`, one `SampleResult`
  // and one per-row `Status` respectively.
  //
  // Per-row failure isolation (M9-T10; design §11.2): every row is always
  // attempted and `row_status[b]` is always written — `Ok` on success (and
  // `out[b]` is the sampled result), or that row's error otherwise (a bad
  // history id → `InvalidArgument`, a non-finite logits row → `Internal`, an
  // empty row → `InvalidArgument`; `out[b]` is then left untouched). One bad
  // row never discards the batch, so the engine loop can fail only that row's
  // sequence and deliver every other row (§11.2).
  //
  // The returned `Status` reports *batch-level* validation only, checked before
  // any work and leaving `out`/`row_status` unspecified: a shape mismatch
  // (`logits.size() != rows.size() * vocab`, `out.size() != rows.size()`,
  // `row_status.size() != rows.size()`), a null `sampler`, or `vocab <= 0` —
  // all `InvalidArgument`. These are engine-built inputs, so a non-Ok return is
  // an engine bug, not request data. `rows.empty()` is a no-op success.
  [[nodiscard]] core::Status Sample(std::span<const float> logits,
                                    std::int64_t vocab,
                                    std::span<const BatchRow> rows,
                                    std::span<SampleResult> out,
                                    std::span<core::Status> row_status);

  // Total bytes currently held in the per-row scratch (capacity, not size) —
  // the metric the allocation-free-steady-state test watches for stability.
  [[nodiscard]] std::size_t scratch_bytes() const;

 private:
  parallel::ThreadPool& pool_;
  // One reusable scratch per batch slot; index b is only ever touched by the
  // single worker running row b (parallel_for grain 1), so no locking needed.
  std::vector<detail::RowScratch> scratch_;
};

}  // namespace engine::sampling
