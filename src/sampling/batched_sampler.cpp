#include "sampling/batched_sampler.h"

#include "core/status.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "sampling/sampler.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

// M7-T06 batched sampler. The batch is threaded one sequence per chunk
// (`parallel_for` grain 1); each row runs the shared `detail::SampleRow` over
// its own reusable scratch slot, so the result is bit-identical to the
// single-sequence reference sampler and every step after the first allocates
// nothing (design §15.6). All shape validation is front-loaded before the
// parallel region (parallel_for §3.4: inside a region there is only per-row
// arithmetic that writes disjoint `out`/scratch/status slots — no throwing, no
// shared mutation).
//
// Per-row failure isolation (M9-T10; design §11.2): each row's recoverable
// outcome (Ok, or a bad-history-id / non-finite-row error) is written into the
// caller's `row_status[b]`, always, for every row — the batch is never
// discarded for one bad row. The batch-level return covers only the
// front-loaded shape/null validation of the engine-built inputs.

namespace engine::sampling {

BatchedSampler::BatchedSampler(parallel::ThreadPool& pool) : pool_(pool) {}

core::Status BatchedSampler::Sample(std::span<const float> logits,
                                    std::int64_t vocab,
                                    std::span<const BatchRow> rows,
                                    std::span<SampleResult> out,
                                    std::span<core::Status> row_status) {
  if (vocab <= 0) {
    return core::InvalidArgumentError("vocab must be > 0 (got {})", vocab);
  }
  if (out.size() != rows.size()) {
    return core::InvalidArgumentError("out size {} must equal rows size {}",
                                      out.size(), rows.size());
  }
  if (row_status.size() != rows.size()) {
    return core::InvalidArgumentError(
        "row_status size {} must equal rows size {}", row_status.size(),
        rows.size());
  }
  const auto batch = static_cast<std::int64_t>(rows.size());
  const auto expected =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(vocab);
  if (logits.size() != expected) {
    return core::InvalidArgumentError(
        "logits size {} must equal batch {} * vocab {}", logits.size(), batch,
        vocab);
  }
  for (std::size_t b = 0; b < rows.size(); ++b) {
    if (rows[b].sampler == nullptr) {
      return core::InvalidArgumentError("rows[{}].sampler is null", b);
    }
  }
  if (rows.empty()) {
    return core::OkStatus();
  }

  // Grow (never shrink) the per-row scratch to the batch size; existing slots
  // keep their capacity, so a stable-size step loop is allocation-free.
  if (scratch_.size() < rows.size()) {
    scratch_.resize(rows.size());
  }

  // Each iteration writes exactly its own `row_status[b]` (and `out[b]` on
  // success) — disjoint slots, so no locking and every slot is fully defined
  // after the region even if the caller passed a span with stale values.
  parallel::parallel_for(
      pool_, batch, /*grain=*/1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
          const auto b = static_cast<std::size_t>(i);
          const BatchRow& row = rows[b];
          const std::span<const float> row_logits = logits.subspan(
              static_cast<std::size_t>(i) * static_cast<std::size_t>(vocab),
              static_cast<std::size_t>(vocab));
          const bool want_logprobs = row.sampler->params().logprobs > 0;
          core::StatusOr<SampleResult> result = detail::SampleRow(
              row_logits, row.context, row.sampler->params(),
              row.sampler->seed(), want_logprobs, scratch_[b]);
          if (result.ok()) {
            out[b] = std::move(result).value();
            row_status[b] = core::OkStatus();
          } else {
            row_status[b] = std::move(result).status();
          }
        }
      });

  // Batch-level validation passed; per-row outcomes live in `row_status`.
  return core::OkStatus();
}

std::size_t BatchedSampler::scratch_bytes() const {
  std::size_t bytes = 0;
  for (const detail::RowScratch& s : scratch_) {
    bytes += s.logits.capacity() * sizeof(float);
    bytes += s.probs.capacity() * sizeof(double);
    bytes += s.lp.capacity() * sizeof(double);
    bytes += s.exp_scratch.capacity() * sizeof(float);
    bytes += s.top_p_order.capacity() * sizeof(std::int32_t);
  }
  return bytes;
}

}  // namespace engine::sampling
