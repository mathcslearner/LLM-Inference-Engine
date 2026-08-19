#pragma once

#include "core/status.h"
#include "model/packed_linear.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <span>

// OptimizedEmbedding (M6-T06; design: docs/design/optimized-cpu-execution.md
// §7). The optimized backend's embedding lookup — the analog of the M5
// `Embedding` module, producing bit-identical fp32 rows via the dispatched
// `kernels::EmbeddingLookup*` gather. It carries two source layouts, chosen at
// build to honor tied embeddings without a `[V, E]` duplicate (§7):
//
//   - FromTable        : an untied model's own `[V, E]` checkpoint table
//                        (zero-copy, checkpoint storage dtype). The gather is
//                        the contiguous widen — bit-exact to the reference.
//   - FromPackedLinear : a tied model shares one physical copy with the packed
//                        lm_head. This holds the same packed bytes (the
//                        `PackedLinear::packed_weight()` handle, shared via the
//                        Tensor's refcounted storage — no second table) and
//                        gathers logical row `v` out of panel `v/kNr`, lane
//                        `v%kNr`. So lookup and projection use *different
//                        physical layouts of the same logical weight*.
//
// The header stays free of any `kernels` type (the packed bytes are a plain
// `tensor::Tensor`), so the `model -> kernels` edge remains a PRIVATE link,
// exactly as PackedLinear (§2.1).

namespace engine::model {

class OptimizedEmbedding {
 public:
  // Untied: build from the model's own embedding table `[V, E]` (rank-2,
  // contiguous, f32/f16/bf16). Validation mirrors `Embedding::Create` — same
  // messages. The table handle is retained zero-copy (checkpoint storage
  // dtype). Malformed weight -> InvalidArgument naming the problem.
  [[nodiscard]] static core::StatusOr<OptimizedEmbedding> FromTable(
      tensor::Tensor table);

  // Tied: share the packed lm_head's storage. Takes the already-built
  // `PackedLinear` for `lm_head`; the embedding gathers logical rows out of its
  // packed layout, holding the same physical bytes via the shared
  // `packed_weight()` handle (so the PackedLinear may then be moved into a
  // `unique_ptr<Linear>` — the storage outlives it). V = out_features,
  // E = in_features. Never fails (the PackedLinear was already validated).
  [[nodiscard]] static OptimizedEmbedding FromPackedLinear(
      const PackedLinear& lm_head);

  // y[T, E] = table[ids[t], :] widened to fp32. `ids` are absolute token ids in
  // [0, V) (the `ForwardRequest.token_ids` span). `y` is caller-allocated
  // contiguous fp32 [T, E]. An out-of-range id (pre-scanned here, naming the
  // index and value like `cpu::embedding_lookup`), or a wrong-shape/dtype `y`,
  // is InvalidArgument; then the dispatched gather runs.
  [[nodiscard]] core::Status forward(std::span<const std::int32_t> ids,
                                     tensor::Tensor& y) const;

  [[nodiscard]] std::int64_t vocab_size() const { return vocab_size_; }
  [[nodiscard]] std::int64_t hidden_size() const { return hidden_size_; }

  // True iff this shares the packed lm_head's storage (tied).
  // Test/introspection seam for the "one physical copy" property (§7).
  [[nodiscard]] bool shares_packed_storage() const { return packed_; }

  // The gather source — the checkpoint table (untied) or the packed lm_head
  // bytes (tied). Exposed so a test can assert tied storage is physically
  // shared with the lm_head (`source().data() ==
  // lm_head.packed_weight().data()`).
  [[nodiscard]] const tensor::Tensor& source() const { return source_; }

 private:
  OptimizedEmbedding(tensor::Tensor source, bool packed,
                     tensor::DataType source_dtype, std::int64_t vocab_size,
                     std::int64_t hidden_size)
      : source_(std::move(source)),
        packed_(packed),
        source_dtype_(source_dtype),
        vocab_size_(vocab_size),
        hidden_size_(hidden_size) {}

  tensor::Tensor source_;  // [V, E] table (untied) or [P, E, kNr] packed (tied)
  bool packed_ = false;
  tensor::DataType source_dtype_;
  std::int64_t vocab_size_ = 0;
  std::int64_t hidden_size_ = 0;
};

}  // namespace engine::model
