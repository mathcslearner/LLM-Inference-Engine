#include "engine/batch.h"

#include "core/status.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace engine::engine {

namespace {

// Bytes reserved by a vector (capacity, not size). For a pointer-element
// vector (e.g. the caches list) the intended element size is the pointer size
// itself, so the sizeof-pointer diagnostic is a false positive here.
template <typename T>
[[nodiscard]] std::size_t VectorBytes(const std::vector<T>& v) {
  return v.capacity() * sizeof(T);  // NOLINT(bugprone-sizeof-expression)
}

}  // namespace

model::ForwardRequest BatchInputs::MakeForwardRequest(
    model::LogitsMode mode, model::ActivationHook* hook) const {
  return model::ForwardRequest{
      .token_ids = token_ids,
      .positions = positions,
      .cache = nullptr,  // batched path: sequences carried by `caches`
      .logits_mode = mode,
      .hook = hook,
      .cu_seqlens = cu_seqlens,
      .caches = caches,
  };
}

BatchAssembler::BatchAssembler(memory::Allocator* allocator)
    : allocator_(allocator) {}

core::Status BatchAssembler::Flatten(std::span<const BatchSeqInput> seqs,
                                     bool decode) {
  inputs_.token_ids.clear();
  inputs_.positions.clear();
  inputs_.cu_seqlens.clear();
  inputs_.caches.clear();
  inputs_.sample_rows.clear();
  inputs_.seq_lens.clear();
  inputs_.block_table = tensor::Tensor{};

  inputs_.cu_seqlens.push_back(0);
  std::int32_t running = 0;
  for (std::size_t b = 0; b < seqs.size(); ++b) {
    const BatchSeqInput& s = seqs[b];
    if (s.cache == nullptr) {
      return core::InvalidArgumentError(
          "BatchAssembler: sequence {} has a null cache", b);
    }
    if (s.sampler == nullptr) {
      return core::InvalidArgumentError(
          "BatchAssembler: sequence {} has a null sampler", b);
    }
    const auto t_b = static_cast<std::int64_t>(s.token_ids.size());
    if (t_b < 1) {
      return core::InvalidArgumentError(
          "BatchAssembler: sequence {} has empty token_ids", b);
    }
    if (decode && t_b != 1) {
      return core::InvalidArgumentError(
          "BatchAssembler: decode sequence {} must contribute exactly one "
          "token (got {})",
          b, t_b);
    }

    // Positions are absolute: sequence b's tokens continue from its committed
    // cache length (a fresh prefill cache ⇒ [0, T); a decode cache of length L
    // ⇒ position L). §8.2.
    const std::int64_t base = s.cache->length();
    for (std::int64_t t = 0; t < t_b; ++t) {
      inputs_.token_ids.push_back(s.token_ids[static_cast<std::size_t>(t)]);
      inputs_.positions.push_back(static_cast<std::int32_t>(base + t));
    }
    running += static_cast<std::int32_t>(t_b);
    inputs_.cu_seqlens.push_back(running);
    inputs_.caches.push_back(s.cache);
    inputs_.sample_rows.push_back(
        sampling::BatchRow{.sampler = s.sampler, .context = s.context});
  }
  return core::OkStatus();
}

core::Status BatchAssembler::AssemblePrefill(
    std::span<const BatchSeqInput> seqs) {
  return Flatten(seqs, /*decode=*/false);
  // block_table stays undefined and seq_lens empty: prefill attention reads K/V
  // through caches[b]->view(layer), so it needs neither (§8.3).
}

core::StatusOr<std::int32_t*> BatchAssembler::EnsureBlockTable(
    std::int64_t count) {
  if (block_table_capacity_ < count) {
    ASSIGN_OR_RETURN(
        block_table_storage_,
        tensor::Tensor::empty(tensor::Shape{count}, tensor::DataType::kInt32,
                              tensor::Device::Cpu(), allocator_));
    block_table_capacity_ = count;
  }
  return block_table_storage_.data_ptr<std::int32_t>();
}

core::Status BatchAssembler::AssembleDecode(
    std::span<const BatchSeqInput> seqs) {
  RETURN_IF_ERROR(Flatten(seqs, /*decode=*/true));

  const auto num_seqs = static_cast<std::int64_t>(seqs.size());
  if (num_seqs == 0) {
    return core::OkStatus();  // empty pass: block_table undefined, seq_lens []
  }

  // Gather each sequence's paged view (block table + length) and the row width.
  struct RowView {
    const std::int32_t* blocks = nullptr;
    std::int64_t num_blocks = 0;
  };
  std::vector<RowView> views;
  views.reserve(seqs.size());
  inputs_.seq_lens.reserve(seqs.size());
  std::int64_t max_blocks = 0;
  for (const BatchSeqInput& s : seqs) {
    ASSIGN_OR_RETURN(kvcache::PagedKvView view, s.cache->paged_view(0));
    views.push_back(
        RowView{.blocks = view.block_table, .num_blocks = view.num_blocks});
    inputs_.seq_lens.push_back(static_cast<std::int32_t>(view.length));
    max_blocks = std::max(max_blocks, view.num_blocks);
  }

  // Build the [B, max_blocks] block-table tensor: row b = its blocks, −1
  // padded.
  const std::int64_t total = num_seqs * max_blocks;
  ASSIGN_OR_RETURN(std::int32_t* data, EnsureBlockTable(total));
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    std::int32_t* row = data + (b * max_blocks);
    const RowView& v = views[static_cast<std::size_t>(b)];
    for (std::int64_t j = 0; j < v.num_blocks; ++j) {
      row[j] = v.blocks[j];
    }
    for (std::int64_t j = v.num_blocks; j < max_blocks; ++j) {
      row[j] = -1;
    }
  }
  ASSIGN_OR_RETURN(const tensor::Tensor flat,
                   block_table_storage_.slice(0, 0, total));
  ASSIGN_OR_RETURN(inputs_.block_table,
                   flat.reshape(tensor::Shape{num_seqs, max_blocks}));
  return core::OkStatus();
}

std::size_t BatchAssembler::staging_bytes() const {
  return VectorBytes(inputs_.token_ids) + VectorBytes(inputs_.positions) +
         VectorBytes(inputs_.cu_seqlens) + VectorBytes(inputs_.caches) +
         VectorBytes(inputs_.sample_rows) + VectorBytes(inputs_.seq_lens) +
         (static_cast<std::size_t>(block_table_capacity_) *
          sizeof(std::int32_t));
}

}  // namespace engine::engine
