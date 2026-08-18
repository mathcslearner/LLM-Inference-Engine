#include "core/check.h"
#include "core/status.h"
#include "cpu/detail.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <span>

namespace engine::cpu {

namespace {

// One row (one token) is one unit of parallel work: rows are independent
// gathers, so any partition is equivalent — bit-identical across thread counts.
constexpr std::int64_t kRowGrain = 1;

// y[t, :] = table[ids[t], :] widened to fp32, for a known table storage dtype.
// Bounds are validated up front (below), so the hot loop carries no per-row
// branch beyond the gather. Returns the first out-of-range id (index, value) so
// the caller can name it, or {-1, 0} when every id is valid.
template <typename TableT>
[[nodiscard]] std::pair<std::int64_t, std::int32_t> EmbeddingImpl(
    const TableT* table, std::span<const std::int32_t> ids, float* y,
    std::int64_t vocab, std::int64_t e_dim) {
  // Pre-scan for an out-of-range id: the gather itself must not touch a bad row
  // (that would be an out-of-bounds read), so validation precedes the threaded
  // fill. Serial and cheap (T ids).
  for (std::size_t t = 0; t < ids.size(); ++t) {
    const std::int32_t id = ids[t];
    if (id < 0 || id >= vocab) {
      return {static_cast<std::int64_t>(t), id};
    }
  }

  const auto rows = static_cast<std::int64_t>(ids.size());
  parallel::parallel_for(
      parallel::DefaultPool(), rows, kRowGrain,
      [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
          const TableT* src =
              table +
              (static_cast<std::int64_t>(ids[static_cast<std::size_t>(t)]) *
               e_dim);
          float* dst = y + (t * e_dim);
          for (std::int64_t e = 0; e < e_dim; ++e) {
            dst[e] = detail::Widen<TableT>(src[e]);
          }
        }
      });
  return {-1, 0};
}

}  // namespace

core::Status embedding_lookup(const tensor::Tensor& table,
                              std::span<const std::int32_t> ids,
                              tensor::Tensor& y) {
  CHECK(table.defined() && y.defined(),
        "cpu::embedding_lookup: table/y must be defined tensors");

  RETURN_IF_ERROR(detail::RequireContiguousRank(
      table, 2, "cpu::embedding_lookup", "table"));
  RETURN_IF_ERROR(
      detail::RequireContiguousRank(y, 2, "cpu::embedding_lookup", "y"));
  RETURN_IF_ERROR(detail::RequireF32(y, "cpu::embedding_lookup", "y"));

  const std::int64_t vocab = table.shape().dim(0);
  const std::int64_t e_dim = table.shape().dim(1);
  const auto rows = static_cast<std::int64_t>(ids.size());
  if (rows < 1 || e_dim < 1) {
    return core::InvalidArgumentError(
        "cpu::embedding_lookup: need T>=1 ids and E>=1 table, got T={} E={}",
        rows, e_dim);
  }
  if (y.shape().dim(0) != rows || y.shape().dim(1) != e_dim) {
    return core::InvalidArgumentError(
        "cpu::embedding_lookup: y must be [{}, {}] to match ids/table, got "
        "[{}, {}]",
        rows, e_dim, y.shape().dim(0), y.shape().dim(1));
  }

  auto* y_data = y.data_ptr<float>();
  std::pair<std::int64_t, std::int32_t> bad{-1, 0};
  switch (table.dtype()) {
    case tensor::DataType::kFloat32:
      bad = EmbeddingImpl<float>(table.data_ptr<float>(), ids, y_data, vocab,
                                 e_dim);
      break;
    case tensor::DataType::kFloat16:
      bad = EmbeddingImpl<tensor::float16>(table.data_ptr<tensor::float16>(),
                                           ids, y_data, vocab, e_dim);
      break;
    case tensor::DataType::kBFloat16:
      bad = EmbeddingImpl<tensor::bfloat16>(table.data_ptr<tensor::bfloat16>(),
                                            ids, y_data, vocab, e_dim);
      break;
    default:
      return core::InvalidArgumentError(
          "cpu::embedding_lookup: table dtype must be f32/f16/bf16, got {}",
          tensor::to_string(table.dtype()));
  }
  if (bad.first >= 0) {
    return core::InvalidArgumentError(
        "cpu::embedding_lookup: ids[{}] = {} is out of range for vocab size {}",
        bad.first, bad.second, vocab);
  }
  return core::OkStatus();
}

}  // namespace engine::cpu
