#include "model/optimized_embedding.h"

#include "core/status.h"
#include "kernels/embedding.h"
#include "model/packed_linear.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <span>
#include <utility>

namespace engine::model {

namespace {

[[nodiscard]] bool IsSupportedFloat(tensor::DataType dtype) {
  return dtype == tensor::DataType::kFloat32 ||
         dtype == tensor::DataType::kFloat16 ||
         dtype == tensor::DataType::kBFloat16;
}

}  // namespace

core::StatusOr<OptimizedEmbedding> OptimizedEmbedding::FromTable(
    tensor::Tensor table) {
  // Validation mirrors Embedding::Create (same terms/messages) so the two
  // backends reject the same malformed weights identically.
  if (!table.defined()) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding: weight is undefined");
  }
  if (table.shape().rank() != 2) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding: weight must be rank-2 [V, E], got rank-{}",
        table.shape().rank());
  }
  if (!table.is_contiguous()) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding: weight must be contiguous");
  }
  if (!IsSupportedFloat(table.dtype())) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding: weight dtype must be f32/f16/bf16, got {}",
        tensor::to_string(table.dtype()));
  }
  const std::int64_t vocab_size = table.shape().dim(0);
  const std::int64_t hidden_size = table.shape().dim(1);
  const tensor::DataType dtype = table.dtype();
  return OptimizedEmbedding(std::move(table), /*packed=*/false, dtype,
                            vocab_size, hidden_size);
}

OptimizedEmbedding OptimizedEmbedding::FromPackedLinear(
    const PackedLinear& lm_head) {
  // Share the packed lm_head's storage: `packed_weight()` is a refcounted
  // Tensor handle, so copying it keeps the same physical bytes alive without a
  // [V, E] duplicate, and independently of the PackedLinear's own lifetime
  // (§7). V/E come from the linear's features (out=V rows, in=E columns).
  const tensor::Tensor& packed = lm_head.packed_weight();
  return {packed, /*packed=*/true, packed.dtype(), lm_head.out_features(),
          lm_head.in_features()};
}

core::Status OptimizedEmbedding::forward(std::span<const std::int32_t> ids,
                                         tensor::Tensor& y) const {
  // Front-load every recoverable check so the parallel region does only
  // move+widen (§5, ADR-003): shape/dtype of `y`, then the id-range pre-scan
  // (the kernel is a raw gather that must not touch a bad row).
  const auto rows = static_cast<std::int64_t>(ids.size());
  if (!y.defined()) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding::forward: y is undefined");
  }
  if (rows < 1) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding::forward: need T>=1 ids, got {}", rows);
  }
  if (y.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding::forward: y must be fp32, got {}",
        tensor::to_string(y.dtype()));
  }
  if (y.shape().rank() != 2 || y.shape().dim(0) != rows ||
      y.shape().dim(1) != hidden_size_) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding::forward: y must be caller-allocated [{}, {}], "
        "got {}",
        rows, hidden_size_, y.shape());
  }
  if (!y.is_contiguous()) {
    return core::InvalidArgumentError(
        "OptimizedEmbedding::forward: y must be contiguous");
  }
  for (std::int64_t t = 0; t < rows; ++t) {
    const std::int32_t id = ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= vocab_size_) {
      return core::InvalidArgumentError(
          "OptimizedEmbedding::forward: ids[{}] = {} is out of range for vocab "
          "size {}",
          t, id, vocab_size_);
    }
  }

  auto* y_data = y.data_ptr<float>();
  if (packed_) {
    kernels::EmbeddingLookupPackedF32(source_.data(), source_dtype_,
                                      hidden_size_, ids.data(), rows, y_data);
  } else {
    kernels::EmbeddingLookupF32(source_.data(), source_dtype_, hidden_size_,
                                ids.data(), rows, y_data);
  }
  return core::OkStatus();
}

}  // namespace engine::model
