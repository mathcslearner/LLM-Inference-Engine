#include "tensor/tensor.h"

#include "core/check.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace engine::tensor {

namespace {

// Alignment for storage allocated by Tensor::empty: one cache line, the same
// value and rationale as CpuAllocator's default (design §6). The abstract
// Allocator interface takes an explicit alignment, so empty() must pick one.
constexpr std::size_t kEmptyAlignment = 64;

// Reserved dtypes are representable (name/size-mapped) but not allocatable —
// or viewable — until the quantization milestones land (M13).
[[nodiscard]] constexpr bool IsReservedDataType(DataType dtype) {
  return dtype == DataType::kInt4 || dtype == DataType::kFP8E4M3;
}

}  // namespace

core::StatusOr<Tensor> Tensor::empty(Shape shape, DataType dtype, Device device,
                                     memory::Allocator* allocator) {
  if (IsReservedDataType(dtype)) {
    return core::UnimplementedError(
        "dtype {} is reserved and not allocatable until its milestone",
        to_string(dtype));
  }
  if (device.is_cuda()) {
    // kCUDA is a reserved Device value: the engine is CPU-first (ADR-004)
    // and has no GPU backend.
    return core::UnimplementedError(
        "device {} is not supported: the engine has no GPU backend (ADR-004)",
        device.ToString());
  }
  if (allocator == nullptr) {
    allocator = memory::DefaultCpuAllocator();
  }
  // Both arguments are call-site-authored; a mismatch is a bug, not data.
  CHECK(allocator->device() == device,
        "allocator device {} does not match requested device {}",
        allocator->device().ToString(), device.ToString());
  std::int64_t bytes = 0;
  if (__builtin_mul_overflow(
          shape.numel(), static_cast<std::int64_t>(itemsize(dtype)), &bytes)) {
    return core::InvalidArgumentError(
        "allocation size overflows int64_t: {} elements of {}", shape.numel(),
        to_string(dtype));
  }
  ASSIGN_OR_RETURN(
      memory::Buffer buffer,
      allocator->Allocate(static_cast<std::size_t>(bytes), kEmptyAlignment));
  const Strides strides = RowMajorStrides(shape);
  return Tensor(std::make_shared<memory::Buffer>(std::move(buffer)),
                /*byte_offset=*/0, shape, strides, dtype, device);
}

core::StatusOr<Tensor> Tensor::from_buffer(
    std::shared_ptr<memory::Buffer> buffer, std::size_t byte_offset,
    Shape shape, DataType dtype) {
  CHECK(buffer != nullptr, "from_buffer: buffer must be non-null");
  if (IsReservedDataType(dtype)) {
    return core::UnimplementedError(
        "dtype {} is reserved and not viewable until its milestone",
        to_string(dtype));
  }
  // Window arithmetic in uint64: numel is bounded by int64_t max, so
  // numel × itemsize (≤ 8) and the offset sum can each overflow.
  std::uint64_t view_bytes = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(shape.numel()),
                             static_cast<std::uint64_t>(itemsize(dtype)),
                             &view_bytes)) {
    return core::InvalidArgumentError(
        "from_buffer: view size overflows uint64_t: {} elements of {}",
        shape.numel(), to_string(dtype));
  }
  std::uint64_t view_end = 0;
  if (__builtin_add_overflow(static_cast<std::uint64_t>(byte_offset),
                             view_bytes, &view_end) ||
      view_end > buffer->size_bytes()) {
    return core::InvalidArgumentError(
        "from_buffer: view of {} bytes at offset {} exceeds buffer size {}",
        view_bytes, byte_offset, buffer->size_bytes());
  }
  const Device device = buffer->device();
  const Strides strides = RowMajorStrides(shape);
  return Tensor(std::move(buffer), byte_offset, shape, strides, dtype, device);
}

core::StatusOr<Tensor> Tensor::slice(int dim, std::int64_t start,
                                     std::int64_t end) const {
  CHECK(defined(), "slice() on an undefined Tensor");
  if (dim < 0 || dim >= shape_.rank()) {
    return core::InvalidArgumentError(
        "slice dim {} out of range for rank-{} shape {}", dim, shape_.rank(),
        shape_);
  }
  if (start < 0 || start > end || end > shape_.dim(dim)) {
    return core::InvalidArgumentError(
        "slice range [{}, {}) invalid for dim {} of size {}", start, end, dim,
        shape_.dim(dim));
  }
  std::array<std::int64_t, static_cast<std::size_t>(kMaxRank)> dims{};
  for (int i = 0; i < shape_.rank(); ++i) {
    dims[static_cast<std::size_t>(i)] = shape_.dim(i);
  }
  dims[static_cast<std::size_t>(dim)] = end - start;
  // Cannot fail: every dim is non-negative and bounded by the source shape.
  ASSIGN_OR_RETURN(
      const Shape new_shape,
      Shape::FromDims({dims.data(), static_cast<std::size_t>(shape_.rank())}));
  const std::size_t byte_offset =
      byte_offset_ + (static_cast<std::size_t>(start * strides_[dim]) *
                      static_cast<std::size_t>(itemsize(dtype_)));
  return Tensor(buffer_, byte_offset, new_shape, strides_, dtype_, device_);
}

core::StatusOr<Tensor> Tensor::reshape(Shape new_shape) const {
  CHECK(defined(), "reshape() on an undefined Tensor");
  if (new_shape.numel() != shape_.numel()) {
    return core::InvalidArgumentError(
        "reshape to {} ({} elements) from {} ({} elements): element counts "
        "must match",
        new_shape, new_shape.numel(), shape_, shape_.numel());
  }
  if (!is_contiguous()) {
    return core::InvalidArgumentError(
        "reshape requires a contiguous tensor (shape {}, strides {}); make a "
        "contiguous copy first",
        shape_, strides_);
  }
  const Strides strides = RowMajorStrides(new_shape);
  return Tensor(buffer_, byte_offset_, new_shape, strides, dtype_, device_);
}

core::StatusOr<Tensor> Tensor::to(Device device) const {
  CHECK(defined(), "to() on an undefined Tensor");
  // Contiguity is checked before the same-device fast path so the contract
  // does not depend on the destination (the retired M2 transfer contract,
  // kept for a future backend).
  if (!is_contiguous()) {
    return core::InvalidArgumentError(
        "to: tensor must be contiguous (shape {}, strides {}); make a "
        "contiguous copy first",
        shape_, strides_);
  }
  if (device == device_) {
    return *this;  // same shared handle; a deep copy is ops::copy
  }
  return core::UnimplementedError(
      "to: transfer {} -> {} is not supported: the engine has no GPU backend "
      "(ADR-004)",
      device_.ToString(), device.ToString());
}

core::StatusOr<Tensor> Tensor::view_as_dtype(DataType new_dtype) const {
  CHECK(defined(), "view_as_dtype() on an undefined Tensor");
  if (IsReservedDataType(new_dtype)) {
    return core::UnimplementedError(
        "dtype {} is reserved and not viewable until its milestone",
        to_string(new_dtype));
  }
  if (itemsize_bits(new_dtype) != itemsize_bits(dtype_)) {
    return core::InvalidArgumentError(
        "view_as_dtype requires equal element sizes: {} is {} bits, {} is {} "
        "bits",
        to_string(dtype_), itemsize_bits(dtype_), to_string(new_dtype),
        itemsize_bits(new_dtype));
  }
  return Tensor(buffer_, byte_offset_, shape_, strides_, new_dtype, device_);
}

}  // namespace engine::tensor
