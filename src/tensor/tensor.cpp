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
// or viewable — until their milestone lands (kInt4: M12, kFP8E4M3: M13).
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
    return core::UnimplementedError(
        "cannot allocate on {}: the CUDA backend lands in M2",
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
