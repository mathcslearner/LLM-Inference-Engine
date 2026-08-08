#pragma once

#include "core/check.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <utility>

// Tensor (M1-T06; design: docs/design/tensor.md §7).
//
// A Tensor is a view onto shared storage: shared_ptr<memory::Buffer> + byte
// offset + Shape + Strides + DataType (+ Device, denormalized from the
// buffer for cheap access).
//
// Copying a Tensor is CHEAP and SHALLOW: it copies the handle, not the data.
// Copies and views alias the same storage — a write through one is visible
// through all. Deep copies are explicit (ops::copy, M1-T09). There is no
// distinction between "owning" and "view" tensors: every Tensor is a shared
// handle, uniformly, and the base allocation lives until the last handle
// dies.
//
// Thread safety (design §9): Tensor follows the shared_ptr model. Distinct
// Tensor handles — including copies and views of the same storage — may be
// used concurrently from different threads without synchronization;
// concurrent access to a *single* Tensor object is safe only if all access
// is const. Data races on the underlying bytes are the caller's problem:
// the tensor layer never locks data.
//
// Const-ness is shallow (design §8): `const Tensor&` makes the handle
// (metadata) immutable, not the pointed-to data — data() is const and
// returns a mutable pointer, like std::span.

namespace engine::tensor {

class Tensor {
 public:
  // Empty handle: no storage (defined() == false), rank 0, kFloat32, cpu.
  // Valid operations on it: destruction, assignment, defined(), and the
  // metadata accessors (which report those defaults); anything touching
  // storage — data access or views — CHECKs. A moved-from Tensor is also
  // undefined (defined() == false), but only destruction, assignment, and
  // defined() are meaningful on it: moves transfer the buffer without
  // resetting the remaining metadata, so its shape/dtype/device are
  // unspecified (design §7).
  Tensor() = default;

  // Allocates uninitialized, contiguous, row-major storage. `allocator` may
  // be null → the process default for `device` (DefaultCpuAllocator); a
  // non-null allocator whose device() differs from `device` is a programmer
  // error (CHECK). Reserved dtypes (kInt4, kFP8E4M3) return Unimplemented
  // until the quantization milestone, as does the reserved kCUDA device
  // (no GPU backend — ADR-004); allocation failure propagates
  // (kOutOfMemory).
  [[nodiscard]] static core::StatusOr<Tensor> empty(
      Shape shape, DataType dtype, Device device,
      memory::Allocator* allocator = nullptr);

  // Wraps existing storage as a contiguous, row-major Tensor view. The
  // Tensor shares ownership of `buffer`: storage lives until the last
  // handle dies (the general facility behind zero-copy checkpoint loading,
  // M4-T04 — model-loading.md §3.5 — but not a safetensors special case).
  // The window [byte_offset, byte_offset + numel×itemsize) must lie within
  // the buffer (InvalidArgument, overflow-checked); reserved dtypes (kInt4,
  // kFP8E4M3) return Unimplemented, consistent with empty(). Device comes
  // from the buffer. A null `buffer` is a programmer error (CHECK).
  [[nodiscard]] static core::StatusOr<Tensor> from_buffer(
      std::shared_ptr<memory::Buffer> buffer, std::size_t byte_offset,
      Shape shape, DataType dtype);

  // --- Metadata ---

  [[nodiscard]] const Shape& shape() const { return shape_; }
  [[nodiscard]] const Strides& strides() const { return strides_; }
  [[nodiscard]] DataType dtype() const { return dtype_; }
  [[nodiscard]] Device device() const { return device_; }
  [[nodiscard]] std::int64_t numel() const { return shape_.numel(); }
  [[nodiscard]] bool is_contiguous() const {
    return IsContiguous(shape_, strides_);
  }
  // False only for the default-constructed / moved-from handle. A tensor
  // with numel() == 0 from empty() IS defined (engaged zero-size buffer).
  [[nodiscard]] bool defined() const { return buffer_ != nullptr; }
  [[nodiscard]] std::size_t byte_offset() const { return byte_offset_; }

  // --- Views: new Tensors sharing this buffer ---
  //
  // Data-dependent misuse (bad ranges, mismatched numel, non-contiguous
  // reshape) is recoverable: InvalidArgument. Calling a view op on an
  // undefined handle is a programmer error: CHECK.

  // Half-open [start, end) along `dim`; requires 0 ≤ dim < rank() and
  // 0 ≤ start ≤ end ≤ shape().dim(dim). Result shares storage: byte offset
  // advances by start * strides()[dim] * itemsize, strides are unchanged,
  // shape dim becomes end - start. Slicing an outer dim keeps contiguity;
  // slicing an inner dim produces a legitimate non-contiguous view. No
  // negative indices, no step.
  [[nodiscard]] core::StatusOr<Tensor> slice(int dim, std::int64_t start,
                                             std::int64_t end) const;

  // Contiguous tensors only; numel must match. Returns a view — never
  // copies. Non-contiguous input → InvalidArgument telling the caller to
  // copy first.
  [[nodiscard]] core::StatusOr<Tensor> reshape(Shape new_shape) const;

  // Bit reinterpretation between dtypes of equal element size (e.g. kInt8 ↔
  // kUInt8, kFloat16 ↔ kBFloat16); same shape/strides/offset. Width
  // mismatch → InvalidArgument; reserved dtypes → Unimplemented (consistent
  // with empty()).
  [[nodiscard]] core::StatusOr<Tensor> view_as_dtype(DataType new_dtype) const;

  // --- Device transfer ---
  //
  // When already on `device`, returns *this: the same shared handle, no
  // copy (reshape's "never copies" philosophy; a deep copy is ops::copy
  // onto a fresh tensor). The source must be contiguous → InvalidArgument
  // otherwise, checked before the same-device fast path so the contract
  // does not depend on the destination. Any other destination (the
  // reserved kCUDA device) → Unimplemented: there is no GPU backend
  // (ADR-004); the signature is kept so a future backend restores
  // cross-device semantics without an API break. Undefined handle → CHECK.
  [[nodiscard]] core::StatusOr<Tensor> to(Device device) const;

  // --- Raw access ---

  // Buffer base plus byte_offset(). CHECKs defined(). nullptr only for
  // tensors whose underlying buffer is zero-sized (created with
  // numel() == 0); a zero-numel VIEW of a non-empty tensor keeps a non-null
  // (possibly past-the-end) pointer. So data() == nullptr implies
  // numel() == 0, but not the converse.
  [[nodiscard]] void* data() const {
    CHECK(defined(), "data() on an undefined Tensor");
    void* base = buffer_->data();
    if (base == nullptr) {
      return nullptr;  // numel-0 tensor: engaged zero-size buffer
    }
    return static_cast<std::byte*>(base) + byte_offset_;
  }

  // Typed access. CHECKs dtype_of<T> == dtype(): T is written in source, so
  // a mismatch is a bug by definition.
  template <typename T>
  [[nodiscard]] T* data_ptr() const {
    CHECK(dtype_of<T> == dtype_, "data_ptr<T>(): T is {} but dtype() is {}",
          to_string(dtype_of<T>), to_string(dtype_));
    return static_cast<T*>(data());
  }

  // CPU-only element accessor for tests and debugging. CHECKs: cpu device,
  // dtype match (via data_ptr), rank match, indices in range. Applies
  // strides — works on non-contiguous views. Never use on a hot path.
  template <typename T>
  [[nodiscard]] T item(std::span<const std::int64_t> indices) const {
    CHECK(device_.is_cpu(), "item() requires a CPU tensor; device is {}",
          device_.ToString());
    CHECK(static_cast<int>(indices.size()) == shape_.rank(),
          "item(): {} indices for a rank-{} tensor", indices.size(),
          shape_.rank());
    std::int64_t element_offset = 0;
    for (int i = 0; i < shape_.rank(); ++i) {
      const std::int64_t index = indices[static_cast<std::size_t>(i)];
      CHECK(index >= 0 && index < shape_.dim(i),
            "item(): index {} out of range for dim {} of size {}", index, i,
            shape_.dim(i));
      element_offset += index * strides_[i];
    }
    return data_ptr<T>()[element_offset];
  }

  // Convenience so call sites can write t.item<float>({1, 2}) — an
  // initializer_list does not convert to std::span until C++26.
  template <typename T>
  [[nodiscard]] T item(std::initializer_list<std::int64_t> indices) const {
    return item<T>(
        std::span<const std::int64_t>{indices.begin(), indices.size()});
  }

 private:
  Tensor(std::shared_ptr<memory::Buffer> buffer, std::size_t byte_offset,
         const Shape& shape, const Strides& strides, DataType dtype,
         Device device)
      : buffer_(std::move(buffer)),
        byte_offset_(byte_offset),
        shape_(shape),
        strides_(strides),
        dtype_(dtype),
        device_(device) {}

  std::shared_ptr<memory::Buffer> buffer_;
  std::size_t byte_offset_ = 0;
  Shape shape_;
  Strides strides_;
  DataType dtype_ = DataType::kFloat32;
  Device device_ = Device::Cpu();
};

}  // namespace engine::tensor
