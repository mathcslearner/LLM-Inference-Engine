#pragma once

#include "core/status.h"
#include "tensor/device.h"

#include <cstddef>
#include <functional>

// Buffer & Allocator (M1-T05; design: docs/design/tensor.md §6).
//
// `Buffer` is a move-only owning handle to one contiguous allocation;
// `Allocator` is the abstract source of Buffers. This is the `memory →
// tensor_base` edge from ADR-002 Amendment 1: this header may include the
// header-only tensor value types (Device) and core, nothing else.
//
// Ownership model: deleters are self-contained. A Buffer never holds a
// pointer back to its Allocator; the deleter closure captures whatever it
// needs to free the memory, so a Buffer may safely outlive the allocator
// that made it (an implementation whose deleter *does* reference the
// allocator — e.g. the M2 caching pool — must document that lifetime
// requirement itself). There is no Deallocate on the Allocator interface
// and no release() on Buffer: deallocation is exclusively the deleter's
// job, which keeps "deleter runs exactly once" trivially true.

namespace engine::memory {

// Move-only owning handle to one contiguous allocation. Destroying an
// engaged Buffer invokes its deleter exactly once — including for zero-size
// buffers, whose data() is nullptr. A moved-from Buffer is disengaged and
// indistinguishable from a default-constructed one (null data, zero size,
// cpu device, no deleter): destruction is a no-op.
class Buffer {
 public:
  using Deleter = std::function<void(void*)>;

  Buffer() = default;  // disengaged

  // Takes ownership of `data`. Preconditions (CHECK — Buffers are built by
  // Allocator implementations, not from external input): `deleter` is
  // non-null, and `data` is null iff `size_bytes` is 0.
  Buffer(void* data, std::size_t size_bytes, tensor::Device device,
         Deleter deleter);

  Buffer(Buffer&& other) noexcept;             // transfers ownership
  Buffer& operator=(Buffer&& other) noexcept;  // destroys current, transfers
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer();  // runs deleter if engaged

  // nullptr iff size_bytes() == 0.
  [[nodiscard]] void* data() const { return data_; }
  [[nodiscard]] std::size_t size_bytes() const { return size_bytes_; }
  [[nodiscard]] tensor::Device device() const { return device_; }

 private:
  // Engagement is tracked by `deleter_` being non-empty — distinct from
  // `data_ == nullptr`, which also holds for engaged zero-size buffers.
  void* data_ = nullptr;
  std::size_t size_bytes_ = 0;
  tensor::Device device_ = tensor::Device::Cpu();
  Deleter deleter_;
};

// Abstract source of Buffers for one device. Implementations must be
// thread-safe: Allocate may be called concurrently from any thread.
class Allocator {
 public:
  Allocator() = default;
  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;
  virtual ~Allocator() = default;

  // Returns uninitialized memory of at least `bytes` bytes, aligned to
  // `alignment`. `alignment` must be a power of two (CHECK — a
  // caller-authored constant, not runtime input). `bytes == 0` is valid and
  // returns an engaged Buffer with data() == nullptr whose deleter still
  // runs exactly once. Failure (e.g. OOM) is kOutOfMemory — allocation
  // never throws (ADR-003).
  [[nodiscard]] virtual core::StatusOr<Buffer> Allocate(
      std::size_t bytes, std::size_t alignment) = 0;

  // The device whose memory this allocator hands out; every returned
  // Buffer's device() matches.
  [[nodiscard]] virtual tensor::Device device() const = 0;

 protected:
  // The base is stateless, so moving it is trivial; concrete allocators need
  // it to be returned by value from StatusOr factories (CudaAllocator::Create,
  // M2-T05). Protected so a derived object is never moved through a base
  // reference (which would slice its state).
  Allocator(Allocator&&) = default;
  Allocator& operator=(Allocator&&) = default;
};

// Aligned host allocator; default alignment 64 bytes (cache line; also
// satisfies every SIMD width we care about on x86-64 hosts).
class CpuAllocator final : public Allocator {
 public:
  // `default_alignment` must be a power of two (CHECK).
  explicit CpuAllocator(std::size_t default_alignment = 64);

  [[nodiscard]] core::StatusOr<Buffer> Allocate(std::size_t bytes,
                                                std::size_t alignment) override;

  // Convenience: allocate at this allocator's configured default alignment.
  [[nodiscard]] core::StatusOr<Buffer> Allocate(std::size_t bytes) {
    return Allocate(bytes, default_alignment_);
  }

  [[nodiscard]] tensor::Device device() const override {
    return tensor::Device::Cpu();
  }

  [[nodiscard]] std::size_t default_alignment() const {
    return default_alignment_;
  }

 private:
  std::size_t default_alignment_;
};

// Process-wide CpuAllocator instance used when Tensor factories are not
// given an explicit allocator. Never destroyed, so the pointer stays valid
// during static destruction.
[[nodiscard]] Allocator* DefaultCpuAllocator();

}  // namespace engine::memory
