#include "memory/allocator.h"

#include "core/check.h"
#include "core/status.h"
#include "tensor/device.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <utility>

namespace engine::memory {

namespace {

[[nodiscard]] constexpr bool IsPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

Buffer::Buffer(void* data, std::size_t size_bytes, tensor::Device device,
               Deleter deleter)
    : data_(data),
      size_bytes_(size_bytes),
      device_(device),
      deleter_(std::move(deleter)) {
  CHECK(deleter_ != nullptr, "engaged Buffer requires a deleter");
  CHECK((data_ == nullptr) == (size_bytes_ == 0),
        "data must be null iff size_bytes is 0 (data null: {}, size_bytes: {})",
        data_ == nullptr, size_bytes_);
}

Buffer::Buffer(Buffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)),
      device_(std::exchange(other.device_, tensor::Device::Cpu())),
      deleter_(std::exchange(other.deleter_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    if (deleter_) {
      deleter_(data_);
    }
    data_ = std::exchange(other.data_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    device_ = std::exchange(other.device_, tensor::Device::Cpu());
    deleter_ = std::exchange(other.deleter_, nullptr);
  }
  return *this;
}

Buffer::~Buffer() {
  if (deleter_) {
    deleter_(data_);
  }
}

CpuAllocator::CpuAllocator(std::size_t default_alignment)
    : default_alignment_(default_alignment) {
  CHECK(IsPowerOfTwo(default_alignment_),
        "default_alignment must be a power of two, got {}", default_alignment_);
}

core::StatusOr<Buffer> CpuAllocator::Allocate(std::size_t bytes,
                                              std::size_t alignment) {
  CHECK(IsPowerOfTwo(alignment), "alignment must be a power of two, got {}",
        alignment);
  const Buffer::Deleter deleter = [](void* ptr) { std::free(ptr); };
  if (bytes == 0) {
    return Buffer(nullptr, 0, device(), deleter);
  }
  // std::aligned_alloc requires the size to be a multiple of the alignment
  // and the alignment to be one the implementation supports; over-align to
  // max_align_t so sub-fundamental requests (1, 2, ...) are always valid.
  const std::size_t effective_alignment =
      std::max(alignment, alignof(std::max_align_t));
  if (bytes >
      std::numeric_limits<std::size_t>::max() - (effective_alignment - 1)) {
    return core::OutOfMemoryError(
        "CPU allocation of {} bytes (alignment {}) overflows size_t", bytes,
        alignment);
  }
  const std::size_t rounded_bytes =
      (bytes + effective_alignment - 1) & ~(effective_alignment - 1);
  void* data = std::aligned_alloc(effective_alignment, rounded_bytes);
  if (data == nullptr) {
    return core::OutOfMemoryError(
        "CPU allocation of {} bytes (alignment {}) failed", bytes, alignment);
  }
  // The Buffer reports the *requested* size, not the rounded one.
  return Buffer(data, bytes, device(), deleter);
}

Allocator* DefaultCpuAllocator() {
  // Intentionally leaked so the pointer stays valid during static
  // destruction (Buffers with self-contained deleters may outlive it
  // anyway, but callers may also allocate from atexit paths).
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static Allocator* const kInstance = new CpuAllocator();
  return kInstance;
}

}  // namespace engine::memory
