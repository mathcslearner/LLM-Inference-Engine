#include "model/mapped_file.h"

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>

namespace engine::model {

core::StatusOr<std::shared_ptr<memory::Buffer>> MapFileReadOnly(
    const std::filesystem::path& path) {
  // v1 is POSIX-only (design §3.3): both supported platforms qualify; a
  // Windows port adds a CreateFileMapping branch behind this signature.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): open(2) is variadic.
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) {
      return core::NotFoundError("cannot open {}: {}", path.string(),
                                 std::strerror(errno));
    }
    return core::InvalidArgumentError("cannot open {}: {}", path.string(),
                                      std::strerror(errno));
  }
  struct stat file_stat{};
  if (::fstat(fd, &file_stat) != 0) {
    const int saved_errno = errno;
    ::close(fd);
    return core::InvalidArgumentError("cannot stat {}: {}", path.string(),
                                      std::strerror(saved_errno));
  }
  if (!S_ISREG(file_stat.st_mode)) {
    ::close(fd);
    return core::InvalidArgumentError("{} is not a regular file",
                                      path.string());
  }
  const auto size = static_cast<std::size_t>(file_stat.st_size);
  if (size == 0) {
    ::close(fd);
    // Engaged zero-size Buffer: data() == nullptr, deleter still runs
    // exactly once (allocator.h contract).
    return std::make_shared<memory::Buffer>(nullptr, 0, tensor::Device::Cpu(),
                                            [](void* /*data*/) {});
  }
  void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, /*offset=*/0);
  const int saved_errno = errno;
  ::close(fd);  // the mapping is independent of the descriptor
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast): MAP_FAILED
  // expands to a C-style cast in the system header.
  if (base == MAP_FAILED) {
    return core::InvalidArgumentError("cannot mmap {}: {}", path.string(),
                                      std::strerror(saved_errno));
  }
  return std::make_shared<memory::Buffer>(
      base, size, tensor::Device::Cpu(),
      [size](void* data) { ::munmap(data, size); });
}

}  // namespace engine::model
