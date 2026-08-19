#include "core/sysinfo.h"

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace engine::core {

std::int64_t host_memory_bytes() {
#if defined(__APPLE__)
  std::int64_t mem = 0;
  std::size_t size = sizeof(mem);
  if (::sysctlbyname("hw.memsize", &mem, &size, nullptr, 0) == 0 && mem > 0) {
    return mem;
  }
#elif defined(__linux__)
  const long pages = ::sysconf(_SC_PHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<std::int64_t>(pages) *
           static_cast<std::int64_t>(page_size);
  }
#endif
  return 0;
}

}  // namespace engine::core
