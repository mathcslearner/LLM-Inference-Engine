#pragma once

#include "core/status.h"
#include "memory/allocator.h"

#include <filesystem>
#include <memory>

// Read-only file mapping (M4-T04; design: docs/design/model-loading.md §3.3).
//
// One function, not a class: the existing ownership vocabulary already says
// everything — a `memory::Buffer` with a self-contained deleter that
// munmaps, so the mapping lives until the last `shared_ptr` drops. Every
// `Tensor::from_buffer` view holds one, which is what keeps a checkpoint
// file mapped while any tensor view into it is alive.

namespace engine::model {

// Maps `path` read-only (PROT_READ, MAP_PRIVATE) and wraps the mapping in a
// shared `memory::Buffer` reporting `Device::Cpu()`. An empty file yields an
// engaged zero-size Buffer (allocator.h convention; mmap of length 0 is
// invalid and never attempted). The mapping is PROT_READ: checkpoint bytes
// are logically immutable, and a stray write through a view faults at the
// write site instead of silently corrupting the file's page-cache copy.
// Errors: `NotFound` when the file does not exist; `InvalidArgument` for a
// non-regular file or any other OS failure, carrying strerror(errno) and
// the path.
[[nodiscard]] core::StatusOr<std::shared_ptr<memory::Buffer>> MapFileReadOnly(
    const std::filesystem::path& path);

}  // namespace engine::model
