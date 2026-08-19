#pragma once

#include <cstdint>

// Host-machine introspection (M8-T07; design: docs/design/paged-kv-cache.md
// §5.3). A tiny per-platform utility, kept in `core` because the KV-budget
// resolver (`kvcache/kv_budget.h`) consumes it to turn a fractional
// `--kv-cache-memory` budget into an absolute byte ceiling. The only entry so
// far is total physical RAM; more host facts can join it here if a later
// milestone needs them.

namespace engine::core {

// Total physical RAM in bytes, or 0 when it cannot be determined. macOS reads
// `hw.memsize` (sysctl); Linux multiplies `_SC_PHYS_PAGES` by `_SC_PAGE_SIZE`;
// any other platform (or a failed query) returns 0. A 0 result means "unknown"
// — the fractional budget spelling is unusable and the caller must fall back to
// an absolute budget (kv_budget.h states the error).
[[nodiscard]] std::int64_t host_memory_bytes();

}  // namespace engine::core
