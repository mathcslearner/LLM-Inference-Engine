#pragma once

#include "core/status.h"

#include <cstdint>
#include <string_view>

// KV-cache memory budget resolution (M8-T07; design:
// docs/design/paged-kv-cache.md §5). Pure, allocation-free helpers that turn
// the `--kv-cache-memory` flag into an absolute byte budget the block pool is
// sized from, and the `--cache-capacity N` tokens spelling into a block count.
// No host introspection lives here (the resolver takes host RAM as an input, so
// it stays deterministic and unit-testable); the driver supplies it via
// `core::host_memory_bytes()`.

namespace engine::kvcache {

// A parsed `--kv-cache-memory` value (§5.1). Either an absolute byte budget or
// a fraction of host RAM (from which the KV budget is derived — §5.3).
struct KvCacheMemorySpec {
  enum class Kind : std::uint8_t { kAbsoluteBytes, kFraction };
  Kind kind = Kind::kAbsoluteBytes;
  std::int64_t bytes = 0;  // valid when kind == kAbsoluteBytes; > 0
  double fraction = 0.0;   // valid when kind == kFraction; in (0, 1]
};

// Parses the `--kv-cache-memory` spelling (§5.1):
//   - Absolute: an integer or decimal with an optional unit suffix
//     (`GiB`/`MiB`/`KiB` = 1024-based, `GB`/`MB`/`KB` = 1000-based, `B` or no
//     suffix = bytes) — `2GiB`, `1500MiB`, `8000000000`.
//   - Fraction: a bare decimal containing `.` in (0, 1] — `0.6`, `1.0`.
// Disambiguation: a unit suffix ⇒ absolute; otherwise a value containing `.` ⇒
// fraction, a bare integer ⇒ absolute bytes. A non-positive/over-1 fraction, a
// non-positive byte count, or unparseable text → `InvalidArgument` naming the
// input.
[[nodiscard]] core::StatusOr<KvCacheMemorySpec> ParseKvCacheMemory(
    std::string_view text);

// The three memory terms the fractional budget subtracts (§5.3). Absolute
// budgets ignore all three.
struct KvBudgetInputs {
  std::int64_t host_ram_bytes = 0;   // core::host_memory_bytes(); 0 == unknown
  std::int64_t weights_bytes = 0;    // model::weight_resident_bytes(loaded)
  std::int64_t workspace_bytes = 0;  // Workspace::BytesFor(config, max_tokens)
};

// Resolves `spec` to an absolute KV byte budget (§5.2/§5.3). Absolute specs
// pass through unchanged. A fractional spec computes
// `fraction·host_ram − weights − workspace`:
//   - host RAM unknown (`host_ram_bytes <= 0`) → `FailedPrecondition` (the
//     operator must use an absolute budget);
//   - a non-positive result → `ResourceExhausted` naming all three terms, so
//     the operator sees why nothing is left for KV.
[[nodiscard]] core::StatusOr<std::int64_t> ResolveKvBudgetBytes(
    const KvCacheMemorySpec& spec, const KvBudgetInputs& inputs);

// Blocks needed to hold `tokens` tokens at `block_size` — `⌈tokens / bs⌉`, the
// `--cache-capacity N` tokens spelling (§5.1). `tokens <= 0` → 1 (an empty
// sequence still owns no blocks, but the pool needs at least one to be usable);
// `block_size` must be > 0 (CHECK — the pool validates the value).
[[nodiscard]] std::int64_t BlocksForTokens(std::int64_t tokens, int block_size);

}  // namespace engine::kvcache
