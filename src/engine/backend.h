#pragma once

#include "core/status.h"
#include "model/registry.h"

#include <string_view>

// Backend selection for the generation loop (M5-T09; design:
// docs/design/model-execution.md §2, §8, §9). The `Backend` enum itself is
// defined in model/registry.h — it names a `BuildModel` parameter, and `model`
// cannot depend on `engine` (ADR-002: engine → model) — so this header
// re-exports it under the `engine` namespace for the generation-loop, CLI, and
// (M17) server call sites, and adds the string↔enum helpers those need. M6
// fills in the `kOptimized` arm behind the same enum, validated token-for-token
// against `kReference`.

namespace engine::engine {

using Backend = model::Backend;
using BuildOptions = model::BuildOptions;

// The stable lowercase name of a backend ("reference" | "optimized"), for logs,
// CLI flags, and config. Total over the enum.
[[nodiscard]] constexpr std::string_view BackendName(Backend backend) {
  switch (backend) {
    case Backend::kReference:
      return "reference";
    case Backend::kOptimized:
      return "optimized";
  }
  return "unknown";
}

// Parses a backend name (case-sensitive lowercase, matching `BackendName`). An
// unrecognized name → InvalidArgument listing the accepted names.
[[nodiscard]] inline core::StatusOr<Backend> ParseBackend(
    std::string_view name) {
  if (name == "reference") {
    return Backend::kReference;
  }
  if (name == "optimized") {
    return Backend::kOptimized;
  }
  return core::InvalidArgumentError(
      R"(unknown backend "{}" (expected "reference" or "optimized"))", name);
}

}  // namespace engine::engine
