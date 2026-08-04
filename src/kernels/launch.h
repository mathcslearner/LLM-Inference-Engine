#pragma once

#include "core/check.h"

#include <algorithm>
#include <cstdint>

// Launch-config helpers for 1-D grid-stride kernels (M2-T08; design:
// docs/design/cuda-backend.md §9.1).
//
// INTERNAL to src/kernels/, but toolkit-free and host-safe: LaunchConfig1D
// is plain arithmetic (unit-tested on every configuration, GPU or not), and
// CUDA_1D_KERNEL_LOOP is only ever *expanded* inside __global__ functions in
// .cu TUs — defining the macro in a host TU is harmless.

namespace engine::kernels {

// Fixed block size for 1-D elementwise launches. Under a grid-stride loop
// any block size is semantically correct; 256 is the conventional
// placeholder-quality default (kernel performance is an M2 non-goal).
inline constexpr unsigned int kDefaultBlockSize = 256;

// Grid cap: launches never exceed this many blocks; larger inputs wrap via
// the grid-stride loop. The cap exists so the wrap path is reachable (and
// testable) at modest sizes — kMaxGridBlocks * kDefaultBlockSize is ~1M
// elements — and can be retuned freely without changing any result.
inline constexpr unsigned int kMaxGridBlocks = 4096;

struct LaunchConfig {
  unsigned int grid;
  unsigned int block;
};

// Grid/block for a 1-D grid-stride kernel over `n` elements. Requires
// n > 0 (CHECK): a grid of 0 is a CUDA launch error, and the §9.2 launcher
// contract early-returns OK on numel() == 0 before computing a config.
[[nodiscard]] constexpr LaunchConfig LaunchConfig1D(std::int64_t n) {
  CHECK(n > 0, "LaunchConfig1D requires n > 0; got {}", n);
  const std::int64_t blocks = ((n - 1) / kDefaultBlockSize) + 1;
  const std::int64_t grid = std::min<std::int64_t>(blocks, kMaxGridBlocks);
  return {.grid = static_cast<unsigned int>(grid), .block = kDefaultBlockSize};
}

}  // namespace engine::kernels

// Grid-stride loop: correct for any n ≥ 0 with any launch config;
// performance-tuning the config never changes semantics (design §9.1).
// Expand only inside __global__ functions.
//
// NOLINTBEGIN(bugprone-macro-parentheses): `i` names the loop variable it
// declares — a declarator cannot be parenthesized.
#define CUDA_1D_KERNEL_LOOP(i, n)                                       \
  for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += static_cast<std::int64_t>(blockDim.x) * gridDim.x)
// NOLINTEND(bugprone-macro-parentheses)
