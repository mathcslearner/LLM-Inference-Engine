#pragma once

#include <gtest/gtest.h>

// GPU-test skip infrastructure (M2-T03; design:
// docs/design/cuda-backend.md §10.1). GPU tests skip — never fail — on
// machines without a CUDA device, so CPU-only CI stays green. The
// CudaTestFixture and expect_tensors_close arrive in M2-T09.

namespace engine::testing {

// True iff CUDA work is possible: built with CUDA and device_count() > 0.
[[nodiscard]] bool HasCudaDevice();

}  // namespace engine::testing

// In a fixture SetUp or test body: skip the test on machines where GPU work
// is impossible (no device, no driver, or a CPU-only build).
#define ENGINE_SKIP_WITHOUT_CUDA()              \
  if (!::engine::testing::HasCudaDevice()) {    \
    GTEST_SKIP() << "no CUDA device available"; \
  }
