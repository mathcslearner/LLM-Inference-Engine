#pragma once

#include "core/check.h"
#include "cuda/cuda_utils.h"
#include "cuda/stream.h"
#include "cuda/stream_handle.h"
#include "memory/allocator.h"
#include "memory/caching_allocator.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <optional>

// GPU-test infrastructure (M2-T03/M2-T09; design:
// docs/design/cuda-backend.md §10). GPU tests skip — never fail — on
// machines without a CUDA device, so CPU-only CI stays green. Everything
// here goes through toolkit-free public headers, so this TU compiles on
// every configuration; on CPU-only builds the fixture always skips before
// touching a stub.
//
// Naming contract (tests/CMakeLists.txt): test suites that need a device —
// in practice, every fixture derived from CudaTestFixture — are named
// `<Thing>GpuTest` / `<Thing>GpuDeathTest`. CTest discovery labels exactly
// those suites `<tree>-gpu`, which both `ctest -L gpu` and `ctest -L <tree>`
// match (-L is a regex). See tests/README.md for the kernel-testing pattern.

namespace engine::testing {

// True iff CUDA work is possible: built with CUDA and device_count() > 0.
[[nodiscard]] bool HasCudaDevice();

// Base fixture for GPU tests (design §10.2). SetUp skips without a device
// (so every derived test skips, never fails), makes device 0 current for
// the whole test via ScopedSetDevice, and provides a per-test CudaStream
// plus a per-test CachingAllocator over DefaultCudaAllocator(0). TearDown
// synchronizes the stream first — surfacing deferred execution errors
// (§4.3) and discharging the §6.4/§7.4 rule that no device work is in
// flight when the pool is destroyed (the pool's destructor CHECKs that no
// Buffers are live, so tests must not leak tensors past the test body).
//
// Tests that construct their own streams/allocators (because those are the
// subject under test) still derive from this fixture for the skip guard and
// device selection, and simply ignore the provided members.
class CudaTestFixture : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // The per-test stream. Only callable after SetUp passed the skip guard —
  // i.e. from test bodies and helpers, not from a skipped test's TearDown.
  [[nodiscard]] cuda::CudaStream& stream() {
    CHECK(stream_.has_value(), "stream() before CudaTestFixture::SetUp");
    return *stream_;
  }
  [[nodiscard]] cuda::StreamHandle stream_handle() { return stream().handle(); }

  // The per-test caching pool over the device-0 default allocator, for
  // passing to Tensor::empty. Same precondition as stream().
  [[nodiscard]] memory::Allocator* allocator() {
    CHECK(allocator_.has_value(), "allocator() before CudaTestFixture::SetUp");
    return &*allocator_;
  }

 private:
  // Engaged only when a device is present; construction order is the
  // teardown order in reverse (guard first, pool last).
  std::optional<cuda::ScopedSetDevice> device_guard_;
  std::optional<cuda::CudaStream> stream_;
  std::optional<memory::CachingAllocator> allocator_;
};

// Synchronizes `stream` (surfacing deferred execution errors, §4.3), copies
// `actual` to the host, and compares against the CPU-resident `expected`
// with ops::allclose — failure messages carry allclose's worst-mismatch
// Summary(). Tolerances default per-dtype (ops.h §10 stance: accepting the
// default is still stating one); pass rtol/atol to override, e.g. {0.0,
// 0.0} for bit-exact kernels. `actual` may live on any device (a CPU
// tensor skips the copy); both tensors must be defined, same shape, same
// dtype, and `actual` contiguous. A free function rather than a fixture
// member so non-fixture code (e.g. .cu TUs) can use it with any stream —
// including cuda::DefaultStream(i) for null-handle launch tests.
void ExpectTensorsClose(const tensor::Tensor& actual,
                        const tensor::Tensor& expected,
                        cuda::CudaStream& stream,
                        std::optional<double> rtol = std::nullopt,
                        std::optional<double> atol = std::nullopt);

}  // namespace engine::testing

// In a fixture SetUp or test body: skip the test on machines where GPU work
// is impossible (no device, no driver, or a CPU-only build).
//
// Deliberately a bare `if` (the usual gtest macro shape): GTEST_SKIP's
// `return` must run in the *caller's* frame, so a do-while(0) wrapper
// would not change behavior — but as with all such macros, don't use it
// as the `if` branch of an outer if/else (the `else` would mis-bind).
#define ENGINE_SKIP_WITHOUT_CUDA()              \
  if (!::engine::testing::HasCudaDevice()) {    \
    GTEST_SKIP() << "no CUDA device available"; \
  }
