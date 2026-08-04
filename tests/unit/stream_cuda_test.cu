#include "common/cuda.h"
#include "core/status.h"
#include "cuda/cuda_check.h"
#include "cuda/stream.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <utility>

// Stream/event tests that need a real asynchronous workload (M2-T04
// acceptance criteria): elapsed time of a known-duration kernel, event
// ordering across two streams via cudaStreamWaitEvent, and destructor
// draining. M2-T08's kernels do not exist yet, so this TU brings its own
// device-side delay kernel — which is why it is a .cu file, buildable only
// on CUDA builds (design §3 allows .cu under tests/). The toolkit-free
// surface is covered for every configuration in stream_test.cpp.

namespace {

using engine::core::IsFailedPrecondition;
using engine::core::StatusOr;
using engine::cuda::CudaEvent;
using engine::cuda::CudaStream;

// Busy-spins for `cycles` device clock cycles, then writes `value` to
// `out`. At the ~1–2 GHz clocks of any supported GPU, 20M cycles is
// ~10–20 ms — long enough that the host observably outruns the device, far
// below any watchdog limit.
__global__ void DelayThenWrite(long long cycles, int value, int* out) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
  *out = value;
}

constexpr long long kDelayCycles = 20'000'000;

// Device-side int the delay kernel writes to, freed on scope exit.
class DeviceInt {
 public:
  DeviceInt() {
    CUDA_CHECK(cudaMalloc(&ptr_, sizeof(int)));
    CUDA_CHECK(cudaMemset(ptr_, 0, sizeof(int)));
  }
  ~DeviceInt() { CUDA_CHECK(cudaFree(ptr_)); }
  DeviceInt(const DeviceInt&) = delete;
  DeviceInt& operator=(const DeviceInt&) = delete;
  DeviceInt(DeviceInt&&) = delete;
  DeviceInt& operator=(DeviceInt&&) = delete;

  [[nodiscard]] int* get() const { return ptr_; }

  // Synchronous readback (pageable destination ⇒ implicitly synchronizing,
  // design §8.2 — deliberate in these tests).
  [[nodiscard]] int Read() const {
    int value = -1;
    CUDA_CHECK(cudaMemcpy(&value, ptr_, sizeof(int), cudaMemcpyDeviceToHost));
    return value;
  }

 private:
  int* ptr_ = nullptr;
};

[[nodiscard]] CudaStream MakeStream() {
  StatusOr<CudaStream> stream = CudaStream::Create();
  CHECK_OK(stream.status());
  return *std::move(stream);
}

[[nodiscard]] CudaEvent MakeEvent(StatusOr<CudaEvent> created) {
  CHECK_OK(created.status());
  return *std::move(created);
}

// GPU tests on CudaTestFixture (M2-T09): the fixture supplies the skip
// guard and device selection; streams/events under test are built locally.
class CudaEventTimingGpuTest : public engine::testing::CudaTestFixture {};
class CudaStreamOrderingGpuTest : public engine::testing::CudaTestFixture {};

// Elapsed time of a known-duration workload is strictly positive.
TEST_F(CudaEventTimingGpuTest, ElapsedMsOfDelayKernelIsPositive) {
  CudaStream stream = MakeStream();
  CudaEvent start = MakeEvent(CudaEvent::Timing());
  CudaEvent stop = MakeEvent(CudaEvent::Timing());
  const DeviceInt flag;

  ASSERT_TRUE(start.Record(stream).ok());
  DelayThenWrite<<<1, 1, 0, stream.get()>>>(kDelayCycles, 1, flag.get());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);  // launch check, design §4.3
  ASSERT_TRUE(stop.Record(stream).ok());

  ASSERT_TRUE(stop.Synchronize().ok());
  const StatusOr<float> elapsed = CudaEvent::ElapsedMs(start, stop);
  ASSERT_TRUE(elapsed.ok()) << elapsed.status();
  EXPECT_GT(*elapsed, 0.0F);
}

// ElapsedMs on a still-running record is FailedPrecondition, not an opaque
// mapped cudaErrorNotReady.
TEST_F(CudaEventTimingGpuTest, ElapsedMsBeforeCompletionIsFailedPrecondition) {
  CudaStream stream = MakeStream();
  CudaEvent start = MakeEvent(CudaEvent::Timing());
  CudaEvent stop = MakeEvent(CudaEvent::Timing());
  const DeviceInt flag;

  ASSERT_TRUE(start.Record(stream).ok());
  DelayThenWrite<<<1, 1, 0, stream.get()>>>(kDelayCycles, 1, flag.get());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_TRUE(stop.Record(stream).ok());

  // The delay kernel runs ~10+ ms; the host reaches this call in far less.
  const StatusOr<float> elapsed = CudaEvent::ElapsedMs(start, stop);
  ASSERT_FALSE(elapsed.ok());
  EXPECT_TRUE(IsFailedPrecondition(elapsed.status())) << elapsed.status();
  ASSERT_TRUE(stream.Synchronize().ok());
}

// Cross-stream ordering (the acceptance criterion): stream B's work must
// not start until the event recorded after stream A's delay kernel has
// completed, so B's readback observes the kernel's write.
TEST_F(CudaStreamOrderingGpuTest, WaitEventOrdersAcrossStreams) {
  CudaStream stream_a = MakeStream();
  CudaStream stream_b = MakeStream();
  CudaEvent done = MakeEvent(CudaEvent::Sync());
  const DeviceInt flag;

  DelayThenWrite<<<1, 1, 0, stream_a.get()>>>(kDelayCycles, 42, flag.get());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_TRUE(done.Record(stream_a).ok());

  ASSERT_TRUE(stream_b.WaitEvent(done).ok());
  int host_value = -1;
  CUDA_CHECK(cudaMemcpyAsync(&host_value, flag.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream_b.get()));
  ASSERT_TRUE(stream_b.Synchronize().ok());
  EXPECT_EQ(host_value, 42);
}

// Destruction-order safety (the acceptance criterion): the destructor
// drains, so a stream destroyed with a kernel in flight finishes that
// kernel first — its write is visible immediately afterwards.
TEST_F(CudaStreamOrderingGpuTest, DestructorDrainsInFlightWork) {
  const DeviceInt flag;
  {
    CudaStream stream = MakeStream();
    DelayThenWrite<<<1, 1, 0, stream.get()>>>(kDelayCycles, 7, flag.get());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }  // draining destructor
  EXPECT_EQ(flag.Read(), 7);
}

}  // namespace
