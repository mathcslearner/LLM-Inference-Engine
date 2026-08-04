#include "cuda/stream.h"

#include "common/cuda.h"
#include "core/status.h"
#include "cuda/cuda_utils.h"
#include "cuda/stream_handle.h"

#include <gtest/gtest.h>

#include <array>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

// Tests for the toolkit-free stream/event API (M2-T04). This TU builds on
// every configuration: CUDA builds exercise the real implementation
// (GPU-dependent assertions skip without a device, design §10.1); CPU-only
// builds exercise the stub. Tests that need toolkit calls or launch a
// kernel — real async workloads for elapsed time, cross-stream ordering,
// and destructor draining — live in stream_cuda_test.cu (CUDA builds only).

namespace {

using engine::core::IsFailedPrecondition;
using engine::core::IsInvalidArgument;
using engine::core::StatusOr;
using engine::cuda::CudaEvent;
using engine::cuda::CudaStream;
using engine::cuda::DefaultStream;
using engine::cuda::device_count;
using engine::cuda::StreamHandle;

// --- StreamHandle (build-independent) ---

static_assert(std::is_trivially_copyable_v<StreamHandle>);
static_assert(StreamHandle{}.get() == nullptr);
static_assert(StreamHandle{}.is_default());

TEST(StreamHandleTest, DefaultIsNull) {
  const StreamHandle handle;
  EXPECT_EQ(handle.get(), nullptr);
  EXPECT_TRUE(handle.is_default());
}

// --- CPU-only builds: the stub taxonomy (design §2.2) ---

#ifndef ENGINE_ENABLE_CUDA

void ExpectUnimplementedWithoutCuda(const engine::core::Status& status) {
  EXPECT_TRUE(engine::core::IsUnimplemented(status)) << status;
  EXPECT_NE(status.message().find("without CUDA"), std::string_view::npos)
      << status;
}

TEST(StreamStubTest, StreamCreateIsUnimplemented) {
  ExpectUnimplementedWithoutCuda(CudaStream::Create().status());
}

TEST(StreamStubTest, EventFactoriesAreUnimplemented) {
  ExpectUnimplementedWithoutCuda(CudaEvent::Timing().status());
  ExpectUnimplementedWithoutCuda(CudaEvent::Sync().status());
}

#endif  // ENGINE_ENABLE_CUDA

// Dies on every configuration: index == device_count() is out of range on a
// GPU machine, out of range of the empty [0, 0) on a toolkit-no-GPU
// machine, and the stub CHECKs unconditionally on CPU-only builds.
// "DefaultStream" appears in the CHECK output on all three paths.
TEST(DefaultStreamDeathTest, BadIndexChecks) {
  EXPECT_DEATH(static_cast<void>(DefaultStream(device_count())),
               "DefaultStream");
}

// --- GPU tests (toolkit-free surface; skip without a device) ---
//
// On CudaTestFixture (M2-T09) for the skip guard and device selection; the
// streams/events under test are built locally, and the fixture's own stream
// is deliberately unused.

class CudaStreamGpuTest : public engine::testing::CudaTestFixture {};
class CudaStreamGpuDeathTest : public engine::testing::CudaTestFixture {};
class CudaEventGpuTest : public engine::testing::CudaTestFixture {};
class CudaEventGpuDeathTest : public engine::testing::CudaTestFixture {};
class DefaultStreamGpuTest : public engine::testing::CudaTestFixture {};

TEST_F(CudaStreamGpuTest, CreateSynchronizeEmpty) {
  StatusOr<CudaStream> stream = CudaStream::Create();
  ASSERT_TRUE(stream.ok()) << stream.status();
  EXPECT_NE(stream->get(), nullptr);
  EXPECT_FALSE(stream->handle().is_default());
  EXPECT_EQ(stream->handle().get(), stream->get());
  EXPECT_GE(stream->device_index(), 0);
  EXPECT_TRUE(stream->Synchronize().ok());
}

TEST_F(CudaStreamGpuTest, MoveTransfersOwnership) {
  StatusOr<CudaStream> created = CudaStream::Create();
  ASSERT_TRUE(created.ok()) << created.status();
  CudaStream source = *std::move(created);
  const auto* raw = source.get();

  CudaStream via_ctor = std::move(source);
  EXPECT_EQ(via_ctor.get(), raw);

  // Move assignment drains and destroys the target's own stream, then
  // steals the source's handle.
  StatusOr<CudaStream> other = CudaStream::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  CudaStream via_assign = *std::move(other);
  via_assign = std::move(via_ctor);
  EXPECT_EQ(via_assign.get(), raw);
  EXPECT_TRUE(via_assign.Synchronize().ok());
}

TEST_F(CudaStreamGpuDeathTest, MovedFromMembersCheck) {
  StatusOr<CudaStream> created = CudaStream::Create();
  ASSERT_TRUE(created.ok()) << created.status();
  CudaStream source = *std::move(created);
  const CudaStream target = std::move(source);
  // Using the moved-from stream is deliberate: every member CHECKs
  // engagement (design §6.2), which is exactly what dies here.
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_DEATH(static_cast<void>(source.get()), "moved-from CudaStream");
  EXPECT_DEATH(static_cast<void>(source.Synchronize()),
               "moved-from CudaStream");
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_NE(target.get(), nullptr);
}

TEST_F(CudaEventGpuTest, RecordSynchronizeQuery) {
  StatusOr<CudaStream> stream = CudaStream::Create();
  ASSERT_TRUE(stream.ok()) << stream.status();
  StatusOr<CudaEvent> event = CudaEvent::Timing();
  ASSERT_TRUE(event.ok()) << event.status();
  EXPECT_TRUE(event->is_timing());

  ASSERT_TRUE(event->Record(*stream).ok());
  ASSERT_TRUE(event->Synchronize().ok());
  const StatusOr<bool> done = event->Query();
  ASSERT_TRUE(done.ok()) << done.status();
  EXPECT_TRUE(*done);
}

TEST_F(CudaEventGpuTest, UnrecordedEventIsComplete) {
  StatusOr<CudaEvent> event = CudaEvent::Sync();
  ASSERT_TRUE(event.ok()) << event.status();
  EXPECT_FALSE(event->is_timing());
  // CUDA semantics, documented on the members: a never-recorded event is
  // complete — Query() true, Synchronize() immediate success — and waiting
  // on it is a no-op.
  const StatusOr<bool> done = event->Query();
  ASSERT_TRUE(done.ok()) << done.status();
  EXPECT_TRUE(*done);
  EXPECT_TRUE(event->Synchronize().ok());

  StatusOr<CudaStream> stream = CudaStream::Create();
  ASSERT_TRUE(stream.ok()) << stream.status();
  EXPECT_TRUE(stream->WaitEvent(*event).ok());
  EXPECT_TRUE(stream->Synchronize().ok());
}

TEST_F(CudaEventGpuTest, ElapsedMsOfCompletedPairIsNonNegative) {
  StatusOr<CudaStream> stream = CudaStream::Create();
  ASSERT_TRUE(stream.ok()) << stream.status();
  StatusOr<CudaEvent> start = CudaEvent::Timing();
  ASSERT_TRUE(start.ok()) << start.status();
  StatusOr<CudaEvent> stop = CudaEvent::Timing();
  ASSERT_TRUE(stop.ok()) << stop.status();

  ASSERT_TRUE(start->Record(*stream).ok());
  ASSERT_TRUE(stop->Record(*stream).ok());
  ASSERT_TRUE(stop->Synchronize().ok());
  const StatusOr<float> elapsed = CudaEvent::ElapsedMs(*start, *stop);
  ASSERT_TRUE(elapsed.ok()) << elapsed.status();
  EXPECT_GE(*elapsed, 0.0F);
}

TEST_F(CudaEventGpuTest, ElapsedMsRejectsSyncEvents) {
  StatusOr<CudaEvent> timing = CudaEvent::Timing();
  ASSERT_TRUE(timing.ok()) << timing.status();
  StatusOr<CudaEvent> sync = CudaEvent::Sync();
  ASSERT_TRUE(sync.ok()) << sync.status();

  const StatusOr<float> elapsed = CudaEvent::ElapsedMs(*timing, *sync);
  ASSERT_FALSE(elapsed.ok());
  EXPECT_TRUE(IsInvalidArgument(elapsed.status())) << elapsed.status();
}

// A never-recorded Timing() event counts as complete for Query() (CUDA
// semantics, tested above), so without its own pre-check it would sail
// through the completion check and make cudaEventElapsedTime fail with
// cudaErrorInvalidResourceHandle → opaque kInternal. The taxonomy demands
// FailedPrecondition instead.
TEST_F(CudaEventGpuTest, ElapsedMsOfNeverRecordedEventIsFailedPrecondition) {
  StatusOr<CudaEvent> start = CudaEvent::Timing();
  ASSERT_TRUE(start.ok()) << start.status();
  StatusOr<CudaEvent> stop = CudaEvent::Timing();
  ASSERT_TRUE(stop.ok()) << stop.status();

  const StatusOr<float> elapsed = CudaEvent::ElapsedMs(*start, *stop);
  ASSERT_FALSE(elapsed.ok());
  EXPECT_TRUE(IsFailedPrecondition(elapsed.status())) << elapsed.status();

  // Recording only one of the pair is still FailedPrecondition.
  StatusOr<CudaStream> stream = CudaStream::Create();
  ASSERT_TRUE(stream.ok()) << stream.status();
  ASSERT_TRUE(start->Record(*stream).ok());
  ASSERT_TRUE(start->Synchronize().ok());
  const StatusOr<float> half_recorded = CudaEvent::ElapsedMs(*start, *stop);
  ASSERT_FALSE(half_recorded.ok());
  EXPECT_TRUE(IsFailedPrecondition(half_recorded.status()))
      << half_recorded.status();
}

TEST_F(CudaEventGpuDeathTest, MovedFromMembersCheck) {
  StatusOr<CudaEvent> created = CudaEvent::Timing();
  ASSERT_TRUE(created.ok()) << created.status();
  CudaEvent source = *std::move(created);
  const CudaEvent target = std::move(source);
  // Using the moved-from event is deliberate: every member CHECKs
  // engagement (design §6.2), which is exactly what dies here.
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_DEATH(static_cast<void>(source.get()), "moved-from CudaEvent");
  EXPECT_DEATH(static_cast<void>(source.Query()), "moved-from CudaEvent");
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_TRUE(target.is_timing());
}

// Lifetime rule (design §6.2): an event may outlive the stream it was
// recorded on — completion state survives stream destruction, and the
// stream's draining destructor has already observed completion anyway.
TEST_F(CudaEventGpuTest, EventOutlivesItsStream) {
  StatusOr<CudaEvent> event = CudaEvent::Timing();
  ASSERT_TRUE(event.ok()) << event.status();
  {
    StatusOr<CudaStream> stream = CudaStream::Create();
    ASSERT_TRUE(stream.ok()) << stream.status();
    ASSERT_TRUE(event->Record(*stream).ok());
  }  // stream drained and destroyed
  const StatusOr<bool> done = event->Query();
  ASSERT_TRUE(done.ok()) << done.status();
  EXPECT_TRUE(*done);
  EXPECT_TRUE(event->Synchronize().ok());
}

// --- DefaultStream ---

TEST_F(DefaultStreamGpuTest, SameObjectEveryCall) {
  CudaStream& first = DefaultStream(0);
  EXPECT_NE(first.get(), nullptr);
  EXPECT_EQ(first.device_index(), 0);
  EXPECT_EQ(&DefaultStream(0), &first);
  EXPECT_TRUE(first.Synchronize().ok());
}

TEST_F(DefaultStreamGpuTest, ConcurrentCallsAgree) {
  constexpr int kThreads = 8;
  std::array<CudaStream*, kThreads> seen{};
  {
    std::array<std::thread, kThreads> threads;
    for (int i = 0; i < kThreads; ++i) {
      threads[static_cast<std::size_t>(i)] = std::thread([&seen, i] {
        seen[static_cast<std::size_t>(i)] = &DefaultStream(0);
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
  }
  for (const CudaStream* stream : seen) {
    EXPECT_EQ(stream, seen[0]);
  }
}

}  // namespace
