#include "cuda/cuda_check.h"

#include "common/cuda.h"
#include "core/status.h"
#include "cuda/cuda_utils.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <string>
#include <string_view>

// Tests for ToStatus and the CUDA error macros (M2-T03, design §4). This TU
// includes the CUDA toolkit, so it is registered only on CUDA builds
// (tests/unit/CMakeLists.txt); most of it needs no GPU — ToStatus and the
// macros are pure host code over cudaError_t values. Tests that call into
// the CUDA runtime and assert specific error codes guard with the shared
// skip predicate, since driverless machines report different errors.

namespace {

using engine::core::Status;
using engine::core::StatusCode;
using engine::cuda::ToStatus;
using engine::testing::HasCudaDevice;

[[nodiscard]] bool Contains(std::string_view haystack,
                            std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

// --- ToStatus: the §4.2 code table ---

TEST(ToStatusTest, CodeMappingTable) {
  const struct {
    cudaError_t err;
    StatusCode expected;
  } kTable[] = {
      {cudaErrorMemoryAllocation, StatusCode::kResourceExhausted},
      {cudaErrorInvalidDevice, StatusCode::kInvalidArgument},
      {cudaErrorInvalidValue, StatusCode::kInvalidArgument},
      {cudaErrorNoDevice, StatusCode::kUnavailable},
      {cudaErrorInsufficientDriver, StatusCode::kUnavailable},
      {cudaErrorNotSupported, StatusCode::kUnavailable},
      // "Everything else" → kInternal; two representatives.
      {cudaErrorIllegalAddress, StatusCode::kInternal},
      {cudaErrorUnknown, StatusCode::kInternal},
  };
  for (const auto& row : kTable) {
    const Status status = ToStatus(row.err, "what");
    EXPECT_EQ(status.code(), row.expected) << status;
  }
}

TEST(ToStatusTest, MessageEmbedsWhatNameAndDescription) {
  const Status status = ToStatus(cudaErrorMemoryAllocation, "allocating 4096");
  EXPECT_TRUE(Contains(status.message(), "allocating 4096")) << status;
  EXPECT_TRUE(
      Contains(status.message(), cudaGetErrorName(cudaErrorMemoryAllocation)))
      << status;
  EXPECT_TRUE(
      Contains(status.message(), cudaGetErrorString(cudaErrorMemoryAllocation)))
      << status;
}

TEST(ToStatusTest, StickyErrorsAreMarked) {
  EXPECT_TRUE(
      Contains(ToStatus(cudaErrorIllegalAddress, "what").message(), "sticky"));
  EXPECT_TRUE(
      Contains(ToStatus(cudaErrorLaunchFailure, "what").message(), "sticky"));
  EXPECT_FALSE(
      Contains(ToStatus(cudaErrorInvalidValue, "what").message(), "sticky"));
}

// --- CUDA_RETURN_IF_ERROR ---

// The macro requires an enclosing Status-returning function.
[[nodiscard]] Status ReturnIfError(cudaError_t err, int& evaluations) {
  auto evaluate = [&evaluations, err] {
    ++evaluations;
    return err;
  };
  CUDA_RETURN_IF_ERROR(evaluate());
  return engine::core::OkStatus();
}

TEST(CudaReturnIfErrorTest, SuccessPassesThroughEvaluatingOnce) {
  int evaluations = 0;
  EXPECT_TRUE(ReturnIfError(cudaSuccess, evaluations).ok());
  EXPECT_EQ(evaluations, 1);
}

TEST(CudaReturnIfErrorTest, FailureReturnsStatusEvaluatingOnce) {
  int evaluations = 0;
  const Status status = ReturnIfError(cudaErrorInvalidValue, evaluations);
  EXPECT_EQ(evaluations, 1);
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
  // The message carries the failing expression and its source location.
  EXPECT_TRUE(Contains(status.message(), "evaluate()")) << status;
  EXPECT_TRUE(Contains(status.message(), "cuda_check_test.cpp")) << status;
  EXPECT_TRUE(
      Contains(status.message(), cudaGetErrorString(cudaErrorInvalidValue)))
      << status;
}

// The M2-T03 acceptance criterion: a deliberately bad runtime call surfaces
// a Status with the CUDA error string embedded.
[[nodiscard]] Status SetAbsurdDevice() {
  CUDA_RETURN_IF_ERROR(cudaSetDevice(1000000));
  return engine::core::OkStatus();
}

TEST(CudaReturnIfErrorTest, BadRuntimeCallSurfacesStatus) {
  const Status status = SetAbsurdDevice();
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(Contains(status.message(), "cudaSetDevice(1000000)")) << status;
  EXPECT_TRUE(Contains(status.message(), "cudaError")) << status;
}

TEST(CudaReturnIfErrorTest, BadRuntimeCallErrorCodeWithDevice) {
  // Driverless machines report cudaErrorNoDevice/InsufficientDriver instead
  // of cudaErrorInvalidDevice; only assert the specific error with a GPU.
  ENGINE_SKIP_WITHOUT_CUDA();
  const Status status = SetAbsurdDevice();
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(
      Contains(status.message(), cudaGetErrorName(cudaErrorInvalidDevice)))
      << status;
}

// --- CUDA_CHECK ---

TEST(CudaCheckTest, SuccessEvaluatesOnceAndContinues) {
  int evaluations = 0;
  auto evaluate = [&evaluations] {
    ++evaluations;
    return cudaSuccess;
  };
  CUDA_CHECK(evaluate());
  EXPECT_EQ(evaluations, 1);
}

TEST(CudaCheckDeathTest, FailureAborts) {
  // A constant error expression: the forked death-test child never touches
  // a CUDA context.
  const cudaError_t err = cudaErrorInvalidValue;
  EXPECT_DEATH(CUDA_CHECK(err), "CUDA_CHECK\\(err\\)");
  EXPECT_DEATH(CUDA_CHECK(err), "cudaErrorInvalidValue");
}

// --- ScopedSetDevice: restore semantics (needs cudaGetDevice) ---

TEST(ScopedSetDeviceTest, RestoresPreviousDevice) {
  ENGINE_SKIP_WITHOUT_CUDA();
  const int last = engine::cuda::device_count() - 1;
  CUDA_CHECK(cudaSetDevice(0));
  {
    const engine::cuda::ScopedSetDevice guard(last);
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    EXPECT_EQ(current, last);
  }
  int current = -1;
  CUDA_CHECK(cudaGetDevice(&current));
  EXPECT_EQ(current, 0);
}

}  // namespace
