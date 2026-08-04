#include "cuda/cuda_utils.h"

#include "common/cuda.h"
#include "core/status.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

// Tests for the toolkit-free device-introspection API (M2-T03). This TU
// builds on every configuration: CUDA builds exercise the real
// implementation (GPU-dependent assertions skip without a device via the
// shared predicate, design §10.1); CPU-only builds exercise the stub. The
// CUDA error macros need the toolkit and are tested in cuda_check_test.cpp
// (CUDA builds only).

namespace {

// engine::core::IsInvalidArgument / IsUnimplemented are used fully
// qualified below: each appears in only one branch of the build-dependent
// #ifdef, so a using-declaration would be flagged unused in the other build.
using engine::core::StatusOr;
using engine::cuda::device_count;
using engine::cuda::DeviceProperties;
using engine::cuda::GetDeviceProperties;
using engine::cuda::ScopedSetDevice;
using engine::testing::HasCudaDevice;

// --- device_count ---

TEST(DeviceCountTest, NonNegativeAndStable) {
  const int first = device_count();
  EXPECT_GE(first, 0);
  // Memoized (design §5.1): repeated calls agree.
  EXPECT_EQ(device_count(), first);
  EXPECT_EQ(device_count(), first);
}

TEST(DeviceCountTest, AgreesWithSkipPredicate) {
  EXPECT_EQ(HasCudaDevice(), device_count() > 0);
}

#ifndef ENGINE_ENABLE_CUDA
TEST(DeviceCountTest, ZeroOnCpuOnlyBuild) { EXPECT_EQ(device_count(), 0); }
#endif

// --- GetDeviceProperties ---

// Out-of-range indices fail with the code the design assigns to each build:
// InvalidArgument naming index and count on CUDA builds (§5.1),
// Unimplemented naming the missing CUDA support on CPU-only builds (§2.2).
TEST(GetDevicePropertiesTest, OutOfRangeIndexFails) {
  for (const int index : {-1, device_count()}) {
    const StatusOr<DeviceProperties> result = GetDeviceProperties(index);
    ASSERT_FALSE(result.ok()) << "index " << index;
#ifdef ENGINE_ENABLE_CUDA
    EXPECT_TRUE(engine::core::IsInvalidArgument(result.status()))
        << result.status();
    const std::string message(result.status().message());
    EXPECT_NE(message.find(std::to_string(index)), std::string::npos)
        << message;
    EXPECT_NE(message.find(std::to_string(device_count())), std::string::npos)
        << message;
#else
    EXPECT_TRUE(engine::core::IsUnimplemented(result.status()))
        << result.status();
    EXPECT_NE(result.status().message().find("without CUDA"),
              std::string_view::npos)
        << result.status();
#endif
  }
}

TEST(GetDevicePropertiesTest, Device0PropertiesAreSane) {
  ENGINE_SKIP_WITHOUT_CUDA();
  const StatusOr<DeviceProperties> properties = GetDeviceProperties(0);
  ASSERT_TRUE(properties.ok()) << properties.status();
  EXPECT_FALSE(properties->name.empty());
  // Any CUDA 12-era GPU: CC 5.0 (Maxwell) through current.
  EXPECT_GE(properties->compute_capability_major, 5);
  EXPECT_GE(properties->compute_capability_minor, 0);
  EXPECT_GT(properties->multiprocessor_count, 0);
  EXPECT_GT(properties->total_memory_bytes, 0U);
}

TEST(GetDevicePropertiesTest, EveryVisibleDeviceHasProperties) {
  ENGINE_SKIP_WITHOUT_CUDA();
  for (int i = 0; i < device_count(); ++i) {
    const StatusOr<DeviceProperties> properties = GetDeviceProperties(i);
    EXPECT_TRUE(properties.ok())
        << "device " << i << ": " << properties.status();
  }
}

// --- ScopedSetDevice ---

TEST(ScopedSetDeviceTest, ValidIndexScopeRuns) {
  ENGINE_SKIP_WITHOUT_CUDA();
  const ScopedSetDevice guard(0);
  // Restoration of the previous device is verified with cudaGetDevice in
  // cuda_check_test.cpp (needs the toolkit).
}

// One spelling that dies on every configuration: index == device_count() is
// out of range on a GPU machine, index 0 with no device fails cudaSetDevice
// on a toolkit-no-GPU machine, and the stub CHECKs unconditionally on
// CPU-only builds. "ScopedSetDevice" appears in the CHECK output as the
// failing function on all three paths.
TEST(ScopedSetDeviceDeathTest, BadIndexChecks) {
  EXPECT_DEATH(const ScopedSetDevice guard(device_count()), "ScopedSetDevice");
}

}  // namespace
