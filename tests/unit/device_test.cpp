#include "tensor/device.h"

#include "core/status.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>

namespace {

using engine::core::IsInvalidArgument;
using engine::core::StatusOr;
using engine::tensor::Device;
using engine::tensor::DeviceType;
using engine::tensor::to_string;

// Enum values are stable — they reach fixtures and serialized test data
// (device.h) — so renumbering must be a loud, deliberate change (mirrors
// dtype_test's stable-value block).
static_assert(static_cast<std::uint8_t>(DeviceType::kCPU) == 0);
static_assert(static_cast<std::uint8_t>(DeviceType::kCUDA) == 1);

// The type is constexpr-usable end to end (except Parse/ToString, which
// build strings); pin the core contracts at compile time (mirrors
// dtype_test's static_assert block).
static_assert(Device{}.type() == DeviceType::kCPU);
static_assert(Device{}.index() == 0);
static_assert(Device{} == Device::Cpu());
static_assert(Device::Cpu().is_cpu());
static_assert(!Device::Cpu().is_cuda());
static_assert(Device::Cuda(3).is_cuda());
static_assert(!Device::Cuda(3).is_cpu());
static_assert(Device::Cuda(3).type() == DeviceType::kCUDA);
static_assert(Device::Cuda(3).index() == 3);
static_assert(Device::Cuda(0) != Device::Cuda(1));
static_assert(Device::Cpu() != Device::Cuda(0));
static_assert(to_string(DeviceType::kCPU) == "cpu");
static_assert(to_string(DeviceType::kCUDA) == "cuda");

// Golden table: device ↔ canonical spec. Round-trip tests below walk it.
struct DeviceGolden {
  Device device;
  std::string_view canonical;
};

constexpr std::array<DeviceGolden, 4> kGolden = {{
    {.device = Device::Cpu(), .canonical = "cpu"},
    {.device = Device::Cuda(0), .canonical = "cuda:0"},
    {.device = Device::Cuda(7), .canonical = "cuda:7"},
    {.device = Device::Cuda(12), .canonical = "cuda:12"},
}};

TEST(DeviceTest, EqualityComparesTypeAndIndex) {
  EXPECT_EQ(Device::Cpu(), Device::Cpu());
  EXPECT_EQ(Device::Cuda(2), Device::Cuda(2));
  EXPECT_NE(Device::Cuda(0), Device::Cuda(1));
  EXPECT_NE(Device::Cpu(), Device::Cuda(0));
  EXPECT_EQ(Device{}, Device::Cpu());
}

TEST(DeviceTest, ToStringProducesCanonicalSpec) {
  for (const DeviceGolden& golden : kGolden) {
    EXPECT_EQ(golden.device.ToString(), golden.canonical);
  }
}

TEST(DeviceTest, FmtFormattingMatchesToString) {
  for (const DeviceGolden& golden : kGolden) {
    EXPECT_EQ(fmt::format("{}", golden.device), golden.canonical);
  }
  EXPECT_EQ(fmt::format("{}", DeviceType::kCPU), "cpu");
  EXPECT_EQ(fmt::format("{}", DeviceType::kCUDA), "cuda");
}

TEST(DeviceTest, ParseAcceptsCanonicalSpecs) {
  for (const DeviceGolden& golden : kGolden) {
    const StatusOr<Device> parsed = Device::Parse(golden.canonical);
    ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
    EXPECT_EQ(*parsed, golden.device);
  }
}

TEST(DeviceTest, ParseBareCudaMeansCudaZero) {
  const StatusOr<Device> parsed = Device::Parse("cuda");
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  EXPECT_EQ(*parsed, Device::Cuda(0));
}

TEST(DeviceTest, ParseToleratesLeadingZeros) {
  const StatusOr<Device> parsed = Device::Parse("cuda:07");
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  EXPECT_EQ(*parsed, Device::Cuda(7));
}

TEST(DeviceTest, ParseAcceptsIntMaxIndex) {
  // The exact from_chars boundary: INT_MAX parses, INT_MAX + 1 (below) does
  // not.
  const StatusOr<Device> parsed = Device::Parse("cuda:2147483647");
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  EXPECT_EQ(*parsed, Device::Cuda(2147483647));
}

TEST(DeviceTest, ParseRoundTripsToString) {
  for (const DeviceGolden& golden : kGolden) {
    const StatusOr<Device> parsed = Device::Parse(golden.device.ToString());
    ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
    EXPECT_EQ(*parsed, golden.device);
  }
}

TEST(DeviceTest, ParseRejectsInvalidSpecs) {
  constexpr std::array<std::string_view, 18> kInvalid = {
      "",
      "cpu:0",
      "cpu:1",
      "gpu",
      "CPU",
      "CUDA",
      "Cuda:0",
      " cpu",
      "cpu ",
      "cuda ",
      "cuda:",
      "cuda: 1",
      "cuda:-1",
      "cuda:+1",
      "cuda:1x",
      "cuda:abc",
      "cuda:2147483648",   // INT_MAX + 1: one past the exact boundary
      "cuda:99999999999",  // exceeds int range
  };
  for (const std::string_view spec : kInvalid) {
    const StatusOr<Device> parsed = Device::Parse(spec);
    ASSERT_FALSE(parsed.ok()) << "spec \"" << spec << "\" should not parse";
    EXPECT_TRUE(IsInvalidArgument(parsed.status()))
        << "spec \"" << spec << "\": " << parsed.status().ToString();
    // The offending spec is quoted in the message.
    EXPECT_NE(parsed.status().message().find(spec), std::string_view::npos);
  }
}

// --- Death tests (invariants CHECKed at construction; design §5) ---

TEST(DeviceDeathTest, CpuWithNonzeroIndexAborts) {
  EXPECT_DEATH((void)Device(DeviceType::kCPU, 1), "cpu device index must be 0");
}

TEST(DeviceDeathTest, NegativeIndexAborts) {
  EXPECT_DEATH((void)Device::Cuda(-1), "negative device index");
}

TEST(DeviceDeathTest, ToStringOnInvalidDeviceTypeAborts) {
  EXPECT_DEATH((void)to_string(static_cast<DeviceType>(9)),
               "invalid DeviceType value 9");
}

}  // namespace
