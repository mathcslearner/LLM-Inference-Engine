#include "tensor/dtype.h"

#include "core/status.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace {

using engine::core::IsInvalidArgument;
using engine::tensor::DataType;
using engine::tensor::dtype_of;
using engine::tensor::from_string;
using engine::tensor::is_floating_point;
using engine::tensor::is_integral;
using engine::tensor::is_sub_byte;
using engine::tensor::itemsize;
using engine::tensor::itemsize_bits;
using engine::tensor::kAllDataTypes;
using engine::tensor::kNumDataTypes;
using engine::tensor::to_string;

// Enum values are stable — they reach fixtures and serialized test data
// (dtype.h) — so renumbering must be a loud, deliberate change.
static_assert(static_cast<std::uint8_t>(DataType::kFloat32) == 0);
static_assert(static_cast<std::uint8_t>(DataType::kFloat16) == 1);
static_assert(static_cast<std::uint8_t>(DataType::kBFloat16) == 2);
static_assert(static_cast<std::uint8_t>(DataType::kInt8) == 3);
static_assert(static_cast<std::uint8_t>(DataType::kUInt8) == 4);
static_assert(static_cast<std::uint8_t>(DataType::kInt32) == 5);
static_assert(static_cast<std::uint8_t>(DataType::kInt64) == 6);
static_assert(static_cast<std::uint8_t>(DataType::kBool) == 7);
static_assert(static_cast<std::uint8_t>(DataType::kFP8E4M3) == 8);
static_assert(static_cast<std::uint8_t>(DataType::kInt4) == 9);

// The count anchor (dtype.h ties kAllDataTypes to it; the golden table below
// is sized by it too).
static_assert(kNumDataTypes == 10);
static_assert(kAllDataTypes.size() == kNumDataTypes);

// The size queries and predicates are constexpr.
static_assert(itemsize_bits(DataType::kInt4) == 4);
static_assert(itemsize(DataType::kFloat32) == 4);
static_assert(is_sub_byte(DataType::kInt4));
static_assert(!is_sub_byte(DataType::kInt8));
static_assert(is_floating_point(DataType::kBFloat16));
static_assert(is_integral(DataType::kInt64));

// The trait mapping is a compile-time contract.
static_assert(dtype_of<float> == DataType::kFloat32);
static_assert(dtype_of<std::int8_t> == DataType::kInt8);
static_assert(dtype_of<std::uint8_t> == DataType::kUInt8);
static_assert(dtype_of<std::int32_t> == DataType::kInt32);
static_assert(dtype_of<std::int64_t> == DataType::kInt64);
static_assert(dtype_of<bool> == DataType::kBool);

// One golden row per DataType; every exhaustive test below walks this table,
// and CoversEveryDataTypeExactlyOnce pins it against kAllDataTypes.
struct DTypeGolden {
  DataType dtype;
  int bits;
  std::string_view name;
  bool floating = false;
  bool integral = false;
};

constexpr std::array<DTypeGolden, 10> kGolden = {{
    {.dtype = DataType::kFloat32,
     .bits = 32,
     .name = "float32",
     .floating = true},
    {.dtype = DataType::kFloat16,
     .bits = 16,
     .name = "float16",
     .floating = true},
    {.dtype = DataType::kBFloat16,
     .bits = 16,
     .name = "bfloat16",
     .floating = true},
    {.dtype = DataType::kInt8, .bits = 8, .name = "int8", .integral = true},
    {.dtype = DataType::kUInt8, .bits = 8, .name = "uint8", .integral = true},
    {.dtype = DataType::kInt32, .bits = 32, .name = "int32", .integral = true},
    {.dtype = DataType::kInt64, .bits = 64, .name = "int64", .integral = true},
    {.dtype = DataType::kBool, .bits = 8, .name = "bool"},
    {.dtype = DataType::kFP8E4M3,
     .bits = 8,
     .name = "fp8_e4m3",
     .floating = true},
    {.dtype = DataType::kInt4, .bits = 4, .name = "int4", .integral = true},
}};

TEST(DTypeTest, CoversEveryDataTypeExactlyOnce) {
  ASSERT_EQ(kGolden.size(), kAllDataTypes.size());
  for (std::size_t i = 0; i < kGolden.size(); ++i) {
    EXPECT_EQ(kGolden[i].dtype, kAllDataTypes[i]);
    for (std::size_t j = i + 1; j < kGolden.size(); ++j) {
      EXPECT_NE(kAllDataTypes[i], kAllDataTypes[j]);
    }
  }
}

TEST(DTypeTest, ItemsizeBitsGolden) {
  for (const DTypeGolden& g : kGolden) {
    EXPECT_EQ(itemsize_bits(g.dtype), g.bits) << to_string(g.dtype);
  }
}

TEST(DTypeTest, ItemsizeBytesGolden) {
  for (const DTypeGolden& g : kGolden) {
    if (is_sub_byte(g.dtype)) {
      continue;  // itemsize() is a CHECK failure; covered by the death test.
    }
    EXPECT_EQ(itemsize(g.dtype), g.bits / 8) << to_string(g.dtype);
  }
}

TEST(DTypeDeathTest, ItemsizeOnSubByteDtypeAborts) {
  EXPECT_DEATH((void)itemsize(DataType::kInt4),
               "itemsize\\(\\) is undefined for sub-byte dtypes");
}

TEST(DTypeDeathTest, ItemsizeBitsOnInvalidValueAborts) {
  EXPECT_DEATH((void)itemsize_bits(static_cast<DataType>(42)),
               "invalid DataType value 42");
}

TEST(DTypeDeathTest, ToStringOnInvalidValueAborts) {
  EXPECT_DEATH((void)to_string(static_cast<DataType>(42)),
               "invalid DataType value 42");
}

TEST(DTypeTest, ToStringGolden) {
  for (const DTypeGolden& g : kGolden) {
    EXPECT_EQ(to_string(g.dtype), g.name);
  }
}

TEST(DTypeTest, NameRoundTripsThroughFromString) {
  for (const DTypeGolden& g : kGolden) {
    auto parsed = from_string(to_string(g.dtype));
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, g.dtype);
  }
}

TEST(DTypeTest, FromStringRejectsUnknownNames) {
  for (const std::string_view name :
       {"", "float64", "Float32", " float32", "int 4", "fp8", "half"}) {
    auto parsed = from_string(name);
    ASSERT_FALSE(parsed.ok()) << name;
    EXPECT_TRUE(IsInvalidArgument(parsed.status())) << parsed.status();
  }
  // The offending name is quoted in the message.
  EXPECT_NE(from_string("float64").status().message().find("\"float64\""),
            std::string_view::npos);
}

TEST(DTypeTest, ClassificationGolden) {
  for (const DTypeGolden& g : kGolden) {
    EXPECT_EQ(is_floating_point(g.dtype), g.floating) << to_string(g.dtype);
    EXPECT_EQ(is_integral(g.dtype), g.integral) << to_string(g.dtype);
    EXPECT_EQ(is_sub_byte(g.dtype), g.dtype == DataType::kInt4)
        << to_string(g.dtype);
  }
}

TEST(DTypeTest, EveryDtypeIsExactlyOneKind) {
  // Floating-point, integral, or kBool — never two of the three.
  for (const DataType dtype : kAllDataTypes) {
    const int kinds = static_cast<int>(is_floating_point(dtype)) +
                      static_cast<int>(is_integral(dtype)) +
                      static_cast<int>(dtype == DataType::kBool);
    EXPECT_EQ(kinds, 1) << to_string(dtype);
  }
}

// Runtime restatement of the compile-time trait contract, so a regression
// shows up in the test log and the mapped type's width matches the enum's.
template <typename T>
void ExpectTraitMapping(DataType expected) {
  EXPECT_EQ(dtype_of<T>, expected);
  EXPECT_EQ(sizeof(T) * 8, static_cast<std::size_t>(itemsize_bits(expected)))
      << to_string(expected);
}

TEST(DTypeTest, TraitMappingMatchesEnumWidths) {
  ExpectTraitMapping<float>(DataType::kFloat32);
  ExpectTraitMapping<std::int8_t>(DataType::kInt8);
  ExpectTraitMapping<std::uint8_t>(DataType::kUInt8);
  ExpectTraitMapping<std::int32_t>(DataType::kInt32);
  ExpectTraitMapping<std::int64_t>(DataType::kInt64);
  ExpectTraitMapping<bool>(DataType::kBool);
}

}  // namespace
