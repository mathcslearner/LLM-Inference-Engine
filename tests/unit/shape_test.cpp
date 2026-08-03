#include "tensor/shape.h"

#include "core/status.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

using engine::core::IsInvalidArgument;
using engine::tensor::IsContiguous;
using engine::tensor::kMaxRank;
using engine::tensor::RowMajorStrides;
using engine::tensor::Shape;
using engine::tensor::Strides;

constexpr std::int64_t kInt64Max = std::numeric_limits<std::int64_t>::max();

// The whole layer is constexpr-usable; pin the core contracts at compile
// time (mirrors dtype_test's static_assert block).
static_assert(kMaxRank == 8);
static_assert(Shape{}.rank() == 0);
static_assert(Shape{}.numel() == 1);
static_assert(Shape{2, 3, 4}.rank() == 3);
static_assert(Shape{2, 3, 4}.dim(1) == 3);
static_assert(Shape{2, 3, 4}.numel() == 24);
static_assert(Shape{2, 0, 3}.numel() == 0);
static_assert(RowMajorStrides(Shape{2, 3, 4}) == Strides{12, 4, 1});
static_assert(IsContiguous(Shape{2, 3}, Strides{3, 1}));
static_assert(!IsContiguous(Shape{2, 2}, Strides{3, 1}));

// One golden row per rank, 0-d through 5-d; every table-driven test below
// walks these.
struct ShapeGolden {
  std::vector<std::int64_t> dims;
  std::int64_t numel;
  std::vector<std::int64_t> row_major_strides;
  std::string_view formatted;
};

const std::array<ShapeGolden, 6> kGolden = {{
    {.dims = {}, .numel = 1, .row_major_strides = {}, .formatted = "[]"},
    {.dims = {5}, .numel = 5, .row_major_strides = {1}, .formatted = "[5]"},
    {.dims = {2, 3},
     .numel = 6,
     .row_major_strides = {3, 1},
     .formatted = "[2, 3]"},
    {.dims = {2, 3, 4},
     .numel = 24,
     .row_major_strides = {12, 4, 1},
     .formatted = "[2, 3, 4]"},
    {.dims = {2, 3, 4, 5},
     .numel = 120,
     .row_major_strides = {60, 20, 5, 1},
     .formatted = "[2, 3, 4, 5]"},
    {.dims = {2, 3, 4, 5, 6},
     .numel = 720,
     .row_major_strides = {360, 120, 30, 6, 1},
     .formatted = "[2, 3, 4, 5, 6]"},
}};

Shape MustFromDims(const std::vector<std::int64_t>& dims) {
  auto shape = Shape::FromDims(dims);
  CHECK_OK(shape.status());
  return *shape;
}

std::vector<std::int64_t> ToVector(std::span<const std::int64_t> values) {
  return {values.begin(), values.end()};
}

TEST(ShapeTest, ConstructionGolden0dThrough5d) {
  for (const ShapeGolden& g : kGolden) {
    const Shape shape = MustFromDims(g.dims);
    ASSERT_EQ(shape.rank(), static_cast<int>(g.dims.size()));
    for (int i = 0; i < shape.rank(); ++i) {
      EXPECT_EQ(shape.dim(i), g.dims[static_cast<std::size_t>(i)]);
    }
    EXPECT_EQ(ToVector(shape.dims()), g.dims);
    EXPECT_EQ(shape.numel(), g.numel);
  }
}

TEST(ShapeTest, FromDimsMatchesLiteralConstruction) {
  EXPECT_EQ(MustFromDims({}), Shape{});
  EXPECT_EQ(MustFromDims({5}), (Shape{5}));
  EXPECT_EQ(MustFromDims({2, 3, 4}), (Shape{2, 3, 4}));
  EXPECT_EQ(MustFromDims({2, 0, 3}), (Shape{2, 0, 3}));
}

TEST(ShapeTest, ZeroDimsGiveNumelZero) {
  EXPECT_EQ((Shape{0}).numel(), 0);
  EXPECT_EQ((Shape{2, 0, 3}).numel(), 0);
  EXPECT_EQ((Shape{0, 0}).numel(), 0);
}

TEST(ShapeTest, SizeOneDimsGiveNumelOne) {
  EXPECT_EQ((Shape{1, 1, 1}).numel(), 1);
}

TEST(ShapeTest, MaxRankShapeIsAccepted) {
  const Shape shape{1, 2, 1, 2, 1, 2, 1, 2};
  EXPECT_EQ(shape.rank(), kMaxRank);
  EXPECT_EQ(shape.numel(), 16);
}

TEST(ShapeTest, FromDimsRejectsNegativeDim) {
  auto shape = Shape::FromDims(std::vector<std::int64_t>{2, -1, 3});
  ASSERT_FALSE(shape.ok());
  EXPECT_TRUE(IsInvalidArgument(shape.status())) << shape.status();
  // The offending value and its position are named in the message.
  EXPECT_NE(shape.status().message().find("-1"), std::string_view::npos)
      << shape.status();
  EXPECT_NE(shape.status().message().find("index 1"), std::string_view::npos)
      << shape.status();
}

TEST(ShapeTest, FromDimsRejectsRankAboveMax) {
  const std::vector<std::int64_t> dims(9, 1);
  auto shape = Shape::FromDims(dims);
  ASSERT_FALSE(shape.ok());
  EXPECT_TRUE(IsInvalidArgument(shape.status())) << shape.status();
}

TEST(ShapeTest, FromDimsRejectsNumelOverflow) {
  for (const std::vector<std::int64_t>& dims :
       {std::vector<std::int64_t>{std::int64_t{1} << 32, std::int64_t{1} << 32},
        std::vector<std::int64_t>{kInt64Max, 2},
        std::vector<std::int64_t>{kInt64Max, kInt64Max}}) {
    auto shape = Shape::FromDims(dims);
    ASSERT_FALSE(shape.ok());
    EXPECT_TRUE(IsInvalidArgument(shape.status())) << shape.status();
    EXPECT_NE(shape.status().message().find("overflows"),
              std::string_view::npos)
        << shape.status();
  }
}

TEST(ShapeTest, OverflowIsJudgedOnNonZeroDims) {
  // A 0 dim makes the true numel 0, so huge co-dims are fine ...
  const Shape shape = MustFromDims({0, kInt64Max});
  EXPECT_EQ(shape.numel(), 0);
  // ... but the non-zero dims must still multiply without overflow (they
  // bound every row-major stride; see ValidateDims in shape.h).
  auto overflowing =
      Shape::FromDims(std::vector<std::int64_t>{0, kInt64Max, 2});
  ASSERT_FALSE(overflowing.ok());
  EXPECT_TRUE(IsInvalidArgument(overflowing.status())) << overflowing.status();
}

TEST(ShapeTest, Equality) {
  EXPECT_EQ(Shape{}, Shape{});
  EXPECT_EQ((Shape{2, 3}), (Shape{2, 3}));
  EXPECT_NE((Shape{2, 3}), (Shape{3, 2}));
  EXPECT_NE((Shape{2, 3}), (Shape{2, 3, 1}));
  EXPECT_NE(Shape{}, (Shape{1}));
  EXPECT_EQ((Strides{12, 4, 1}), (Strides{12, 4, 1}));
  EXPECT_NE((Strides{12, 4, 1}), (Strides{12, 4, 2}));
}

TEST(ShapeTest, RowMajorStridesGolden) {
  for (const ShapeGolden& g : kGolden) {
    const Strides strides = RowMajorStrides(MustFromDims(g.dims));
    EXPECT_EQ(ToVector(strides.values()), g.row_major_strides)
        << fmt::format("shape {}", MustFromDims(g.dims));
  }
}

TEST(ShapeTest, RowMajorStridesTreatZeroDimsAsSizeOne) {
  EXPECT_EQ(RowMajorStrides(Shape{2, 0, 3}), (Strides{3, 3, 1}));
  EXPECT_EQ(RowMajorStrides(Shape{0}), (Strides{1}));
  EXPECT_EQ(RowMajorStrides(Shape{2, 1, 3}), (Strides{3, 3, 1}));
}

TEST(ShapeTest, RowMajorStridesAreContiguous) {
  for (const ShapeGolden& g : kGolden) {
    const Shape shape = MustFromDims(g.dims);
    EXPECT_TRUE(IsContiguous(shape, RowMajorStrides(shape)))
        << fmt::format("shape {}", shape);
  }
}

TEST(ShapeTest, ScalarIsContiguous) {
  EXPECT_TRUE(IsContiguous(Shape{}, Strides{}));
}

TEST(ShapeTest, SlicedStridesContiguity) {
  // Slicing dim 1 of a [2, 3] tensor to [2, 2] keeps strides [3, 1]: rows
  // are no longer adjacent, so the result is not contiguous.
  EXPECT_FALSE(IsContiguous(Shape{2, 2}, Strides{3, 1}));
  // Slicing an outermost dim only drops whole rows: [4, 3] → [2, 3] keeps
  // strides [3, 1] and stays contiguous (the view starts at an offset).
  EXPECT_TRUE(IsContiguous(Shape{2, 3}, Strides{3, 1}));
  // Slicing a middle dim of [2, 3, 4] to [2, 2, 4] with strides [12, 4, 1]
  // leaves a gap between the outer slabs.
  EXPECT_FALSE(IsContiguous(Shape{2, 2, 4}, Strides{12, 4, 1}));
}

TEST(ShapeTest, NonRowMajorStridesAreNotContiguous) {
  // Transposed [2, 3] view.
  EXPECT_FALSE(IsContiguous(Shape{3, 2}, Strides{1, 3}));
  // Inner stride must be exactly 1.
  EXPECT_FALSE(IsContiguous(Shape{2, 3}, Strides{6, 2}));
}

TEST(ShapeTest, SizeOneDimsImposeNoStrideConstraint) {
  EXPECT_TRUE(IsContiguous(Shape{1, 1}, Strides{999, 42}));
  EXPECT_TRUE(IsContiguous(Shape{3, 1}, Strides{1, 7}));
  EXPECT_TRUE(IsContiguous(Shape{1, 3}, Strides{3, 1}));
}

TEST(ShapeTest, EmptyShapesAreContiguousUnderAnyStrides) {
  EXPECT_TRUE(IsContiguous(Shape{2, 0, 3}, Strides{999, 999, 999}));
  EXPECT_TRUE(IsContiguous(Shape{0}, Strides{7}));
}

TEST(ShapeTest, FormattingGolden) {
  for (const ShapeGolden& g : kGolden) {
    EXPECT_EQ(fmt::format("{}", MustFromDims(g.dims)), g.formatted);
  }
  EXPECT_EQ(fmt::format("{}", Shape{0}), "[0]");
  EXPECT_EQ(fmt::format("{}", Strides{12, 4, 1}), "[12, 4, 1]");
  EXPECT_EQ(fmt::format("{}", Strides{}), "[]");
}

TEST(ShapeDeathTest, LiteralNegativeDimAborts) {
  EXPECT_DEATH((void)Shape({2, -1}), "negative dim -1 at index 1");
}

TEST(ShapeDeathTest, LiteralRankAboveMaxAborts) {
  EXPECT_DEATH((void)Shape({1, 1, 1, 1, 1, 1, 1, 1, 1}),
               "rank 9 exceeds kMaxRank \\(8\\)");
}

TEST(ShapeDeathTest, LiteralNumelOverflowAborts) {
  EXPECT_DEATH((void)Shape({kInt64Max, 2}), "dim product overflows int64_t");
}

TEST(ShapeDeathTest, DimIndexOutOfRangeAborts) {
  const Shape shape{2, 3, 4};
  EXPECT_DEATH((void)shape.dim(3), "index 3 out of range for size 3");
  EXPECT_DEATH((void)shape.dim(-1), "index -1 out of range for size 3");
}

TEST(ShapeDeathTest, IsContiguousRankMismatchAborts) {
  EXPECT_DEATH((void)IsContiguous(Shape{2, 3}, Strides{1}),
               "strides rank 1 != shape rank 2");
}

TEST(ShapeDeathTest, PushBackBeyondCapacityAborts) {
  Strides strides;
  for (int i = 0; i < kMaxRank; ++i) {
    strides.push_back(1);
  }
  EXPECT_DEATH(strides.push_back(1), "DimVector is full");
}

}  // namespace
