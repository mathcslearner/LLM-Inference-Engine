#include "tensor/ops.h"

#include "core/status.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace ops = engine::tensor::ops;

using engine::core::IsInvalidArgument;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::tensor::bfloat16;
using engine::tensor::DataType;
using engine::tensor::Device;
using engine::tensor::DimVector;
using engine::tensor::float16;
using engine::tensor::Shape;
using engine::tensor::Tensor;
using ops::AllCloseResult;

// Unwraps a StatusOr, failing the test on error.
template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

// Every element of a rank-1 or rank-2 CPU tensor equals `expected`.
template <typename T>
void ExpectAllEqual(const Tensor& tensor, T expected) {
  ASSERT_LE(tensor.shape().rank(), 2);
  if (tensor.shape().rank() == 1) {
    for (std::int64_t i = 0; i < tensor.shape().dim(0); ++i) {
      EXPECT_EQ(tensor.item<T>({i}), expected) << "at [" << i << "]";
    }
    return;
  }
  for (std::int64_t i = 0; i < tensor.shape().dim(0); ++i) {
    for (std::int64_t j = 0; j < tensor.shape().dim(1); ++j) {
      EXPECT_EQ(tensor.item<T>({i, j}), expected)
          << "at [" << i << ", " << j << "]";
    }
  }
}

// --- Factories ---

TEST(OpsFactoryTest, ZerosAndOnesEveryAllocatableDtype) {
  const Shape shape{2, 3};
  ExpectAllEqual<float>(Unwrap(ops::zeros(shape, DataType::kFloat32)), 0.0F);
  ExpectAllEqual<float>(Unwrap(ops::ones(shape, DataType::kFloat32)), 1.0F);
  ExpectAllEqual<std::int8_t>(Unwrap(ops::zeros(shape, DataType::kInt8)), 0);
  ExpectAllEqual<std::int8_t>(Unwrap(ops::ones(shape, DataType::kInt8)), 1);
  ExpectAllEqual<std::uint8_t>(Unwrap(ops::zeros(shape, DataType::kUInt8)), 0);
  ExpectAllEqual<std::uint8_t>(Unwrap(ops::ones(shape, DataType::kUInt8)), 1);
  ExpectAllEqual<std::int32_t>(Unwrap(ops::zeros(shape, DataType::kInt32)), 0);
  ExpectAllEqual<std::int32_t>(Unwrap(ops::ones(shape, DataType::kInt32)), 1);
  ExpectAllEqual<std::int64_t>(Unwrap(ops::zeros(shape, DataType::kInt64)), 0);
  ExpectAllEqual<std::int64_t>(Unwrap(ops::ones(shape, DataType::kInt64)), 1);
  ExpectAllEqual<bool>(Unwrap(ops::zeros(shape, DataType::kBool)), false);
  ExpectAllEqual<bool>(Unwrap(ops::ones(shape, DataType::kBool)), true);

  // Halves: assert the exact bit patterns, not just widened values.
  const Tensor half_zeros = Unwrap(ops::zeros(Shape{2}, DataType::kFloat16));
  const Tensor half_ones = Unwrap(ops::ones(Shape{2}, DataType::kFloat16));
  EXPECT_EQ(half_zeros.item<float16>({0}).to_bits(), 0x0000);
  EXPECT_EQ(half_ones.item<float16>({1}).to_bits(), 0x3C00);
  const Tensor bf_zeros = Unwrap(ops::zeros(Shape{2}, DataType::kBFloat16));
  const Tensor bf_ones = Unwrap(ops::ones(Shape{2}, DataType::kBFloat16));
  EXPECT_EQ(bf_zeros.item<bfloat16>({0}).to_bits(), 0x0000);
  EXPECT_EQ(bf_ones.item<bfloat16>({1}).to_bits(), 0x3F80);
}

TEST(OpsFactoryTest, FactoryMetadata) {
  const Tensor t = Unwrap(ops::zeros(Shape{2, 3}, DataType::kInt32));
  EXPECT_EQ(t.shape(), (Shape{2, 3}));
  EXPECT_EQ(t.dtype(), DataType::kInt32);
  EXPECT_EQ(t.device(), Device::Cpu());
  EXPECT_TRUE(t.is_contiguous());
}

TEST(OpsFactoryTest, FullFloatingValues) {
  ExpectAllEqual<float>(Unwrap(ops::full(Shape{3}, DataType::kFloat32, 2.5)),
                        2.5F);
  // The fp16 fill narrows double -> float -> fp16 (M1-T07 policy): the
  // stored bits are exactly float16(0.1F).
  const Tensor half = Unwrap(ops::full(Shape{2}, DataType::kFloat16, 0.1));
  EXPECT_EQ(half.item<float16>({0}).to_bits(), float16(0.1F).to_bits());
  const Tensor bf = Unwrap(ops::full(Shape{2}, DataType::kBFloat16, 0.1));
  EXPECT_EQ(bf.item<bfloat16>({0}).to_bits(), bfloat16(0.1F).to_bits());
  // Non-finite values pass through for floating dtypes.
  const Tensor inf = Unwrap(ops::full(Shape{1}, DataType::kFloat32,
                                      std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(std::isinf(inf.item<float>({0})));
  const Tensor nan = Unwrap(ops::full(
      Shape{1}, DataType::kFloat32, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(std::isnan(nan.item<float>({0})));
}

TEST(OpsFactoryTest, FullIntegerRangePolicy) {
  ExpectAllEqual<std::int8_t>(
      Unwrap(ops::full(Shape{2}, DataType::kInt8, -128.0)), -128);
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kInt8, -129.0).status()));
  ExpectAllEqual<std::uint8_t>(
      Unwrap(ops::full(Shape{2}, DataType::kUInt8, 255.0)), 255);
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kUInt8, 256.0).status()));
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kUInt8, -1.0).status()));
  // Non-integral, NaN, and inf values are not representable in any integer.
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kInt32, 0.5).status()));
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kInt64,
                                  std::numeric_limits<double>::quiet_NaN())
                            .status()));
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kInt64,
                                  std::numeric_limits<double>::infinity())
                            .status()));
  // int64 bounds: -2^63 is representable; 2^63 (exact as a double) is one
  // past the max.
  ExpectAllEqual<std::int64_t>(
      Unwrap(ops::full(Shape{2}, DataType::kInt64, -0x1p63)),
      std::numeric_limits<std::int64_t>::min());
  EXPECT_TRUE(IsInvalidArgument(
      ops::full(Shape{2}, DataType::kInt64, 0x1p63).status()));
  // kBool accepts exactly 0 or 1.
  EXPECT_TRUE(
      IsInvalidArgument(ops::full(Shape{2}, DataType::kBool, 2.0).status()));
}

TEST(OpsFactoryTest, ReservedDtypesAreUnimplemented) {
  // Unimplemented (the dtype's real blocker) wins over the value complaint.
  EXPECT_TRUE(
      IsUnimplemented(ops::full(Shape{2}, DataType::kInt4, 0.5).status()));
  EXPECT_TRUE(
      IsUnimplemented(ops::zeros(Shape{2}, DataType::kFP8E4M3).status()));
  EXPECT_TRUE(IsUnimplemented(ops::arange(0, 4, 1, DataType::kInt4).status()));
}

TEST(OpsFactoryTest, ZeroSizeTensors) {
  const Tensor t = Unwrap(ops::zeros(Shape{0, 3}, DataType::kFloat32));
  EXPECT_EQ(t.numel(), 0);
  EXPECT_TRUE(t.defined());
}

TEST(OpsArangeTest, Goldens) {
  const Tensor def = Unwrap(ops::arange(0, 5));
  EXPECT_EQ(def.dtype(), DataType::kInt64);  // NumPy/PyTorch default
  EXPECT_EQ(def.shape(), (Shape{5}));
  for (std::int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(def.item<std::int64_t>({i}), i);
  }

  const Tensor stepped = Unwrap(ops::arange(1, 10, 3, DataType::kInt32));
  EXPECT_EQ(stepped.shape(), (Shape{3}));
  EXPECT_EQ(stepped.item<std::int32_t>({0}), 1);
  EXPECT_EQ(stepped.item<std::int32_t>({1}), 4);
  EXPECT_EQ(stepped.item<std::int32_t>({2}), 7);

  const Tensor down = Unwrap(ops::arange(5, 0, -2, DataType::kInt32));
  EXPECT_EQ(down.shape(), (Shape{3}));
  EXPECT_EQ(down.item<std::int32_t>({0}), 5);
  EXPECT_EQ(down.item<std::int32_t>({1}), 3);
  EXPECT_EQ(down.item<std::int32_t>({2}), 1);

  const Tensor floats = Unwrap(ops::arange(0, 3, 1, DataType::kFloat32));
  EXPECT_EQ(floats.item<float>({0}), 0.0F);
  EXPECT_EQ(floats.item<float>({1}), 1.0F);
  EXPECT_EQ(floats.item<float>({2}), 2.0F);

  const Tensor halves = Unwrap(ops::arange(0, 4, 1, DataType::kFloat16));
  EXPECT_EQ(halves.item<float16>({3}).to_bits(), float16(3.0F).to_bits());

  const Tensor bools = Unwrap(ops::arange(0, 2, 1, DataType::kBool));
  EXPECT_FALSE(bools.item<bool>({0}));
  EXPECT_TRUE(bools.item<bool>({1}));
}

TEST(OpsArangeTest, EmptyRanges) {
  EXPECT_EQ(Unwrap(ops::arange(3, 3)).numel(), 0);
  EXPECT_EQ(Unwrap(ops::arange(3, 2, 1)).numel(), 0);   // wrong direction
  EXPECT_EQ(Unwrap(ops::arange(2, 3, -1)).numel(), 0);  // wrong direction
}

TEST(OpsArangeTest, InvalidArguments) {
  EXPECT_TRUE(IsInvalidArgument(ops::arange(0, 5, 0).status()));
  // Values 0, 100, 200: the last does not fit int8.
  EXPECT_TRUE(
      IsInvalidArgument(ops::arange(0, 300, 100, DataType::kInt8).status()));
  // Values -1, 0: -1 is not representable as kBool.
  EXPECT_TRUE(
      IsInvalidArgument(ops::arange(-1, 1, 1, DataType::kBool).status()));
  // end - start overflows int64.
  EXPECT_TRUE(IsInvalidArgument(
      ops::arange(-2, std::numeric_limits<std::int64_t>::max(), 1).status()));
}

// --- Seeded fills ---

TEST(OpsFillUniformTest, GoldenValuesFp32) {
  // Golden values pinned by the determinism contract in ops.h: mt19937_64
  // seeded with 42, u = (engine() >> 11) * 2^-53, value = low + u * extent.
  const Tensor t = Unwrap(ops::zeros(Shape{6}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(t, 0.0, 1.0, 42).ok());
  const float kExpected[6] = {0x1.82a3bep-1F, 0x1.472f2p-1F,  0x1.81192cp-1F,
                              0x1.171622p-3F, 0x1.ce7946p-1F, 0x1.814dc6p-4F};
  for (std::int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(t.item<float>({i}), kExpected[i]) << "at [" << i << "]";
  }
}

TEST(OpsFillUniformTest, GoldenValuesScaledRange) {
  const Tensor t = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(t, -2.0, 3.0, 7).ok());
  const float kExpected[4] = {0x1.c59cfap+0F, 0x1.5f8d82p+1F, -0x1.69b5bp+0F,
                              0x1.3ad30ep+1F};
  for (std::int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(t.item<float>({i}), kExpected[i]) << "at [" << i << "]";
  }
}

TEST(OpsFillUniformTest, GoldenBitsHalves) {
  const Tensor half = Unwrap(ops::zeros(Shape{4}, DataType::kFloat16));
  ASSERT_TRUE(ops::fill_uniform(half, 0.0, 1.0, 42).ok());
  const std::uint16_t kHalfBits[4] = {0x3A0B, 0x391D, 0x3A04, 0x305C};
  for (std::int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(half.item<float16>({i}).to_bits(), kHalfBits[i]);
  }
  const Tensor bf = Unwrap(ops::zeros(Shape{4}, DataType::kBFloat16));
  ASSERT_TRUE(ops::fill_uniform(bf, 0.0, 1.0, 42).ok());
  const std::uint16_t kBfBits[4] = {0x3F41, 0x3F24, 0x3F41, 0x3E0C};
  for (std::int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(bf.item<bfloat16>({i}).to_bits(), kBfBits[i]);
  }
}

TEST(OpsFillUniformTest, HalfFillIsNarrowedFp32Fill) {
  // Same seed => same double sequence; the half path must narrow exactly
  // like float16(<fp32 path value>).
  const Tensor f32 = Unwrap(ops::zeros(Shape{16}, DataType::kFloat32));
  const Tensor f16 = Unwrap(ops::zeros(Shape{16}, DataType::kFloat16));
  ASSERT_TRUE(ops::fill_uniform(f32, -1.0, 1.0, 2024).ok());
  ASSERT_TRUE(ops::fill_uniform(f16, -1.0, 1.0, 2024).ok());
  for (std::int64_t i = 0; i < 16; ++i) {
    EXPECT_EQ(f16.item<float16>({i}).to_bits(),
              float16(f32.item<float>({i})).to_bits());
  }
}

TEST(OpsFillUniformTest, DeterministicAndSeedSensitive) {
  const Tensor a = Unwrap(ops::zeros(Shape{32}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{32}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(a, 0.0, 1.0, 9).ok());
  ASSERT_TRUE(ops::fill_uniform(b, 0.0, 1.0, 9).ok());
  const AllCloseResult same = Unwrap(ops::allclose(a, b, 0.0, 0.0));
  EXPECT_TRUE(same.allclose) << same.Summary();

  ASSERT_TRUE(ops::fill_uniform(b, 0.0, 1.0, 10).ok());
  const AllCloseResult different = Unwrap(ops::allclose(a, b, 0.0, 0.0));
  EXPECT_FALSE(different.allclose);
}

TEST(OpsFillUniformTest, BoundsRespected) {
  const Tensor t = Unwrap(ops::zeros(Shape{256}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(t, 0.0, 1.0, 1).ok());
  for (std::int64_t i = 0; i < 256; ++i) {
    const auto v = t.item<float>({i});
    // [0, 1) in double; narrowing to fp32 may round up to exactly 1.
    EXPECT_GE(v, 0.0F);
    EXPECT_LE(v, 1.0F);
  }
}

TEST(OpsFillUniformTest, StridedViewFillsOnlyViewElements) {
  // Fill logical row-major order on a view == fill of an equally-shaped
  // fresh tensor, and the parent's other columns stay untouched.
  const Tensor parent = Unwrap(ops::zeros(Shape{2, 4}, DataType::kFloat32));
  const Tensor view = Unwrap(parent.slice(1, 1, 3));
  ASSERT_FALSE(view.is_contiguous());
  ASSERT_TRUE(ops::fill_uniform(view, 0.0, 1.0, 42).ok());

  const Tensor fresh = Unwrap(ops::zeros(Shape{2, 2}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(fresh, 0.0, 1.0, 42).ok());
  const AllCloseResult close = Unwrap(ops::allclose(view, fresh, 0.0, 0.0));
  EXPECT_TRUE(close.allclose) << close.Summary();

  for (std::int64_t row = 0; row < 2; ++row) {
    EXPECT_EQ(parent.item<float>({row, 0}), 0.0F);
    EXPECT_EQ(parent.item<float>({row, 3}), 0.0F);
  }
}

TEST(OpsFillUniformTest, RankZeroTensor) {
  const Tensor scalar = Unwrap(ops::zeros(Shape{}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(scalar, 0.0, 1.0, 42).ok());
  EXPECT_EQ(scalar.item<float>({}), 0x1.82a3bep-1F);  // first seed-42 value
}

TEST(OpsFillUniformTest, InvalidArguments) {
  const Tensor ints = Unwrap(ops::zeros(Shape{2}, DataType::kInt32));
  EXPECT_TRUE(IsInvalidArgument(ops::fill_uniform(ints, 0.0, 1.0, 0)));
  const Tensor t = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(ops::fill_uniform(t, 1.0, 0.0, 0)));
  EXPECT_TRUE(IsInvalidArgument(
      ops::fill_uniform(t, 0.0, std::numeric_limits<double>::infinity(), 0)));
  EXPECT_TRUE(IsInvalidArgument(
      ops::fill_uniform(t, std::numeric_limits<double>::quiet_NaN(), 1.0, 0)));
  // low <= high but high - low overflows to inf.
  EXPECT_TRUE(IsInvalidArgument(
      ops::fill_uniform(t, std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::max(), 0)));
}

TEST(OpsFillNormalTest, GoldenValuesFp32) {
  // Box-Muller on mt19937_64(123), per the ops.h contract.
  const Tensor t = Unwrap(ops::zeros(Shape{6}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(t, 0.0, 1.0, 123).ok());
  const float kExpected[6] = {-0x1.6e3328p+0F, -0x1.0cc7fcp-1F, -0x1.f5f71ap-6F,
                              -0x1.6c2722p-2F, 0x1.3a1c54p-1F,  0x1.b5720cp+0F};
  for (std::int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(t.item<float>({i}), kExpected[i]) << "at [" << i << "]";
  }
}

TEST(OpsFillNormalTest, GoldenValuesAffineOddCount) {
  // mean/stddev are an affine map of the same z sequence; the odd element
  // count exercises the discarded-z1 tail.
  const Tensor t = Unwrap(ops::zeros(Shape{3}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(t, 5.0, 2.0, 123).ok());
  const float kExpected[3] = {0x1.11ccd8p+1F, 0x1.f99c02p+1F, 0x1.3c1412p+2F};
  for (std::int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(t.item<float>({i}), kExpected[i]) << "at [" << i << "]";
  }
}

TEST(OpsFillNormalTest, DeterministicAndSeedSensitive) {
  const Tensor a = Unwrap(ops::zeros(Shape{33}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{33}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(a, 0.0, 1.0, 77).ok());
  ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, 77).ok());
  const AllCloseResult same = Unwrap(ops::allclose(a, b, 0.0, 0.0));
  EXPECT_TRUE(same.allclose) << same.Summary();

  ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, 78).ok());
  EXPECT_FALSE(Unwrap(ops::allclose(a, b, 0.0, 0.0)).allclose);
}

TEST(OpsFillNormalTest, SampleStatistics) {
  // Deterministic given the seed, so these are golden facts about seed 999,
  // sized so a correct sampler passes with wide margin: the sample mean of
  // n=10000 unit normals has stddev 0.01, the sample variance ~0.014.
  constexpr std::int64_t kN = 10000;
  const Tensor t = Unwrap(ops::zeros(Shape{kN}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(t, 0.0, 1.0, 999).ok());
  double sum = 0.0;
  double sum_sq = 0.0;
  for (std::int64_t i = 0; i < kN; ++i) {
    const double v = t.item<float>({i});
    sum += v;
    sum_sq += v * v;
  }
  const double mean = sum / kN;
  const double variance = (sum_sq / kN) - (mean * mean);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(variance, 1.0, 0.1);
}

TEST(OpsFillNormalTest, InvalidArguments) {
  const Tensor t = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(ops::fill_normal(t, 0.0, -1.0, 0)));
  EXPECT_TRUE(IsInvalidArgument(
      ops::fill_normal(t, std::numeric_limits<double>::infinity(), 1.0, 0)));
  EXPECT_TRUE(IsInvalidArgument(
      ops::fill_normal(t, 0.0, std::numeric_limits<double>::quiet_NaN(), 0)));
  const Tensor ints = Unwrap(ops::zeros(Shape{2}, DataType::kInt64));
  EXPECT_TRUE(IsInvalidArgument(ops::fill_normal(ints, 0.0, 1.0, 0)));
}

// --- allclose ---

TEST(OpsAllCloseTest, IdenticalTensorsAreClose) {
  const Tensor t = Unwrap(ops::zeros(Shape{2, 3}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(t, -1.0, 1.0, 3).ok());
  const AllCloseResult result = Unwrap(ops::allclose(t, t));
  EXPECT_TRUE(result.allclose);
  EXPECT_EQ(result.numel, 6);
  EXPECT_EQ(result.mismatch_count, 0);
  EXPECT_EQ(result.max_abs_diff, 0.0);
}

TEST(OpsAllCloseTest, WorstMismatchReport) {
  const Tensor a = Unwrap(ops::zeros(Shape{2, 3}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{2, 3}, DataType::kFloat32));
  b.data_ptr<float>()[4] = 0.25F;  // logical index [1, 1]

  const AllCloseResult result = Unwrap(ops::allclose(a, b));
  EXPECT_FALSE(result.allclose);
  EXPECT_EQ(result.mismatch_count, 1);
  EXPECT_EQ(result.worst_index, (DimVector{1, 1}));
  EXPECT_EQ(result.worst_a, 0.0);
  EXPECT_EQ(result.worst_b, 0.25);
  EXPECT_EQ(result.max_abs_diff, 0.25);
  // The acceptance criterion: the failure message names the worst
  // mismatch's index and both values.
  EXPECT_EQ(result.Summary(),
            "allclose: FAILED — 1 of 6 elements mismatch; worst at [1, 1]: "
            "a=0, b=0.25 (abs diff 0.25); max abs diff 0.25 (rtol=1e-05, "
            "atol=1e-08)");
  EXPECT_EQ(Unwrap(ops::allclose(a, a)).Summary(),
            "allclose: OK — 6 elements, max abs diff 0 (rtol=1e-05, "
            "atol=1e-08)");
}

TEST(OpsAllCloseTest, WorstMismatchIsTheLargest) {
  const Tensor a = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  b.data_ptr<float>()[1] = 0.5F;
  b.data_ptr<float>()[2] = -2.0F;  // the worst
  b.data_ptr<float>()[3] = 1.0F;
  const AllCloseResult result = Unwrap(ops::allclose(a, b));
  EXPECT_EQ(result.mismatch_count, 3);
  EXPECT_EQ(result.worst_index, (DimVector{2}));
  EXPECT_EQ(result.worst_b, -2.0);
  EXPECT_EQ(result.max_abs_diff, 2.0);
}

TEST(OpsAllCloseTest, RtolAndAtolSemantics) {
  const Tensor a = Unwrap(ops::full(Shape{1}, DataType::kFloat32, 1000.0));
  const Tensor b =
      Unwrap(ops::full(Shape{1}, DataType::kFloat32, 1000.0078125));
  // |a-b| = 0.0078125: within rtol 1e-5 of |b| (~0.01), not within 1e-6.
  EXPECT_TRUE(Unwrap(ops::allclose(a, b, 1e-5, 0.0)).allclose);
  EXPECT_FALSE(Unwrap(ops::allclose(a, b, 1e-6, 0.0)).allclose);
  // Same difference judged by atol alone.
  EXPECT_TRUE(Unwrap(ops::allclose(a, b, 0.0, 0.01)).allclose);
  EXPECT_FALSE(Unwrap(ops::allclose(a, b, 0.0, 0.001)).allclose);
}

TEST(OpsAllCloseTest, PerDtypeDefaultTolerances) {
  // One fp16 ulp at 1.0: within the fp16 default (rtol 1e-3), outside the
  // fp32-grade tolerances.
  const Tensor a = Unwrap(ops::ones(Shape{1}, DataType::kFloat16));
  const Tensor b = Unwrap(ops::ones(Shape{1}, DataType::kFloat16));
  b.data_ptr<float16>()[0] = float16::from_bits(0x3C01);  // 1 + 2^-10
  EXPECT_TRUE(Unwrap(ops::allclose(a, b)).allclose);
  EXPECT_FALSE(Unwrap(ops::allclose(a, b, 1e-5, 1e-8)).allclose);

  // One bf16 ulp at 1.0 (1 + 2^-7): within the bf16 default rtol 1.6e-2.
  const Tensor ba = Unwrap(ops::ones(Shape{1}, DataType::kBFloat16));
  const Tensor bb = Unwrap(ops::ones(Shape{1}, DataType::kBFloat16));
  bb.data_ptr<bfloat16>()[0] = bfloat16::from_bits(0x3F81);
  EXPECT_TRUE(Unwrap(ops::allclose(ba, bb)).allclose);
  EXPECT_FALSE(Unwrap(ops::allclose(ba, bb, 1e-3, 1e-5)).allclose);
}

TEST(OpsAllCloseTest, IntegerDtypesCompareExactly) {
  const Tensor a = Unwrap(ops::full(Shape{2}, DataType::kInt32, 100.0));
  const Tensor b = Unwrap(ops::full(Shape{2}, DataType::kInt32, 101.0));
  const AllCloseResult result = Unwrap(ops::allclose(a, b));
  EXPECT_FALSE(result.allclose);
  EXPECT_EQ(result.rtol, 0.0);  // integer default
  EXPECT_EQ(result.max_abs_diff, 1.0);
  // Integer comparison is exact regardless of explicit tolerances.
  EXPECT_FALSE(Unwrap(ops::allclose(a, b, 0.5, 10.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(a, a)).allclose);

  const Tensor ta = Unwrap(ops::ones(Shape{2}, DataType::kBool));
  const Tensor tb = Unwrap(ops::zeros(Shape{2}, DataType::kBool));
  EXPECT_FALSE(Unwrap(ops::allclose(ta, tb)).allclose);
}

TEST(OpsAllCloseTest, NanIsNeverClose) {
  const Tensor a = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  a.data_ptr<float>()[1] = 100.0F;
  b.data_ptr<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  // NaN vs NaN is also a mismatch.
  const Tensor nn = Unwrap(ops::full(Shape{1}, DataType::kFloat32,
                                     std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(Unwrap(ops::allclose(nn, nn)).allclose);

  const AllCloseResult result = Unwrap(ops::allclose(a, b));
  EXPECT_FALSE(result.allclose);
  EXPECT_EQ(result.mismatch_count, 2);
  // The NaN mismatch is maximally bad and wins the worst slot; the numeric
  // mismatch still drives max_abs_diff.
  EXPECT_EQ(result.worst_index, (DimVector{0}));
  EXPECT_TRUE(std::isnan(result.worst_b));
  EXPECT_EQ(result.max_abs_diff, 100.0);
}

TEST(OpsAllCloseTest, InfinityHandling) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const Tensor pos = Unwrap(ops::full(Shape{1}, DataType::kFloat32, kInf));
  const Tensor neg = Unwrap(ops::full(Shape{1}, DataType::kFloat32, -kInf));
  const Tensor fin = Unwrap(ops::full(Shape{1}, DataType::kFloat32, 1.0));
  EXPECT_TRUE(Unwrap(ops::allclose(pos, pos)).allclose);
  const AllCloseResult opposite = Unwrap(ops::allclose(pos, neg));
  EXPECT_FALSE(opposite.allclose);
  EXPECT_EQ(opposite.max_abs_diff, kInf);
  EXPECT_FALSE(Unwrap(ops::allclose(pos, fin)).allclose);
}

TEST(OpsAllCloseTest, StridedViews) {
  const Tensor parent = Unwrap(
      Unwrap(ops::arange(0, 8, 1, DataType::kFloat32)).reshape(Shape{2, 4}));
  const Tensor view = Unwrap(parent.slice(1, 1, 3));  // [[1, 2], [5, 6]]
  ASSERT_FALSE(view.is_contiguous());
  const Tensor expected = Unwrap(ops::zeros(Shape{2, 2}, DataType::kFloat32));
  expected.data_ptr<float>()[0] = 1.0F;
  expected.data_ptr<float>()[1] = 2.0F;
  expected.data_ptr<float>()[2] = 5.0F;
  expected.data_ptr<float>()[3] = 6.0F;
  EXPECT_TRUE(Unwrap(ops::allclose(view, expected, 0.0, 0.0)).allclose);

  expected.data_ptr<float>()[3] = 7.0F;
  const AllCloseResult mismatch =
      Unwrap(ops::allclose(view, expected, 0.0, 0.0));
  EXPECT_FALSE(mismatch.allclose);
  EXPECT_EQ(mismatch.worst_index, (DimVector{1, 1}));  // view coordinates
}

TEST(OpsAllCloseTest, EmptyAndScalarTensors) {
  const Tensor e1 = Unwrap(ops::zeros(Shape{0}, DataType::kFloat32));
  const Tensor e2 = Unwrap(ops::zeros(Shape{0}, DataType::kFloat32));
  const AllCloseResult empty = Unwrap(ops::allclose(e1, e2));
  EXPECT_TRUE(empty.allclose);
  EXPECT_EQ(empty.numel, 0);

  const Tensor s1 = Unwrap(ops::full(Shape{}, DataType::kFloat32, 1.5));
  const Tensor s2 = Unwrap(ops::full(Shape{}, DataType::kFloat32, 1.6));
  EXPECT_TRUE(Unwrap(ops::allclose(s1, s1)).allclose);
  const AllCloseResult scalar = Unwrap(ops::allclose(s1, s2));
  EXPECT_FALSE(scalar.allclose);
  EXPECT_EQ(scalar.worst_index, DimVector{});
}

TEST(OpsAllCloseTest, InvalidArguments) {
  const Tensor f32 = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  const Tensor i32 = Unwrap(ops::zeros(Shape{2}, DataType::kInt32));
  const Tensor other = Unwrap(ops::zeros(Shape{3}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(ops::allclose(f32, i32).status()));
  EXPECT_TRUE(IsInvalidArgument(ops::allclose(f32, other).status()));
  EXPECT_TRUE(IsInvalidArgument(ops::allclose(f32, f32, -1.0, 0.0).status()));
  EXPECT_TRUE(IsInvalidArgument(
      ops::allclose(f32, f32, 0.0, std::numeric_limits<double>::quiet_NaN())
          .status()));
}

TEST(OpsAllCloseTest, DefaultToleranceTable) {
  constexpr ops::AllCloseTolerance kF32 =
      ops::default_allclose_tolerance(DataType::kFloat32);
  static_assert(kF32.rtol == 1e-5 && kF32.atol == 1e-8);
  constexpr ops::AllCloseTolerance kF16 =
      ops::default_allclose_tolerance(DataType::kFloat16);
  static_assert(kF16.rtol == 1e-3 && kF16.atol == 1e-5);
  constexpr ops::AllCloseTolerance kBf16 =
      ops::default_allclose_tolerance(DataType::kBFloat16);
  static_assert(kBf16.rtol == 1.6e-2 && kBf16.atol == 1e-5);
  constexpr ops::AllCloseTolerance kInt =
      ops::default_allclose_tolerance(DataType::kInt32);
  static_assert(kInt.rtol == 0.0 && kInt.atol == 0.0);
}

// --- Printing ---

TEST(OpsToStringTest, TwoDimensional) {
  const Tensor t = Unwrap(
      Unwrap(ops::arange(0, 6, 1, DataType::kFloat32)).reshape(Shape{2, 3}));
  EXPECT_EQ(ops::to_string(t),
            "Tensor(shape=[2, 3], dtype=float32, device=cpu)\n"
            "[[0, 1, 2],\n"
            " [3, 4, 5]]");
}

TEST(OpsToStringTest, TruncatesLongDims) {
  const Tensor t = Unwrap(ops::arange(0, 10));
  EXPECT_EQ(ops::to_string(t),
            "Tensor(shape=[10], dtype=int64, device=cpu)\n"
            "[0, 1, 2, ..., 7, 8, 9]");
  // A larger edge_items shows everything.
  EXPECT_EQ(ops::to_string(t, 5),
            "Tensor(shape=[10], dtype=int64, device=cpu)\n"
            "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
}

TEST(OpsToStringTest, ScalarEmptyAndBool) {
  EXPECT_EQ(ops::to_string(Unwrap(ops::full(Shape{}, DataType::kFloat32, 1.5))),
            "Tensor(shape=[], dtype=float32, device=cpu)\n1.5");
  EXPECT_EQ(ops::to_string(Unwrap(ops::zeros(Shape{0}, DataType::kInt32))),
            "Tensor(shape=[0], dtype=int32, device=cpu)\n[]");
  EXPECT_EQ(ops::to_string(Unwrap(ops::arange(0, 2, 1, DataType::kBool))),
            "Tensor(shape=[2], dtype=bool, device=cpu)\n[false, true]");
}

TEST(OpsToStringTest, NestedIndentation) {
  const Tensor t = Unwrap(
      Unwrap(ops::arange(0, 8, 1, DataType::kInt32)).reshape(Shape{2, 2, 2}));
  EXPECT_EQ(ops::to_string(t),
            "Tensor(shape=[2, 2, 2], dtype=int32, device=cpu)\n"
            "[[[0, 1],\n"
            "  [2, 3]],\n"
            " [[4, 5],\n"
            "  [6, 7]]]");
}

TEST(OpsToStringTest, StridedViewPrintsLogicalValues) {
  const Tensor parent = Unwrap(
      Unwrap(ops::arange(0, 8, 1, DataType::kInt32)).reshape(Shape{2, 4}));
  const Tensor view = Unwrap(parent.slice(1, 1, 3));
  EXPECT_EQ(ops::to_string(view),
            "Tensor(shape=[2, 2], dtype=int32, device=cpu)\n"
            "[[1, 2],\n"
            " [5, 6]]");
}

TEST(OpsToStringTest, UndefinedHandle) {
  EXPECT_EQ(ops::to_string(Tensor{}), "Tensor(undefined)");
}

// --- copy ---

TEST(OpsCopyTest, ContiguousRoundTripEveryAllocatableDtype) {
  for (const DataType dtype :
       {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16,
        DataType::kInt8, DataType::kUInt8, DataType::kInt32, DataType::kInt64,
        DataType::kBool}) {
    const std::int64_t n = dtype == DataType::kBool ? 2 : 6;
    const Tensor src = Unwrap(ops::arange(0, n, 1, dtype));
    const Tensor dst = Unwrap(ops::zeros(Shape{n}, dtype));
    ASSERT_TRUE(ops::copy(dst, src).ok());
    const AllCloseResult close = Unwrap(ops::allclose(dst, src, 0.0, 0.0));
    EXPECT_TRUE(close.allclose)
        << engine::tensor::to_string(dtype) << ": " << close.Summary();
  }
}

TEST(OpsCopyTest, CopiesAreDeep) {
  const Tensor src = Unwrap(ops::arange(0, 4, 1, DataType::kFloat32));
  const Tensor dst = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(dst, src).ok());
  src.data_ptr<float>()[0] = 42.0F;
  EXPECT_EQ(dst.item<float>({0}), 0.0F);  // dst has its own bytes
}

TEST(OpsCopyTest, StridedDestinationWritesOnlyViewElements) {
  const Tensor parent = Unwrap(ops::zeros(Shape{2, 4}, DataType::kFloat32));
  const Tensor view = Unwrap(parent.slice(1, 1, 3));
  ASSERT_FALSE(view.is_contiguous());
  const Tensor src = Unwrap(
      Unwrap(ops::arange(1, 5, 1, DataType::kFloat32)).reshape(Shape{2, 2}));
  ASSERT_TRUE(ops::copy(view, src).ok());
  // Columns 1..2 hold src's rows; columns 0 and 3 stay untouched.
  EXPECT_EQ(parent.item<float>({0, 1}), 1.0F);
  EXPECT_EQ(parent.item<float>({0, 2}), 2.0F);
  EXPECT_EQ(parent.item<float>({1, 1}), 3.0F);
  EXPECT_EQ(parent.item<float>({1, 2}), 4.0F);
  for (std::int64_t row = 0; row < 2; ++row) {
    EXPECT_EQ(parent.item<float>({row, 0}), 0.0F);
    EXPECT_EQ(parent.item<float>({row, 3}), 0.0F);
  }
}

TEST(OpsCopyTest, StridedSourceReadsLogicalValues) {
  const Tensor parent = Unwrap(
      Unwrap(ops::arange(0, 8, 1, DataType::kInt32)).reshape(Shape{2, 4}));
  const Tensor view = Unwrap(parent.slice(1, 1, 3));  // [[1, 2], [5, 6]]
  const Tensor dst = Unwrap(ops::zeros(Shape{2, 2}, DataType::kInt32));
  ASSERT_TRUE(ops::copy(dst, view).ok());
  EXPECT_EQ(dst.item<std::int32_t>({0, 0}), 1);
  EXPECT_EQ(dst.item<std::int32_t>({0, 1}), 2);
  EXPECT_EQ(dst.item<std::int32_t>({1, 0}), 5);
  EXPECT_EQ(dst.item<std::int32_t>({1, 1}), 6);
}

TEST(OpsCopyTest, StridedBothSides) {
  const Tensor src_parent = Unwrap(
      Unwrap(ops::arange(0, 8, 1, DataType::kInt64)).reshape(Shape{2, 4}));
  const Tensor src = Unwrap(src_parent.slice(1, 1, 3));  // [[1, 2], [5, 6]]
  const Tensor dst_parent = Unwrap(ops::zeros(Shape{2, 4}, DataType::kInt64));
  const Tensor dst = Unwrap(dst_parent.slice(1, 2, 4));
  ASSERT_FALSE(src.is_contiguous());
  ASSERT_FALSE(dst.is_contiguous());
  ASSERT_TRUE(ops::copy(dst, src).ok());
  EXPECT_EQ(dst_parent.item<std::int64_t>({0, 2}), 1);
  EXPECT_EQ(dst_parent.item<std::int64_t>({0, 3}), 2);
  EXPECT_EQ(dst_parent.item<std::int64_t>({1, 2}), 5);
  EXPECT_EQ(dst_parent.item<std::int64_t>({1, 3}), 6);
  EXPECT_EQ(dst_parent.item<std::int64_t>({0, 0}), 0);
  EXPECT_EQ(dst_parent.item<std::int64_t>({1, 1}), 0);
}

TEST(OpsCopyTest, SelfCopyIsANoOp) {
  const Tensor t = Unwrap(ops::arange(0, 4, 1, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(t, t).ok());
  for (std::int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(t.item<float>({i}), static_cast<float>(i));
  }
}

TEST(OpsCopyTest, EmptyTensors) {
  const Tensor a = Unwrap(ops::zeros(Shape{0, 3}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{0, 3}, DataType::kFloat32));
  EXPECT_TRUE(ops::copy(a, b).ok());
}

TEST(OpsCopyTest, InvalidArguments) {
  const Tensor f32 = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  const Tensor i32 = Unwrap(ops::zeros(Shape{2}, DataType::kInt32));
  const Tensor other = Unwrap(ops::zeros(Shape{3}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(ops::copy(f32, i32)));  // no implicit cast
  EXPECT_TRUE(IsInvalidArgument(ops::copy(f32, other)));
}

// --- cast ---

TEST(OpsCastTest, AllSupportedPairs) {
  // Values 0..3 are exactly representable in every supported dtype, so
  // every pair must produce exactly arange of the target dtype.
  constexpr DataType kCastable[] = {DataType::kFloat32,  DataType::kFloat16,
                                    DataType::kBFloat16, DataType::kInt8,
                                    DataType::kUInt8,    DataType::kInt32,
                                    DataType::kInt64};
  for (const DataType src_dtype : kCastable) {
    for (const DataType dst_dtype : kCastable) {
      const Tensor src = Unwrap(ops::arange(0, 4, 1, src_dtype));
      const Tensor out = Unwrap(ops::cast(src, dst_dtype));
      EXPECT_EQ(out.dtype(), dst_dtype);
      EXPECT_EQ(out.shape(), (Shape{4}));
      EXPECT_TRUE(out.is_contiguous());
      const Tensor expected = Unwrap(ops::arange(0, 4, 1, dst_dtype));
      const AllCloseResult close =
          Unwrap(ops::allclose(out, expected, 0.0, 0.0));
      EXPECT_TRUE(close.allclose)
          << engine::tensor::to_string(src_dtype) << " -> "
          << engine::tensor::to_string(dst_dtype) << ": " << close.Summary();
    }
  }
}

TEST(OpsCastTest, Fp32ToHalvesMatchesM1T07Exactly) {
  // The acceptance criterion: cast narrows bit-identically to the half.h
  // constructors. Rounding boundaries (ties round to even), overflow to
  // inf, subnormals, signed zero, NaN, infinities, plus seeded random
  // values.
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const float kBoundary[] = {
      1.0F + 0x1p-11F,    // fp16 tie at 1.0 -> rounds to even (1.0)
      1.0F + 0x1.8p-10F,  // fp16 tie -> rounds up to even
      65504.0F,           // fp16 max finite
      65520.0F,           // fp16 overflow tie -> inf
      65519.9F,           // just below the tie -> stays 65504
      0x1p-25F,           // fp16 subnormal tie -> 0
      0x1.8p-24F,         // fp16 subnormal, rounds up
      1.0F + 0x1p-8F,     // bf16 tie at 1.0
      0.1F,
      -0.0F,
      kInf,
      -kInf,
      std::numeric_limits<float>::quiet_NaN(),
  };
  constexpr std::int64_t kNumBoundary = std::size(kBoundary);
  constexpr std::int64_t kNumRandom = 64;
  const Tensor src =
      Unwrap(ops::zeros(Shape{kNumBoundary + kNumRandom}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_uniform(src, -70000.0, 70000.0, 424242).ok());
  for (std::int64_t i = 0; i < kNumBoundary; ++i) {
    src.data_ptr<float>()[i] = kBoundary[i];
  }

  const Tensor f16 = Unwrap(ops::cast(src, DataType::kFloat16));
  const Tensor bf16 = Unwrap(ops::cast(src, DataType::kBFloat16));
  for (std::int64_t i = 0; i < src.numel(); ++i) {
    const auto value = src.item<float>({i});
    EXPECT_EQ(f16.item<float16>({i}).to_bits(), float16(value).to_bits())
        << "fp16 at [" << i << "], value " << value;
    EXPECT_EQ(bf16.item<bfloat16>({i}).to_bits(), bfloat16(value).to_bits())
        << "bf16 at [" << i << "], value " << value;
  }
}

TEST(OpsCastTest, HalvesWidenExactly) {
  const Tensor f16 = Unwrap(ops::zeros(Shape{3}, DataType::kFloat16));
  f16.data_ptr<float16>()[0] = float16::from_bits(0x3555);  // ~0.3333
  f16.data_ptr<float16>()[1] = float16::from_bits(0x0001);  // min subnormal
  f16.data_ptr<float16>()[2] = float16::from_bits(0xFBFF);  // -max finite
  const Tensor widened = Unwrap(ops::cast(f16, DataType::kFloat32));
  for (std::int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(widened.item<float>({i}),
              static_cast<float>(f16.item<float16>({i})));
  }
}

TEST(OpsCastTest, SameDtypeCastIsADeepCopy) {
  const Tensor src = Unwrap(ops::arange(0, 4, 1, DataType::kFloat32));
  const Tensor out = Unwrap(ops::cast(src, DataType::kFloat32));
  EXPECT_NE(out.data(), src.data());
  src.data_ptr<float>()[0] = 42.0F;
  EXPECT_EQ(out.item<float>({0}), 0.0F);
}

TEST(OpsCastTest, StridedSource) {
  const Tensor parent = Unwrap(
      Unwrap(ops::arange(0, 8, 1, DataType::kInt32)).reshape(Shape{2, 4}));
  const Tensor view = Unwrap(parent.slice(1, 1, 3));  // [[1, 2], [5, 6]]
  ASSERT_FALSE(view.is_contiguous());
  const Tensor out = Unwrap(ops::cast(view, DataType::kFloat32));
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_EQ(out.item<float>({0, 0}), 1.0F);
  EXPECT_EQ(out.item<float>({0, 1}), 2.0F);
  EXPECT_EQ(out.item<float>({1, 0}), 5.0F);
  EXPECT_EQ(out.item<float>({1, 1}), 6.0F);
}

TEST(OpsCastTest, FloatToIntTruncatesTowardZero) {
  const Tensor src = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
  src.data_ptr<float>()[0] = 2.7F;
  src.data_ptr<float>()[1] = -2.7F;
  src.data_ptr<float>()[2] = 0.9F;
  src.data_ptr<float>()[3] = -0.9F;
  const Tensor out = Unwrap(ops::cast(src, DataType::kInt32));
  EXPECT_EQ(out.item<std::int32_t>({0}), 2);
  EXPECT_EQ(out.item<std::int32_t>({1}), -2);
  EXPECT_EQ(out.item<std::int32_t>({2}), 0);
  EXPECT_EQ(out.item<std::int32_t>({3}), 0);
}

TEST(OpsCastTest, IntToFloatRounds) {
  // 2^24 + 1 is not representable in fp32; nearest-even gives 2^24.
  const Tensor i32 = Unwrap(ops::full(Shape{1}, DataType::kInt32, 16777217.0));
  EXPECT_EQ(Unwrap(ops::cast(i32, DataType::kFloat32)).item<float>({0}),
            16777216.0F);
  // Beyond the fp16 range: overflows to +inf (NumPy behavior), not an error.
  const Tensor big = Unwrap(ops::full(Shape{1}, DataType::kInt32, 70000.0));
  const float widened =
      Unwrap(ops::cast(big, DataType::kFloat16)).item<float16>({0});
  EXPECT_TRUE(std::isinf(widened));
  EXPECT_GT(widened, 0.0F);
}

TEST(OpsCastTest, UnrepresentableIntegerTargetsAreInvalid) {
  const Tensor nan = Unwrap(ops::full(
      Shape{2}, DataType::kFloat32, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(IsInvalidArgument(ops::cast(nan, DataType::kInt32).status()));
  const Tensor inf = Unwrap(ops::full(Shape{2}, DataType::kFloat32,
                                      std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(IsInvalidArgument(ops::cast(inf, DataType::kInt64).status()));
  const Tensor big = Unwrap(ops::full(Shape{2}, DataType::kFloat32, 300.0));
  EXPECT_TRUE(IsInvalidArgument(ops::cast(big, DataType::kInt8).status()));
  const Tensor negative = Unwrap(ops::full(Shape{2}, DataType::kInt32, -1.0));
  EXPECT_TRUE(
      IsInvalidArgument(ops::cast(negative, DataType::kUInt8).status()));

  // Integer->integer narrowing checks the range too; the error names the
  // first offending value and its logical index.
  const Tensor i32 = Unwrap(ops::arange(127, 129, 1, DataType::kInt32));
  const auto narrowed = ops::cast(i32, DataType::kInt8);
  EXPECT_TRUE(IsInvalidArgument(narrowed.status()));
  EXPECT_NE(narrowed.status().message().find("value 128 at index [1]"),
            std::string_view::npos)
      << narrowed.status().ToString();
  // In-range narrowing works.
  const Tensor ok = Unwrap(ops::arange(-128, -126, 1, DataType::kInt32));
  EXPECT_EQ(Unwrap(ops::cast(ok, DataType::kInt8)).item<std::int8_t>({0}),
            -128);
}

TEST(OpsCastTest, BoolIsUnsupported) {
  const Tensor b = Unwrap(ops::zeros(Shape{2}, DataType::kBool));
  EXPECT_TRUE(IsInvalidArgument(ops::cast(b, DataType::kFloat32).status()));
  const Tensor f32 = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(ops::cast(f32, DataType::kBool).status()));
}

TEST(OpsCastTest, ReservedTargetIsUnimplemented) {
  const Tensor f32 = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  EXPECT_TRUE(IsUnimplemented(ops::cast(f32, DataType::kInt4).status()));
  // Unimplemented (the dtype's real blocker) wins over the kBool complaint.
  const Tensor b = Unwrap(ops::zeros(Shape{2}, DataType::kBool));
  EXPECT_TRUE(IsUnimplemented(ops::cast(b, DataType::kFP8E4M3).status()));
}

TEST(OpsCastTest, EmptyTensor) {
  const Tensor empty = Unwrap(ops::zeros(Shape{0, 3}, DataType::kFloat32));
  const Tensor out = Unwrap(ops::cast(empty, DataType::kInt32));
  EXPECT_EQ(out.numel(), 0);
  EXPECT_EQ(out.dtype(), DataType::kInt32);
  EXPECT_EQ(out.shape(), (Shape{0, 3}));
}

// --- Death tests (programmer errors per the design §7 table) ---

TEST(OpsDeathTest, FillOnUndefinedTensorAborts) {
  const Tensor undefined;
  EXPECT_DEATH((void)ops::fill_uniform(undefined, 0.0, 1.0, 0),
               "undefined Tensor");
  EXPECT_DEATH((void)ops::fill_normal(undefined, 0.0, 1.0, 0),
               "undefined Tensor");
}

TEST(OpsDeathTest, CopyOnUndefinedTensorAborts) {
  const Tensor undefined;
  const Tensor defined = Unwrap(ops::zeros(Shape{1}, DataType::kFloat32));
  EXPECT_DEATH((void)ops::copy(undefined, defined), "undefined Tensor");
  EXPECT_DEATH((void)ops::copy(defined, undefined), "undefined Tensor");
}

TEST(OpsDeathTest, CastOnUndefinedTensorAborts) {
  const Tensor undefined;
  EXPECT_DEATH((void)ops::cast(undefined, DataType::kFloat32),
               "undefined Tensor");
}

TEST(OpsDeathTest, AllCloseOnUndefinedTensorAborts) {
  const Tensor undefined;
  const Tensor defined = Unwrap(ops::zeros(Shape{1}, DataType::kFloat32));
  EXPECT_DEATH((void)ops::allclose(undefined, defined), "undefined Tensor");
}

TEST(OpsDeathTest, ToStringBadEdgeItemsAborts) {
  const Tensor t = Unwrap(ops::zeros(Shape{2}, DataType::kFloat32));
  EXPECT_DEATH((void)ops::to_string(t, 0), "edge_items");
}

TEST(OpsDeathTest, ReservedDtypeToleranceAborts) {
  EXPECT_DEATH((void)ops::default_allclose_tolerance(DataType::kInt4),
               "reserved dtype");
}

}  // namespace
