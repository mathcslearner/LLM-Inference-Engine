#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "kernels/embedding.h"
#include "kernels/gemm.h"
#include "model/loader.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

// Embedding-lookup kernels (M6-T06; design: optimized-cpu-execution.md §7,
// §10). A pure gather + exact fp16/bf16->fp32 widen, so every result is
// **bit-identical** to a single-threaded reference gather AND to the
// `cpu::embedding_lookup` oracle, across ISAs and thread counts (§10). Both
// source layouts (row-major table, packed lm_head) gather the same logical row.
// Registered SCALAR_PASS (§9): the forced-scalar pass exercises the scalar
// widen path.
namespace {

namespace ops = engine::tensor::ops;
namespace k = engine::kernels;
using engine::core::StatusOr;
using engine::model::load_model;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] std::span<const std::int32_t> IdsSpan(
    const std::vector<std::int32_t>& ids) {
  return {ids.data(), ids.size()};
}

// A single-threaded, layout-agnostic reference: gather fp32 rows out of the
// table already cast to fp32 (an exact, unique widen). The kernels widen the
// half table at gather time; since widening is lossless the two agree to the
// last bit — so this reference is simultaneously the numeric oracle and the
// thread-count/partition-independent baseline.
[[nodiscard]] Tensor GatherRef(const Tensor& table_f32,
                               const std::vector<std::int32_t>& ids) {
  const std::int64_t e = table_f32.shape().dim(1);
  const auto rows = static_cast<std::int64_t>(ids.size());
  Tensor out = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  const auto* src = table_f32.data_ptr<float>();
  auto* dst = out.data_ptr<float>();
  for (std::int64_t t = 0; t < rows; ++t) {
    std::memcpy(
        dst + (t * e),
        src + (static_cast<std::int64_t>(ids[static_cast<std::size_t>(t)]) * e),
        static_cast<std::size_t>(e) * sizeof(float));
  }
  return out;
}

// Pack a logical table [V, E] into the [P, E, kNr] panel layout (kernels/gemm).
[[nodiscard]] Tensor PackTable(const Tensor& w) {
  const std::int64_t v = w.shape().dim(0);
  const std::int64_t e = w.shape().dim(1);
  Tensor wp = Unwrap(Tensor::empty(Shape{k::PackedPanels(v), e, k::kNr},
                                   w.dtype(), w.device()));
  switch (w.dtype()) {
    case DataType::kBFloat16:
    case DataType::kFloat16:
      k::PackWeightPanels(reinterpret_cast<const std::uint16_t*>(w.data()), v,
                          e, reinterpret_cast<std::uint16_t*>(wp.data()));
      break;
    case DataType::kFloat32:
      k::PackWeightPanels(w.data_ptr<float>(), v, e, wp.data_ptr<float>());
      break;
    default:
      ADD_FAILURE() << "unsupported table dtype";
  }
  return wp;
}

[[nodiscard]] Tensor RunRowMajor(const Tensor& table,
                                 const std::vector<std::int32_t>& ids,
                                 std::int64_t e) {
  const auto rows = static_cast<std::int64_t>(ids.size());
  Tensor y = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  k::EmbeddingLookupF32(table.data(), table.dtype(), e, ids.data(), rows,
                        y.data_ptr<float>());
  return y;
}

[[nodiscard]] Tensor RunPacked(const Tensor& packed,
                               const std::vector<std::int32_t>& ids,
                               std::int64_t e) {
  const auto rows = static_cast<std::int64_t>(ids.size());
  Tensor y = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  k::EmbeddingLookupPackedF32(packed.data(), packed.dtype(), e, ids.data(),
                              rows, y.data_ptr<float>());
  return y;
}

void ExpectBitExact(const Tensor& got, const Tensor& want) {
  const ops::AllCloseResult r = Unwrap(ops::allclose(got, want, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Sweep both layouts vs the reference across dtypes/shapes/id sets ---

struct Case {
  std::int64_t v;
  std::int64_t e;
  const char* label;
};

TEST(EmbeddingKernelTest, BothLayoutsMatchReference) {
  // E includes a value > kGatherChunk (512) that is not a multiple of it
  // (1000), and V values with and without a padded last panel (17 pads; 512
  // and 64 are multiples of kNr).
  const Case cases[] = {
      {.v = 17, .e = 3, .label = "pad/tiny-e"},
      {.v = 64, .e = 64, .label = "no-pad"},
      {.v = 512, .e = 64, .label = "fixture-shaped"},
      {.v = 33, .e = 1000, .label = "wide-e-chunked"},
      {.v = 129, .e = 4096, .label = "large"},
  };
  const DataType dtypes[] = {DataType::kFloat32, DataType::kFloat16,
                             DataType::kBFloat16};
  std::uint64_t seed = 1;
  for (const Case& c : cases) {
    for (const DataType dt : dtypes) {
      // Random logical table in the target dtype, plus its exact fp32 cast.
      const Tensor table_f32 =
          Unwrap(ops::zeros(Shape{c.v, c.e}, DataType::kFloat32));
      ASSERT_TRUE(ops::fill_normal(table_f32, 0.0, 1.0, seed++).ok());
      const Tensor table = Unwrap(ops::cast(table_f32, dt));
      // Re-derive the exact fp32 view of the stored (possibly rounded) table,
      // so the reference reflects the bits the kernel actually reads.
      const Tensor stored_f32 = Unwrap(ops::cast(table, DataType::kFloat32));
      const Tensor packed = PackTable(table);

      // Id set: random with repeats + the edge ids {0, V-1}.
      std::vector<std::int32_t> ids = {0, static_cast<std::int32_t>(c.v - 1), 0,
                                       static_cast<std::int32_t>(c.v - 1)};
      std::uint64_t r = (seed * 2654435761U) + 12345U;
      for (int i = 0; i < 20; ++i) {
        r = (r * 6364136223846793005ULL) + 1442695040888963407ULL;
        ids.push_back(
            static_cast<std::int32_t>(r % static_cast<std::uint64_t>(c.v)));
      }

      const Tensor ref = GatherRef(stored_f32, ids);
      SCOPED_TRACE(std::string(c.label) + "/" +
                   std::string(engine::tensor::to_string(dt)));
      ExpectBitExact(RunRowMajor(table, ids, c.e), ref);
      ExpectBitExact(RunPacked(packed, ids, c.e), ref);
    }
  }
}

// --- Packed padded lane: ids near V-1 must skip the zero-pad rows ---

TEST(EmbeddingKernelTest, PackedRespectsZeroPaddedLastPanel) {
  const std::int64_t v = 17;  // last panel rows 16..31 are pad
  const std::int64_t e = 8;
  const Tensor table_f32 = Unwrap(ops::zeros(Shape{v, e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(table_f32, 0.0, 1.0, 99).ok());
  const Tensor table = Unwrap(ops::cast(table_f32, DataType::kBFloat16));
  const Tensor stored_f32 = Unwrap(ops::cast(table, DataType::kFloat32));
  const Tensor packed = PackTable(table);

  const std::vector<std::int32_t> ids = {16, 15, 16, 0, 16};  // 16 == V-1
  ExpectBitExact(RunPacked(packed, ids, e), GatherRef(stored_f32, ids));
  ExpectBitExact(RunPacked(packed, ids, e), RunRowMajor(table, ids, e));
}

// --- Special bit patterns widen exactly (the "exact widen" clause, §10) ---

TEST(EmbeddingKernelTest, HalfSpecialValuesWidenExactly) {
  // Rows carrying NaN/inf/subnormal bit patterns must widen identically to the
  // cpu oracle (which goes through tensor/half.h) — the kernel uses the M3-T06
  // convert variants, bit-exact per element including NaN payloads.
  const std::int64_t v = 4;
  const std::int64_t e = 4;
  const Tensor bf = Unwrap(ops::zeros(Shape{v, e}, DataType::kBFloat16));
  auto* bits = reinterpret_cast<std::uint16_t*>(bf.data());
  const std::uint16_t patterns[] = {
      0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001, 0x3F80, 0x0000, 0xC000,
      0x7FA5, 0x0002, 0xFFFF, 0x4049, 0x1234, 0xABCD, 0x0080, 0x7F81};
  for (std::int64_t i = 0; i < v * e; ++i) {
    bits[i] = patterns[i % 16];
  }
  const std::vector<std::int32_t> ids = {0, 1, 2, 3, 3, 0};

  // Oracle: cpu::embedding_lookup over the same bf16 table.
  Tensor oracle = Unwrap(ops::zeros(Shape{6, e}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::embedding_lookup(bf, IdsSpan(ids), oracle).ok());

  const Tensor row = RunRowMajor(bf, ids, e);
  const Tensor pak = RunPacked(PackTable(bf), ids, e);
  // Compare raw bits so NaN != NaN doesn't defeat the check.
  const auto* od = reinterpret_cast<const std::uint32_t*>(oracle.data());
  const auto* rd = reinterpret_cast<const std::uint32_t*>(row.data());
  const auto* pd = reinterpret_cast<const std::uint32_t*>(pak.data());
  for (std::int64_t i = 0; i < 6 * e; ++i) {
    EXPECT_EQ(rd[i], od[i]) << "row-major i=" << i;
    EXPECT_EQ(pd[i], od[i]) << "packed i=" << i;
  }
}

// --- Row-major agrees with the cpu oracle on the real fixture table ---

TEST(EmbeddingKernelTest, RowMajorMatchesCpuOracleOnFixture) {
  // tiny-llama embed_tokens.weight is a real bf16 [512, 64] table.
  auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const auto it = loaded->weights.find("embed_tokens.weight");
  ASSERT_NE(it, loaded->weights.end());
  const Tensor table = it->second;
  const std::int64_t e = table.shape().dim(1);

  std::vector<std::int32_t> ids;
  std::uint64_t r = 7;
  for (int i = 0; i < 40; ++i) {
    r = (r * 6364136223846793005ULL) + 1442695040888963407ULL;
    ids.push_back(static_cast<std::int32_t>(
        r % static_cast<std::uint64_t>(table.shape().dim(0))));
  }
  Tensor oracle = Unwrap(ops::zeros(
      Shape{static_cast<std::int64_t>(ids.size()), e}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::embedding_lookup(table, IdsSpan(ids), oracle).ok());

  ExpectBitExact(RunRowMajor(table, ids, e), oracle);
  ExpectBitExact(RunPacked(PackTable(table), ids, e), oracle);
}

// --- Single-token (decode-shaped) path ---

TEST(EmbeddingKernelTest, SingleTokenMatchesReference) {
  const std::int64_t v = 512;
  const std::int64_t e = 64;
  const Tensor table_f32 = Unwrap(ops::zeros(Shape{v, e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(table_f32, 0.0, 1.0, 4242).ok());
  const Tensor table = Unwrap(ops::cast(table_f32, DataType::kBFloat16));
  const Tensor stored_f32 = Unwrap(ops::cast(table, DataType::kFloat32));
  const std::vector<std::int32_t> ids = {321};
  ExpectBitExact(RunRowMajor(table, ids, e), GatherRef(stored_f32, ids));
  ExpectBitExact(RunPacked(PackTable(table), ids, e),
                 GatherRef(stored_f32, ids));
}

}  // namespace
