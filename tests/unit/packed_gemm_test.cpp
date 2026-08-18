#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "kernels/dispatch.h"
#include "kernels/gemm.h"
#include "kernels/internal/gemm_impl.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Packed-weight GEMM/GEMV kernels (M6-T02; design:
// docs/design/optimized-cpu-execution.md §3, §10). Validated against the
// cpu::gemm oracle (which M5 validated against HuggingFace). Registered
// SCALAR_PASS (§9): the forced-scalar pass proves the scalar variant on both
// hosts, and the scalar packed kernel is bit-identical to a naive triple loop
// (single fp32 accumulator, ascending k, half.h widening), so the "vs oracle"
// numbers are exact under scalar and Class T under NEON/AVX2.
namespace {

namespace ops = engine::tensor::ops;
namespace k = engine::kernels;
using engine::core::StatusOr;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

// Pack a checkpoint-order weight [N, K] into the [P, K, kNr] panel layout via
// the public pack routine (dispatching on storage dtype).
[[nodiscard]] Tensor PackWeight(const Tensor& w) {
  const std::int64_t n = w.shape().dim(0);
  const std::int64_t kk = w.shape().dim(1);
  const std::int64_t panels = k::PackedPanels(n);
  Tensor wp =
      Unwrap(Tensor::empty(Shape{panels, kk, k::kNr}, w.dtype(), w.device()));
  switch (w.dtype()) {
    case DataType::kBFloat16:
    case DataType::kFloat16:
      k::PackWeightPanels(reinterpret_cast<const std::uint16_t*>(w.data()), n,
                          kk, reinterpret_cast<std::uint16_t*>(wp.data()));
      break;
    case DataType::kFloat32:
      k::PackWeightPanels(w.data_ptr<float>(), n, kk, wp.data_ptr<float>());
      break;
    default:
      ADD_FAILURE() << "unsupported weight dtype";
  }
  return wp;
}

// The variant Select would return for the active ISA, called on one giant tile
// [0,m)×[0,P) — the same per-element FMA sequence PackedGemm produces, just
// unthreaded and un-tiled. Bit-identical to PackedGemm for any ISA (the MR
// grouping and tile partition never change which k-order a given output sums).
void SingleTileReference(const float* x, std::int64_t m, std::int64_t kk,
                         const Tensor& wp, DataType dt, std::int64_t n,
                         const float* bias, float* y) {
  const std::int64_t panels = k::PackedPanels(n);
  const k::Isa isa = k::SelectedIsa();
  switch (dt) {
    case DataType::kBFloat16:
      k::detail::Bf16TileVariant(isa)(
          x, kk, n, reinterpret_cast<const std::uint16_t*>(wp.data()), bias, y,
          0, m, 0, panels);
      break;
    case DataType::kFloat16:
      k::detail::F16TileVariant(isa)(
          x, kk, n, reinterpret_cast<const std::uint16_t*>(wp.data()), bias, y,
          0, m, 0, panels);
      break;
    case DataType::kFloat32:
      k::detail::F32TileVariant(isa)(x, kk, n, wp.data_ptr<float>(), bias, y, 0,
                                     m, 0, panels);
      break;
    default:
      ADD_FAILURE() << "unsupported weight dtype";
  }
}

// --- Pack layout: Wp[p, k, r] == W[p*kNr + r, k], zero-padded (§3.2) ---

TEST(PackedGemmTest, PackLayoutIsCorrectWithZeroPad) {
  // N not a multiple of kNr so the last panel pads (rows 17..31 zero).
  const std::int64_t n = 17;
  const std::int64_t kk = 5;
  const Tensor w = Unwrap(ops::zeros(Shape{n, kk}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w, 0.0, 1.0, 7).ok());
  const Tensor wp = PackWeight(w);

  const std::int64_t panels = k::PackedPanels(n);
  ASSERT_EQ(wp.shape().dim(0), panels);
  ASSERT_EQ(wp.shape().dim(1), kk);
  ASSERT_EQ(wp.shape().dim(2), k::kNr);

  const auto* wd = w.data_ptr<float>();
  const auto* pd = wp.data_ptr<float>();
  for (std::int64_t p = 0; p < panels; ++p) {
    for (std::int64_t kc = 0; kc < kk; ++kc) {
      for (std::int64_t r = 0; r < k::kNr; ++r) {
        const std::int64_t row = (p * k::kNr) + r;
        const float got = pd[(((p * kk) + kc) * k::kNr) + r];
        const float want = row < n ? wd[(row * kk) + kc] : 0.0F;
        EXPECT_FLOAT_EQ(got, want)
            << "p=" << p << " k=" << kc << " r=" << r << " row=" << row;
      }
    }
  }
}

// --- PackedGemm vs cpu::gemm across model shapes & dtypes (Class T, §10) ---

struct Shape3 {
  std::int64_t m;
  std::int64_t k;
  std::int64_t n;
  const char* label;
};

// Projection-like shapes (tiny-llama/qwen2 dims + 1B-class K/N + tails),
// decode (m=1), skinny (m<=MR), wide, k=1, odd N forcing panel pad.
constexpr Shape3 kShapes[] = {
    {.m = 16, .k = 64, .n = 64, .label = "qkv_tiny"},
    {.m = 16, .k = 64, .n = 96, .label = "qkv_qwen_headdim"},
    {.m = 16, .k = 176, .n = 64, .label = "down_tiny"},
    {.m = 8, .k = 64, .n = 512, .label = "lm_head_wide"},
    {.m = 1, .k = 64, .n = 64, .label = "decode_gemv"},
    {.m = 1, .k = 2048, .n = 512, .label = "decode_gemv_1b"},
    {.m = 4, .k = 2048, .n = 8192, .label = "gate_1b_skinny"},
    {.m = 33, .k = 8192, .n = 256, .label = "down_1b"},
    {.m = 130, .k = 300, .n = 70, .label = "multi_tile"},
    {.m = 7, .k = 1, .n = 33, .label = "k_one_oddN"},
    {.m = 65, .k = 129, .n = 1000, .label = "odd_tail_vocab"},
    {.m = 5, .k = 40, .n = 17, .label = "tiny_pad"},
};

void RunShapeVsOracle(const Shape3& s, DataType wdt, bool with_bias) {
  std::uint64_t seed = 100 + static_cast<std::uint64_t>(s.k + s.n + s.m);
  const Tensor x = Unwrap(ops::zeros(Shape{s.m, s.k}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, seed++).ok());

  // Build the weight in fp32 then cast to the storage dtype so the oracle and
  // the packed kernel read the *same* stored bits.
  const Tensor w_f32 = Unwrap(ops::zeros(Shape{s.n, s.k}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w_f32, 0.0, 1.0, seed++).ok());
  const Tensor w =
      wdt == DataType::kFloat32 ? w_f32 : Unwrap(ops::cast(w_f32, wdt));

  Tensor bias;
  const Tensor* bias_ptr = nullptr;
  if (with_bias) {
    bias = Unwrap(ops::zeros(Shape{s.n}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(bias, 0.0, 1.0, seed++).ok());
    bias_ptr = &bias;
  }

  Tensor ref = Unwrap(ops::zeros(Shape{s.m, s.n}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::gemm(x, w, bias_ptr, ref).ok());

  const Tensor wp = PackWeight(w);
  const Tensor out = Unwrap(ops::zeros(Shape{s.m, s.n}, DataType::kFloat32));
  const float* bias_data =
      bias_ptr != nullptr ? bias.data_ptr<float>() : nullptr;
  k::PackedGemm(x.data_ptr<float>(), s.m, s.k, wp.data(), wdt, s.n, bias_data,
                out.data_ptr<float>());

  // GEMM tolerance (§10): atol scaled by sqrt(K) (a longer dot admits more
  // rounding). Scalar is exact (bit-identical to the oracle); NEON/AVX2 land
  // well inside this band.
  const double atol = 1e-5 * std::sqrt(static_cast<double>(s.k));
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, 1e-4, atol));
  EXPECT_TRUE(r.allclose) << s.label
                          << " dtype=" << engine::tensor::to_string(wdt)
                          << " bias=" << with_bias << ": " << r.Summary();
}

TEST(PackedGemmTest, MatchesOracleAcrossShapesBf16) {
  for (const Shape3& s : kShapes) {
    RunShapeVsOracle(s, DataType::kBFloat16, /*with_bias=*/false);
    RunShapeVsOracle(s, DataType::kBFloat16, /*with_bias=*/true);
  }
}

TEST(PackedGemmTest, MatchesOracleAcrossShapesF16) {
  for (const Shape3& s : kShapes) {
    RunShapeVsOracle(s, DataType::kFloat16, /*with_bias=*/false);
  }
}

TEST(PackedGemmTest, MatchesOracleAcrossShapesF32) {
  for (const Shape3& s : kShapes) {
    RunShapeVsOracle(s, DataType::kFloat32, /*with_bias=*/false);
    RunShapeVsOracle(s, DataType::kFloat32, /*with_bias=*/true);
  }
}

// --- Tiling/threading invariance: PackedGemm == the un-tiled variant
// (bit-exact) ---

TEST(PackedGemmTest, TilingAndThreadingAreBitExactVsSingleTile) {
  const Shape3 shapes[] = {
      {.m = 130, .k = 300, .n = 70, .label = "multi_tile"},
      {.m = 200, .k = 512, .n = 519, .label = "many_panels"},
      {.m = 1, .k = 4096, .n = 300, .label = "gemv"},
  };
  for (const Shape3& s : shapes) {
    std::uint64_t seed = 7000 + static_cast<std::uint64_t>(s.m);
    const Tensor x = Unwrap(ops::zeros(Shape{s.m, s.k}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, seed++).ok());
    const Tensor w_f32 =
        Unwrap(ops::zeros(Shape{s.n, s.k}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(w_f32, 0.0, 1.0, seed++).ok());
    const Tensor w = Unwrap(ops::cast(w_f32, DataType::kBFloat16));
    const Tensor wp = PackWeight(w);

    const Tensor threaded =
        Unwrap(ops::zeros(Shape{s.m, s.n}, DataType::kFloat32));
    k::PackedGemm(x.data_ptr<float>(), s.m, s.k, wp.data(), DataType::kBFloat16,
                  s.n, nullptr, threaded.data_ptr<float>());

    const Tensor single =
        Unwrap(ops::zeros(Shape{s.m, s.n}, DataType::kFloat32));
    SingleTileReference(x.data_ptr<float>(), s.m, s.k, wp, DataType::kBFloat16,
                        s.n, nullptr, single.data_ptr<float>());

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(threaded, single, 0.0, 0.0));
    EXPECT_TRUE(r.allclose) << s.label << ": " << r.Summary();
  }
}

// --- The decode GEMV path matches the prefill GEMM row-for-row (bit-exact) ---

TEST(PackedGemmTest, GemvMatchesGemmRow) {
  const std::int64_t m = 12;
  const std::int64_t kk = 320;
  const std::int64_t n = 200;
  const Tensor x = Unwrap(ops::zeros(Shape{m, kk}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 42).ok());
  const Tensor w_f32 = Unwrap(ops::zeros(Shape{n, kk}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w_f32, 0.0, 1.0, 43).ok());
  const Tensor w = Unwrap(ops::cast(w_f32, DataType::kBFloat16));
  const Tensor wp = PackWeight(w);

  const Tensor gemm_out = Unwrap(ops::zeros(Shape{m, n}, DataType::kFloat32));
  k::PackedGemm(x.data_ptr<float>(), m, kk, wp.data(), DataType::kBFloat16, n,
                nullptr, gemm_out.data_ptr<float>());

  for (std::int64_t row = 0; row < m; ++row) {
    const Tensor gemv_out = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
    k::PackedGemv(x.data_ptr<float>() + (row * kk), kk, wp.data(),
                  DataType::kBFloat16, n, nullptr, gemv_out.data_ptr<float>());
    const auto* g = gemm_out.data_ptr<float>() + (row * n);
    const auto* v = gemv_out.data_ptr<float>();
    for (std::int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(g[j], v[j]) << "row=" << row << " j=" << j;
    }
  }
}

// --- The build's vector slot is actually populated (M3-audit vacuity guard)
// ---

TEST(PackedGemmTest, VectorVariantSlotPopulated) {
  const k::Isa isa = k::SelectedIsa();
  if (isa == k::Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass: no vector variant expected";
  }
  // On a host with a vector ISA the selected tile variant must not be the
  // scalar one — otherwise every bit-compare above would be scalar-vs-scalar.
  EXPECT_NE(reinterpret_cast<void*>(k::detail::Bf16TileVariant(isa)),
            reinterpret_cast<void*>(&k::scalar::GemmTileBf16));
  EXPECT_NE(reinterpret_cast<void*>(k::detail::F32TileVariant(isa)),
            reinterpret_cast<void*>(&k::scalar::GemmTileF32));
}

// --- ops.safetensors GEMM goldens replayed through the packed kernel ---

TEST(PackedGemmTest, MatchesFixtureGoldens) {
  const auto file =
      engine::model::SafetensorsFile::Open(engine::testing::FixturesDir() /
                                           "models/tiny-llama/expected/"
                                           "ops.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();

  struct Case {
    const char* name;
    bool has_bias;
  };
  constexpr Case cases[] = {
      {.name = "q_proj_prefill", .has_bias = false},
      {.name = "gate_proj_wide", .has_bias = false},
      {.name = "down_proj", .has_bias = false},
      {.name = "lm_head_wide", .has_bias = false},
      {.name = "decode_gemv", .has_bias = false},
      {.name = "skinny_tall", .has_bias = false},
      {.name = "k_one", .has_bias = false},
      {.name = "odd_tails", .has_bias = true},
      {.name = "fp16_weight", .has_bias = false},
      {.name = "bf16_bias", .has_bias = true},
  };

  for (const Case& c : cases) {
    const std::string base(c.name);
    const Tensor x = Unwrap(file->tensor(base + ".a"));
    const Tensor w = Unwrap(file->tensor(base + ".b"));
    const Tensor expected = Unwrap(file->tensor(base + ".c"));
    Tensor bias_f32;
    const float* bias_data = nullptr;
    if (c.has_bias) {
      bias_f32 = Unwrap(
          ops::cast(Unwrap(file->tensor(base + ".bias")), DataType::kFloat32));
      bias_data = bias_f32.data_ptr<float>();
    }
    const std::int64_t m = x.shape().dim(0);
    const std::int64_t kk = x.shape().dim(1);
    const std::int64_t n = w.shape().dim(0);

    const Tensor wp = PackWeight(w);
    const Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
    k::PackedGemm(x.data_ptr<float>(), m, kk, wp.data(), w.dtype(), n,
                  bias_data, out.data_ptr<float>());

    const double atol = 1e-4 + (1e-5 * std::sqrt(static_cast<double>(kk)));
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, 1e-4, atol));
    EXPECT_TRUE(r.allclose) << c.name << ": " << r.Summary();
  }
}

}  // namespace
