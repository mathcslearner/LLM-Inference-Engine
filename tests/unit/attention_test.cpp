#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "kvcache/kv_cache.h"
#include "model/loader.h"
#include "model/modules.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

// cpu::attention & the Attention module goldens and properties (M5-T05; design:
// docs/design/model-execution.md §4.2, §3.1-3.2, §6.3, §12, §13). The oracle is
// HF's LlamaAttention output (attention.safetensors): per case, the op golden
// is q_rot/k_all/v_all -> ctx (cpu::attention alone), and the module golden is
// x -> out (the whole Attention::forward, projections + RoPE + cache + o_proj).
// HF reduces in its own order, so agreement is Class T within the tolerances
// stated per test (observed max-abs-diff printed).
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::Status;
using engine::core::StatusOr;
using engine::kvcache::CacheGeometry;
using engine::kvcache::KvCache;
using engine::kvcache::KvView;
using engine::model::Attention;
using engine::model::Linear;
using engine::model::load_model;
using engine::model::LoadedModel;
using engine::model::ReferenceLinear;
using engine::model::Rope;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// tiny-llama attention dims (fixture invariants — tiny_llama.CONFIG_JSON).
constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 16;
constexpr int kHidden = 64;
constexpr float kTheta = 10000.0F;
constexpr std::int64_t kMaxPos = 128;
// The score scale HF uses: head_dim**-0.5 == 1/sqrt(16) == 0.25.
const float kScale =
    static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));

struct AttnCase {
  const char* name;
  int layer;
  int p;  // cache length before the call
  int t;  // new tokens this call
};
constexpr AttnCase kCases[] = {
    {.name = "l0_prefill_empty", .layer = 0, .p = 0, .t = 8},
    {.name = "l1_prefill_empty", .layer = 1, .p = 0, .t = 8},
    {.name = "l0_prefill_continue", .layer = 0, .p = 5, .t = 6},
    {.name = "l1_prefill_continue", .layer = 1, .p = 5, .t = 6},
    {.name = "l0_decode", .layer = 0, .p = 7, .t = 1},
};

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] SafetensorsFile AttnFile() {
  auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/attention.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

// Fresh, writable f32 copy of a (possibly mapped, read-only) fixture tensor.
[[nodiscard]] Tensor CopyOf(const Tensor& src) {
  Tensor dst = Unwrap(ops::zeros(src.shape(), DataType::kFloat32));
  EXPECT_TRUE(ops::copy(dst, src).ok());
  return dst;
}

[[nodiscard]] std::span<const std::int32_t> PosSpan(const Tensor& p) {
  return {p.data_ptr<std::int32_t>(), static_cast<std::size_t>(p.numel())};
}

// A minimal test-double KvCache (M5-T06 lands the real SimpleKvCache + the
// token-by-token KV invariant). Stores each layer's K/V as a contiguous
// head-major [Hkv, len, d] tensor, rebuilt on append; `Preload` seeds a
// non-empty cache from a head-major past block (the prefill-continuation
// cases).
class FakeKvCache final : public KvCache {
 public:
  FakeKvCache(CacheGeometry geom, std::int64_t capacity)
      : geom_(geom),
        capacity_(capacity),
        k_(static_cast<std::size_t>(geom.num_layers)),
        v_(static_cast<std::size_t>(geom.num_layers)),
        len_(static_cast<std::size_t>(geom.num_layers), 0) {}

  // Test helper: seed `layer` with head-major [Hkv, P, d] past K/V.
  void Preload(int layer, const Tensor& k_hm, const Tensor& v_hm) {
    const auto i = static_cast<std::size_t>(layer);
    k_[i] = CopyOf(k_hm);
    v_[i] = CopyOf(v_hm);
    len_[i] = k_hm.shape().dim(1);
  }

  [[nodiscard]] CacheGeometry geometry() const override { return geom_; }
  [[nodiscard]] std::int64_t length() const override { return len_[0]; }
  [[nodiscard]] std::int64_t capacity() const override { return capacity_; }
  [[nodiscard]] std::int64_t layer_length(int layer) const {
    return len_[static_cast<std::size_t>(layer)];
  }

  [[nodiscard]] Status append(int layer, const Tensor& k,
                              const Tensor& v) override {
    if (layer < 0 || layer >= geom_.num_layers) {
      return engine::core::InvalidArgumentError("FakeKvCache: layer {} OOR",
                                                layer);
    }
    const std::int64_t t = k.shape().dim(0);
    const std::int64_t hkv = k.shape().dim(1);
    const std::int64_t d = k.shape().dim(2);
    const auto i = static_cast<std::size_t>(layer);
    const std::int64_t old = len_[i];
    const std::int64_t nl = old + t;
    if (nl > capacity_) {
      return engine::core::ResourceExhaustedError(
          "FakeKvCache: append to {} exceeds capacity {}", nl, capacity_);
    }
    Tensor nk = Unwrap(ops::zeros(Shape{hkv, nl, d}, DataType::kFloat32));
    Tensor nv = Unwrap(ops::zeros(Shape{hkv, nl, d}, DataType::kFloat32));
    AppendInto(nk, k_[i], k, old, hkv, nl, d, t);
    AppendInto(nv, v_[i], v, old, hkv, nl, d, t);
    k_[i] = std::move(nk);
    v_[i] = std::move(nv);
    len_[i] = nl;
    return engine::core::OkStatus();
  }

  [[nodiscard]] StatusOr<KvView> view(int layer) const override {
    if (layer < 0 || layer >= geom_.num_layers) {
      return engine::core::InvalidArgumentError("FakeKvCache: layer {} OOR",
                                                layer);
    }
    const auto i = static_cast<std::size_t>(layer);
    return KvView{.k = k_[i], .v = v_[i]};
  }

  [[nodiscard]] Status truncate(std::int64_t new_length) override {
    if (new_length < 0 || new_length > len_[0]) {
      return engine::core::InvalidArgumentError("FakeKvCache: bad truncate {}",
                                                new_length);
    }
    for (std::size_t i = 0; i < len_.size(); ++i) {
      if (new_length == 0) {
        k_[i] = Tensor{};
        v_[i] = Tensor{};
      }
      len_[i] = new_length;
    }
    return engine::core::OkStatus();
  }

 private:
  // dst[Hkv, nl, d] = concat over len-axis of old[Hkv, old, d] (head-major) and
  // the token-major new block src[T, Hkv, d] transposed to head-major.
  static void AppendInto(Tensor& dst, const Tensor& old_t, const Tensor& src,
                         std::int64_t old, std::int64_t hkv, std::int64_t nl,
                         std::int64_t d, std::int64_t t) {
    auto* dp = dst.data_ptr<float>();
    if (old > 0) {
      const auto* op = old_t.data_ptr<float>();
      for (std::int64_t h = 0; h < hkv; ++h) {
        for (std::int64_t s = 0; s < old; ++s) {
          for (std::int64_t e = 0; e < d; ++e) {
            dp[(((h * nl) + s) * d) + e] = op[(((h * old) + s) * d) + e];
          }
        }
      }
    }
    const auto* sp = src.data_ptr<float>();
    for (std::int64_t ti = 0; ti < t; ++ti) {
      for (std::int64_t h = 0; h < hkv; ++h) {
        for (std::int64_t e = 0; e < d; ++e) {
          dp[(((h * nl) + (old + ti)) * d) + e] =
              sp[(((ti * hkv) + h) * d) + e];
        }
      }
    }
  }

  CacheGeometry geom_;
  std::int64_t capacity_;
  std::vector<Tensor> k_;
  std::vector<Tensor> v_;
  std::vector<std::int64_t> len_;
};

[[nodiscard]] std::unique_ptr<Linear> MakeLinear(Tensor w) {
  auto rl = ReferenceLinear::Create(std::move(w));
  EXPECT_TRUE(rl.ok()) << rl.status().ToString();
  return std::make_unique<ReferenceLinear>(*std::move(rl));
}

[[nodiscard]] Tensor Weight(const LoadedModel& m, const std::string& name) {
  const auto it = m.weights.find(name);
  EXPECT_NE(it, m.weights.end()) << "missing weight: " << name;
  return it->second;
}

[[nodiscard]] Attention BuildAttention(const LoadedModel& m, int layer) {
  const std::string pre = "layers." + std::to_string(layer) + ".attn.";
  Rope rope = Unwrap(Rope::Create(kHeadDim, kTheta, std::nullopt, kMaxPos));
  return Unwrap(Attention::Create(MakeLinear(Weight(m, pre + "q_proj.weight")),
                                  MakeLinear(Weight(m, pre + "k_proj.weight")),
                                  MakeLinear(Weight(m, pre + "v_proj.weight")),
                                  MakeLinear(Weight(m, pre + "o_proj.weight")),
                                  std::move(rope), kHeads, kKvHeads, kHeadDim));
}

// ===========================================================================
// Op-level goldens: cpu::attention(q_rot, k_all, v_all) == HF ctx.
// ===========================================================================

TEST(CpuAttentionTest, OpMatchesFixtureGolden) {
  const SafetensorsFile file = AttnFile();
  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-4;
  for (const AttnCase& c : kCases) {
    SCOPED_TRACE(c.name);
    const std::string base(c.name);
    const Tensor q = Unwrap(file.tensor(base + ".q_rot"));  // [T, H, d]
    const Tensor k = Unwrap(file.tensor(base + ".k_all"));  // [Hkv, L, d]
    const Tensor v = Unwrap(file.tensor(base + ".v_all"));  // [Hkv, L, d]
    const Tensor ctx = Unwrap(file.tensor(base + ".ctx"));  // [T, H, d]

    Tensor out = Unwrap(ops::zeros(ctx.shape(), DataType::kFloat32));
    const Status s = engine::cpu::attention(q, k, v, kScale, out);
    ASSERT_TRUE(s.ok()) << s.ToString();

    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ctx, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[attn] op " << c.name
              << " ctx max_abs_diff=" << r.max_abs_diff << "\n";
  }
}

// One serial attention output row (head hi, query ti) — mirrors
// cpu::attention's fp order exactly (dot ascending over d, scale after; softmax
// = max, exp, 1/sum, multiply; context ascending over s) so the op must match
// bit-for-bit. `score` is scratch of length L, reused across rows.
void SerialAttnRow(const float* qd, const float* kd, const float* vd, float* od,
                   std::int64_t hi, std::int64_t ti, std::int64_t heads,
                   std::int64_t hk, std::int64_t l, std::int64_t d,
                   std::int64_t limit, float scale, float* score) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  const float* qv = qd + (((ti * heads) + hi) * d);
  for (std::int64_t s = 0; s < l; ++s) {
    if (s > limit) {
      score[s] = kNegInf;
      continue;
    }
    const float* kv = kd + (((hk * l) + s) * d);
    float dot = 0.0F;
    for (std::int64_t e = 0; e < d; ++e) {
      dot += qv[e] * kv[e];
    }
    score[s] = dot * scale;
  }
  float m = score[0];
  for (std::int64_t s = 1; s < l; ++s) {
    m = std::max(m, score[s]);
  }
  float sum = 0.0F;
  for (std::int64_t s = 0; s < l; ++s) {
    const float ex = std::exp(score[s] - m);
    score[s] = ex;
    sum += ex;
  }
  const float inv = 1.0F / sum;
  for (std::int64_t s = 0; s < l; ++s) {
    score[s] *= inv;
  }
  float* ov = od + (((ti * heads) + hi) * d);
  for (std::int64_t e = 0; e < d; ++e) {
    float acc = 0.0F;
    for (std::int64_t s = 0; s < l; ++s) {
      acc += score[s] * vd[(((hk * l) + s) * d) + e];
    }
    ov[e] = acc;
  }
}

// Full single-threaded attention oracle over [T,H,d] q and [Hkv,L,d] k/v.
[[nodiscard]] Tensor SerialAttention(const Tensor& q, const Tensor& k,
                                     const Tensor& v, float scale) {
  const std::int64_t t = q.shape().dim(0);
  const std::int64_t h = q.shape().dim(1);
  const std::int64_t d = q.shape().dim(2);
  const std::int64_t hkv = k.shape().dim(0);
  const std::int64_t l = k.shape().dim(1);
  const std::int64_t g = h / hkv;
  const std::int64_t past = l - t;
  Tensor out = Unwrap(ops::zeros(Shape{t, h, d}, DataType::kFloat32));
  const auto* qd = q.data_ptr<float>();
  const auto* kd = k.data_ptr<float>();
  const auto* vd = v.data_ptr<float>();
  auto* od = out.data_ptr<float>();
  std::vector<float> score(static_cast<std::size_t>(l));
  for (std::int64_t hi = 0; hi < h; ++hi) {
    const std::int64_t hk = hi / g;
    for (std::int64_t ti = 0; ti < t; ++ti) {
      SerialAttnRow(qd, kd, vd, od, hi, ti, h, hk, l, d, past + ti, scale,
                    score.data());
    }
  }
  return out;
}

// Threading independence + math: the op equals a single-threaded
// reimplementation with the SAME ascending accumulation order — bit-identical
// (atol=rtol=0), so the parallel partition changes no bits.
TEST(CpuAttentionTest, OpIsBitExactVsSerial) {
  const SafetensorsFile file = AttnFile();
  const Tensor q =
      Unwrap(file.tensor("l1_prefill_continue.q_rot"));  // [6,4,16]
  const Tensor k =
      Unwrap(file.tensor("l1_prefill_continue.k_all"));  // [2,11,16]
  const Tensor v = Unwrap(file.tensor("l1_prefill_continue.v_all"));

  Tensor got = Unwrap(ops::zeros(q.shape(), DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::attention(q, k, v, kScale, got).ok());

  const Tensor naive = SerialAttention(q, k, v, kScale);
  const ops::AllCloseResult r = Unwrap(ops::allclose(got, naive, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// The causal mask: a future key/value must not influence an earlier query.
TEST(CpuAttentionTest, OpCausalMaskHidesFutureKeys) {
  // T=2, H=Hkv=1, d=2, L=2 (P=0): query 0 sees only key 0; query 1 sees both.
  const Tensor q = Unwrap(ops::zeros(Shape{2, 1, 2}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(q, 0.0, 1.0, 1).ok());
  const Tensor k = Unwrap(ops::zeros(Shape{1, 2, 2}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(k, 0.0, 1.0, 2).ok());
  const Tensor v = Unwrap(ops::zeros(Shape{1, 2, 2}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(v, 0.0, 1.0, 3).ok());

  Tensor base = Unwrap(ops::zeros(Shape{2, 1, 2}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::attention(q, k, v, kScale, base).ok());

  // Perturb the future position (index 1) of K and V.
  const Tensor k2 = CopyOf(k);
  const Tensor v2 = CopyOf(v);
  k2.data_ptr<float>()[(1 * 2) + 0] += 5.0F;  // k[0,1,0]
  v2.data_ptr<float>()[(1 * 2) + 1] -= 7.0F;  // v[0,1,1]
  Tensor pert = Unwrap(ops::zeros(Shape{2, 1, 2}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::attention(q, k2, v2, kScale, pert).ok());

  // Query 0 (row 0) is unchanged; query 1 (row 1) changes.
  EXPECT_EQ(base.data_ptr<float>()[0], pert.data_ptr<float>()[0]);
  EXPECT_EQ(base.data_ptr<float>()[1], pert.data_ptr<float>()[1]);
  EXPECT_NE(base.data_ptr<float>()[2], pert.data_ptr<float>()[2]);
}

// GQA head indexing: query head h reads kv head h/g (repeat_interleave), NOT
// h % Hkv (tiling). With L==1 the softmax is 1, so out[t,h,:] == v[h/g, 0, :].
TEST(CpuAttentionTest, OpGqaUsesInterleavedKvHead) {
  // H=4, Hkv=2, g=2, T=1, L=1, d=2. v head 0 and head 1 are distinct.
  const Tensor q = Unwrap(ops::ones(Shape{1, 4, 2}, DataType::kFloat32));
  const Tensor k = Unwrap(ops::ones(Shape{2, 1, 2}, DataType::kFloat32));
  const Tensor v = Unwrap(ops::zeros(Shape{2, 1, 2}, DataType::kFloat32));
  auto* vd = v.data_ptr<float>();
  vd[0] = 10.0F;
  vd[1] = 11.0F;  // v head 0
  vd[2] = 20.0F;
  vd[3] = 21.0F;  // v head 1

  Tensor out = Unwrap(ops::zeros(Shape{1, 4, 2}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::attention(q, k, v, kScale, out).ok());
  const auto* od = out.data_ptr<float>();
  // Heads 0,1 -> kv 0 (10,11); heads 2,3 -> kv 1 (20,21). h % Hkv would give
  // head 1 -> kv 1, so this distinguishes interleave from tiling.
  EXPECT_FLOAT_EQ(od[0], 10.0F);
  EXPECT_FLOAT_EQ(od[1], 11.0F);  // h0
  EXPECT_FLOAT_EQ(od[2], 10.0F);
  EXPECT_FLOAT_EQ(od[3], 11.0F);  // h1
  EXPECT_FLOAT_EQ(od[4], 20.0F);
  EXPECT_FLOAT_EQ(od[5], 21.0F);  // h2
  EXPECT_FLOAT_EQ(od[6], 20.0F);
  EXPECT_FLOAT_EQ(od[7], 21.0F);  // h3
}

TEST(CpuAttentionTest, OpRejectsMalformedInputs) {
  const Tensor q = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat32));
  const Tensor k = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat32));
  const Tensor v = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat32));
  Tensor out = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat32));

  // q not rank-3.
  {
    const Tensor bad = Unwrap(ops::zeros(Shape{2, 4}, DataType::kFloat32));
    const Status s = engine::cpu::attention(bad, k, v, kScale, out);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("q must be rank-3"), std::string::npos)
        << s.message();
  }
  // q not f32.
  {
    const Tensor bad = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat16));
    const Status s = engine::cpu::attention(bad, k, v, kScale, out);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("q must be f32"), std::string::npos)
        << s.message();
  }
  // H not a multiple of Hkv (q H=2, k Hkv=... make H%Hkv!=0 via H=3).
  {
    const Tensor q3 = Unwrap(ops::zeros(Shape{2, 3, 4}, DataType::kFloat32));
    const Tensor k2 = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat32));
    const Tensor v2 = Unwrap(ops::zeros(Shape{2, 2, 4}, DataType::kFloat32));
    Tensor o3 = Unwrap(ops::zeros(Shape{2, 3, 4}, DataType::kFloat32));
    const Status s = engine::cpu::attention(q3, k2, v2, kScale, o3);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("multiple of Hkv"), std::string::npos)
        << s.message();
  }
  // d mismatch between q and k.
  {
    const Tensor k8 = Unwrap(ops::zeros(Shape{2, 2, 8}, DataType::kFloat32));
    const Tensor v8 = Unwrap(ops::zeros(Shape{2, 2, 8}, DataType::kFloat32));
    const Status s = engine::cpu::attention(q, k8, v8, kScale, out);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("k must be [Hkv"), std::string::npos)
        << s.message();
  }
  // v shape disagrees with k.
  {
    const Tensor v2 = Unwrap(ops::zeros(Shape{2, 2, 8}, DataType::kFloat32));
    const Status s = engine::cpu::attention(q, k, v2, kScale, out);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("v must match k"), std::string::npos)
        << s.message();
  }
  // Cache length L < T.
  {
    const Tensor q3 = Unwrap(ops::zeros(Shape{3, 2, 4}, DataType::kFloat32));
    Tensor o3 = Unwrap(ops::zeros(Shape{3, 2, 4}, DataType::kFloat32));
    const Status s = engine::cpu::attention(q3, k, v, kScale, o3);  // L=2 < T=3
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("must be >= T"), std::string::npos)
        << s.message();
  }
  // out wrong shape.
  {
    Tensor bad = Unwrap(ops::zeros(Shape{2, 2, 8}, DataType::kFloat32));
    const Status s = engine::cpu::attention(q, k, v, kScale, bad);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("out must be"), std::string::npos)
        << s.message();
  }
}

// ===========================================================================
// Module goldens: Attention::forward(x) == HF out, over the loaded checkpoint.
// ===========================================================================

TEST(AttentionModuleTest, MatchesFixtureGolden) {
  const auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const SafetensorsFile file = AttnFile();
  const CacheGeometry geom{.num_layers = 2,
                           .num_kv_heads = kKvHeads,
                           .head_dim = kHeadDim,
                           .dtype = DataType::kFloat32};
  constexpr double kRtol = 2e-4;
  constexpr double kAtol = 2e-4;

  for (const AttnCase& c : kCases) {
    SCOPED_TRACE(c.name);
    const std::string base(c.name);
    const Attention attn = BuildAttention(*loaded, c.layer);

    FakeKvCache cache(geom, kMaxPos);
    if (c.p > 0) {
      cache.Preload(c.layer, Unwrap(file.tensor(base + ".past_k")),
                    Unwrap(file.tensor(base + ".past_v")));
    }

    const Tensor x = CopyOf(Unwrap(file.tensor(base + ".x")));  // [T, E]
    const Tensor positions = Unwrap(file.tensor(base + ".positions"));
    const Tensor expected = Unwrap(file.tensor(base + ".out"));  // [T, E]

    Tensor y = Unwrap(ops::zeros(Shape{c.t, kHidden}, DataType::kFloat32));
    const Status s = attn.forward(x, PosSpan(positions), c.layer, cache, y);
    ASSERT_TRUE(s.ok()) << s.ToString();

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(y, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << r.Summary();
    std::cerr << "[attn] module " << c.name
              << " out max_abs_diff=" << r.max_abs_diff << "\n";

    // The accumulated cache view equals HF's k_all/v_all, and its length is
    // P+T (§6.2's accumulation, at op scope). This is a *wiring* check — past
    // preserved, new K/V appended in order — so it carries the raw projection
    // (+RoPE) Class-T spread (the GEMM output before o_proj/softmax average it
    // down), a looser band than the final `out`. Observed worst ~6e-4.
    constexpr double kCacheTol = 1e-3;
    const KvView kv = Unwrap(cache.view(c.layer));
    EXPECT_EQ(kv.k.shape().dim(1), c.p + c.t);
    EXPECT_EQ(cache.layer_length(c.layer), c.p + c.t);
    const Tensor k_all = Unwrap(file.tensor(base + ".k_all"));
    const Tensor v_all = Unwrap(file.tensor(base + ".v_all"));
    const ops::AllCloseResult rk =
        Unwrap(ops::allclose(kv.k, k_all, kCacheTol, kCacheTol));
    const ops::AllCloseResult rv =
        Unwrap(ops::allclose(kv.v, v_all, kCacheTol, kCacheTol));
    EXPECT_TRUE(rk.allclose) << "k_all: " << rk.Summary();
    EXPECT_TRUE(rv.allclose) << "v_all: " << rv.Summary();
  }
}

// ===========================================================================
// Attention::Create validation and forward error paths.
// ===========================================================================

// Small synthetic projections: E=6, H=2, Hkv=1, d=4.
[[nodiscard]] std::unique_ptr<Linear> ZeroLinear(std::int64_t out,
                                                 std::int64_t in) {
  return MakeLinear(Unwrap(ops::zeros(Shape{out, in}, DataType::kFloat32)));
}

TEST(AttentionModuleTest, CreateValidatesShapes) {
  constexpr int h = 2;
  constexpr int hkv = 1;
  constexpr int d = 4;
  constexpr std::int64_t e = 6;
  const std::int64_t q_rows = static_cast<std::int64_t>(h) * d;     // 8
  const std::int64_t kv_rows = static_cast<std::int64_t>(hkv) * d;  // 4
  auto rope = [&] {
    return Unwrap(Rope::Create(d, kTheta, std::nullopt, kMaxPos));
  };

  // Well-formed baseline succeeds.
  EXPECT_TRUE(Attention::Create(ZeroLinear(q_rows, e), ZeroLinear(kv_rows, e),
                                ZeroLinear(kv_rows, e), ZeroLinear(e, q_rows),
                                rope(), h, hkv, d)
                  .ok());

  // Null projection.
  {
    const auto r = Attention::Create(nullptr, ZeroLinear(kv_rows, e),
                                     ZeroLinear(kv_rows, e),
                                     ZeroLinear(e, q_rows), rope(), h, hkv, d);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("non-null"), std::string::npos)
        << r.status().message();
  }
  // H not a multiple of Hkv.
  {
    const std::int64_t three_d = static_cast<std::int64_t>(3) * d;
    const auto r = Attention::Create(
        ZeroLinear(three_d, e), ZeroLinear(kv_rows, e), ZeroLinear(kv_rows, e),
        ZeroLinear(e, three_d), rope(), 3, hkv * 2, d);  // 3 % 2 != 0
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("multiple of num_kv_heads"),
              std::string::npos)
        << r.status().message();
  }
  // q_proj out_features wrong.
  {
    const auto r = Attention::Create(
        ZeroLinear(q_rows + 1, e), ZeroLinear(kv_rows, e),
        ZeroLinear(kv_rows, e), ZeroLinear(e, q_rows), rope(), h, hkv, d);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("q_proj out_features"),
              std::string::npos)
        << r.status().message();
  }
  // o_proj in_features wrong.
  {
    const auto r = Attention::Create(
        ZeroLinear(q_rows, e), ZeroLinear(kv_rows, e), ZeroLinear(kv_rows, e),
        ZeroLinear(e, q_rows + 1), rope(), h, hkv, d);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("o_proj in_features"),
              std::string::npos)
        << r.status().message();
  }
  // rope head_dim mismatch.
  {
    const auto r = Attention::Create(
        ZeroLinear(q_rows, e), ZeroLinear(kv_rows, e), ZeroLinear(kv_rows, e),
        ZeroLinear(e, q_rows),
        Unwrap(Rope::Create(d + 2, kTheta, std::nullopt, kMaxPos)), h, hkv, d);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("rope head_dim"), std::string::npos)
        << r.status().message();
  }
}

TEST(AttentionModuleTest, ForwardRejectsMalformedInputs) {
  constexpr int h = 2;
  constexpr int hkv = 1;
  constexpr int d = 4;
  constexpr std::int64_t e = 6;
  const CacheGeometry geom{.num_layers = 1,
                           .num_kv_heads = hkv,
                           .head_dim = d,
                           .dtype = DataType::kFloat32};
  const std::int64_t q_rows = static_cast<std::int64_t>(h) * d;
  const std::int64_t kv_rows = static_cast<std::int64_t>(hkv) * d;
  const Attention attn = Unwrap(Attention::Create(
      ZeroLinear(q_rows, e), ZeroLinear(kv_rows, e), ZeroLinear(kv_rows, e),
      ZeroLinear(e, q_rows),
      Unwrap(Rope::Create(d, kTheta, std::nullopt, kMaxPos)), h, hkv, d));

  const std::vector<std::int32_t> pos2 = {0, 1};

  // x wrong hidden dim.
  {
    FakeKvCache cache(geom, kMaxPos);
    const Tensor x = Unwrap(ops::zeros(Shape{2, e + 1}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, e}, DataType::kFloat32));
    const Status s = attn.forward(x, pos2, 0, cache, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be"), std::string::npos) << s.message();
  }
  // positions length != T.
  {
    FakeKvCache cache(geom, kMaxPos);
    const Tensor x = Unwrap(ops::zeros(Shape{2, e}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, e}, DataType::kFloat32));
    const std::vector<std::int32_t> pos1 = {0};
    const Status s = attn.forward(x, pos1, 0, cache, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("positions length"), std::string::npos)
        << s.message();
  }
  // y wrong shape.
  {
    FakeKvCache cache(geom, kMaxPos);
    const Tensor x = Unwrap(ops::zeros(Shape{2, e}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, e + 1}, DataType::kFloat32));
    const Status s = attn.forward(x, pos2, 0, cache, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be caller-allocated"), std::string::npos)
        << s.message();
  }
  // Over-capacity append surfaces the cache's ResourceExhausted.
  {
    FakeKvCache cache(geom, /*capacity=*/1);
    const Tensor x = Unwrap(ops::zeros(Shape{2, e}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, e}, DataType::kFloat32));
    const Status s = attn.forward(x, pos2, 0, cache, y);  // T=2 > cap 1
    EXPECT_TRUE(IsResourceExhausted(s)) << s.ToString();
  }
}

// ===========================================================================
// FakeKvCache self-check: the test double round-trips append -> view.
// ===========================================================================

TEST(FakeKvCacheTest, AppendThenViewIsHeadMajor) {
  const CacheGeometry geom{.num_layers = 1,
                           .num_kv_heads = 2,
                           .head_dim = 3,
                           .dtype = DataType::kFloat32};
  FakeKvCache cache(geom, 8);
  // Two appends: T=2 then T=1, token-major [T, Hkv, d].
  const Tensor k1 = Unwrap(ops::zeros(Shape{2, 2, 3}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(k1, 0.0, 1.0, 7).ok());
  const Tensor v1 = CopyOf(k1);
  ASSERT_TRUE(cache.append(0, k1, v1).ok());
  const Tensor k2 = Unwrap(ops::zeros(Shape{1, 2, 3}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(k2, 0.0, 1.0, 8).ok());
  const Tensor v2 = CopyOf(k2);
  ASSERT_TRUE(cache.append(0, k2, v2).ok());

  const KvView kv = Unwrap(cache.view(0));
  ASSERT_EQ(kv.k.shape().dim(0), 2);  // Hkv
  ASSERT_EQ(kv.k.shape().dim(1), 3);  // len = 2 + 1
  ASSERT_EQ(kv.k.shape().dim(2), 3);  // d
  // Head-major: kv.k[h, s, e] == source token-major value.
  for (std::int64_t h = 0; h < 2; ++h) {
    for (std::int64_t e = 0; e < 3; ++e) {
      EXPECT_FLOAT_EQ((kv.k.item<float>({h, 0, e})),
                      (k1.item<float>({0, h, e})));
      EXPECT_FLOAT_EQ((kv.k.item<float>({h, 1, e})),
                      (k1.item<float>({1, h, e})));
      EXPECT_FLOAT_EQ((kv.k.item<float>({h, 2, e})),
                      (k2.item<float>({0, h, e})));
    }
  }
  EXPECT_EQ(cache.length(), 3);
  ASSERT_TRUE(cache.truncate(0).ok());
  EXPECT_EQ(cache.length(), 0);
}

}  // namespace
