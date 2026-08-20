#include "engine/batch.h"

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_cache.h"
#include "model/model.h"
#include "sampling/params.h"
#include "sampling/sampler.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

// Batch-assembly tests (M9-T05, updated M9-T07; design:
// docs/design/scheduler-runtime.md §8.2/§8.4/§13). Hand-verified assembled
// metadata — token_ids / positions / cu_seqlens / caches / sample_rows — for
// prefill and decode with exact element-by-element `EXPECT_EQ`; the empty pass;
// front-loaded validation; the additive `ForwardRequest` wiring; and
// allocation-free staging after warm-up. **As of M9-T07** the assembler stages
// no block-table tensor or lengths: block growth happens inside the forward, so
// the batched decode kernel self-sources each sequence's block table + length
// from `paged_view(layer)` after the append (§8.4 as-built). Pure orchestration
// over real caches — no kernel/forward, so no SCALAR_PASS.

namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::StatusOr;
using engine::engine::BatchAssembler;
using engine::engine::BatchInputs;
using engine::engine::BatchSeqInput;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::kvcache::KvCache;
using engine::kvcache::PagedKvCache;
using engine::model::ForwardRequest;
using engine::model::LogitsMode;
using engine::sampling::Sampler;
using engine::sampling::SamplingParams;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

constexpr int kBs = 8;  // smallest valid block size (§4).

[[nodiscard]] CacheGeometry TinyGeom() {
  return CacheGeometry{.num_layers = 2,
                       .num_kv_heads = 2,
                       .head_dim = 16,
                       .dtype = DataType::kFloat32};
}

[[nodiscard]] BlockPool MakePool(std::int64_t num_blocks) {
  return Unwrap(BlockPool::Create(TinyGeom(), kBs, num_blocks, nullptr));
}

// A token-major [count, Hkv, d] fp32 block (values are irrelevant to assembly —
// only lengths/positions matter — so zeros suffice).
[[nodiscard]] Tensor ZeroKV(std::int64_t count) {
  const CacheGeometry g = TinyGeom();
  return Unwrap(
      ops::zeros(Shape{count, g.num_kv_heads, g.head_dim}, DataType::kFloat32));
}

// Append `count` tokens (all layers) to a paged cache, so its length() reflects
// a decoded sequence.
void AppendAll(PagedKvCache& cache, std::int64_t count) {
  const CacheGeometry g = TinyGeom();
  for (int layer = 0; layer < g.num_layers; ++layer) {
    ASSERT_TRUE(cache.append(layer, ZeroKV(count), ZeroKV(count)).ok());
  }
}

// A minimal greedy sampler (assembly only borrows the pointer).
[[nodiscard]] Sampler MakeSampler() {
  return Unwrap(Sampler::Create(SamplingParams::Greedy(16)));
}

// ---------------------------------------------------------------- prefill ----

// Two prefills of different lengths (fresh caches, positions [0, T)).
TEST(BatchAssemblerPrefill, TwoPrefillsDifferentLengths) {
  BlockPool pool = MakePool(16);
  PagedKvCache c0(&pool);
  PagedKvCache c1(&pool);
  std::vector<Sampler> samplers;
  samplers.reserve(2);
  samplers.push_back(MakeSampler());
  samplers.push_back(MakeSampler());

  const std::vector<std::int32_t> ids0 = {5, 6, 7};        // T=3
  const std::vector<std::int32_t> ids1 = {2, 3, 4, 8, 9};  // T=5
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = ids0,
                    .cache = &c0,
                    .sampler = samplers.data(),
                    .context = {.prompt_ids = ids0, .generated_ids = {}}},
      BatchSeqInput{.token_ids = ids1,
                    .cache = &c1,
                    .sampler = &samplers[1],
                    .context = {.prompt_ids = ids1, .generated_ids = {}}},
  };

  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssemblePrefill(seqs).ok());
  const BatchInputs& in = asm_.inputs();

  EXPECT_EQ(in.num_seqs(), 2);
  EXPECT_EQ(in.num_tokens(), 8);
  EXPECT_EQ(in.token_ids, (std::vector<std::int32_t>{5, 6, 7, 2, 3, 4, 8, 9}));
  EXPECT_EQ(in.positions, (std::vector<std::int32_t>{0, 1, 2, 0, 1, 2, 3, 4}));
  EXPECT_EQ(in.cu_seqlens, (std::vector<std::int32_t>{0, 3, 8}));
  EXPECT_EQ(in.caches, (std::vector<KvCache*>{&c0, &c1}));

  ASSERT_EQ(in.sample_rows.size(), 2U);
  EXPECT_EQ(in.sample_rows[0].sampler, samplers.data());
  EXPECT_EQ(in.sample_rows[1].sampler, &samplers[1]);
  EXPECT_EQ(in.sample_rows[0].context.prompt_ids.data(), ids0.data());
  EXPECT_EQ(in.sample_rows[1].context.prompt_ids.data(), ids1.data());
}

// Resume-style prefill: token span is prompt ++ generated into a fresh cache;
// positions still [0, T).
TEST(BatchAssemblerPrefill, ResumeContextPositionsFromZero) {
  BlockPool pool = MakePool(16);
  PagedKvCache c0(&pool);
  const Sampler s = MakeSampler();
  const std::vector<std::int32_t> prompt = {5, 6};
  const std::vector<std::int32_t> generated = {7, 8, 9};
  const std::vector<std::int32_t> feed = {5, 6, 7, 8,
                                          9};  // prompt ++ generated
  const std::vector<BatchSeqInput> seqs = {BatchSeqInput{
      .token_ids = feed,
      .cache = &c0,
      .sampler = &s,
      .context = {.prompt_ids = prompt, .generated_ids = generated}}};

  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssemblePrefill(seqs).ok());
  const BatchInputs& in = asm_.inputs();
  EXPECT_EQ(in.positions, (std::vector<std::int32_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(in.cu_seqlens, (std::vector<std::int32_t>{0, 5}));
}

// ----------------------------------------------------------------- decode ----

// Three decodes with heterogeneous cache lengths → each token flattened at its
// cache's current length (the position the batched forward feeds RoPE).
TEST(BatchAssemblerDecode, ThreeDecodesFlattenExact) {
  BlockPool pool = MakePool(16);
  PagedKvCache a(&pool);
  PagedKvCache b(&pool);
  PagedKvCache c(&pool);
  AppendAll(a, 5);
  AppendAll(b, 17);
  AppendAll(c, 24);

  std::vector<Sampler> samplers;
  samplers.reserve(3);
  for (int i = 0; i < 3; ++i) {
    samplers.push_back(MakeSampler());
  }
  const std::vector<std::int32_t> ta = {7};
  const std::vector<std::int32_t> tb = {11};
  const std::vector<std::int32_t> tc = {13};
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = ta, .cache = &a, .sampler = samplers.data()},
      BatchSeqInput{.token_ids = tb, .cache = &b, .sampler = &samplers[1]},
      BatchSeqInput{.token_ids = tc, .cache = &c, .sampler = &samplers[2]},
  };

  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssembleDecode(seqs).ok());
  const BatchInputs& in = asm_.inputs();

  EXPECT_EQ(in.num_seqs(), 3);
  EXPECT_EQ(in.token_ids, (std::vector<std::int32_t>{7, 11, 13}));
  // Each decode token sits at its cache's current length.
  EXPECT_EQ(in.positions, (std::vector<std::int32_t>{5, 17, 24}));
  EXPECT_EQ(in.cu_seqlens, (std::vector<std::int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(in.caches, (std::vector<KvCache*>{&a, &b, &c}));
}

// A single decode whose length is an exact block multiple: the token still
// flattens at the cache length (the boundary-crossing block is grown inside the
// forward, not here — §8.4 as-built).
TEST(BatchAssemblerDecode, SingleDecodePositionAtCacheLength) {
  BlockPool pool = MakePool(16);
  PagedKvCache a(&pool);
  AppendAll(a, 16);  // exactly 2 blocks, no partial tail
  const Sampler s = MakeSampler();
  const std::vector<std::int32_t> t = {3};
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = t, .cache = &a, .sampler = &s}};

  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssembleDecode(seqs).ok());
  const BatchInputs& in = asm_.inputs();
  EXPECT_EQ(in.positions, (std::vector<std::int32_t>{16}));
  EXPECT_EQ(in.cu_seqlens, (std::vector<std::int32_t>{0, 1}));
}

// ------------------------------------------------------------------ mixed ----

// The same assembler reused for a prefill then a decode fully overwrites the
// prior staging (no stale tail).
TEST(BatchAssembler, PrefillThenDecodeOverwrites) {
  BlockPool pool = MakePool(16);
  PagedKvCache p(&pool);
  PagedKvCache d(&pool);
  AppendAll(d, 10);
  const Sampler sp = MakeSampler();
  const Sampler sd = MakeSampler();

  BatchAssembler asm_;
  const std::vector<std::int32_t> pids = {1, 2, 3, 4};
  const std::vector<BatchSeqInput> prefill = {
      BatchSeqInput{.token_ids = pids, .cache = &p, .sampler = &sp}};
  ASSERT_TRUE(asm_.AssemblePrefill(prefill).ok());
  EXPECT_EQ(asm_.inputs().num_tokens(), 4);

  const std::vector<std::int32_t> dt = {9};
  const std::vector<BatchSeqInput> decode = {
      BatchSeqInput{.token_ids = dt, .cache = &d, .sampler = &sd}};
  ASSERT_TRUE(asm_.AssembleDecode(decode).ok());
  const BatchInputs& in = asm_.inputs();
  EXPECT_EQ(in.num_seqs(), 1);
  EXPECT_EQ(in.token_ids, (std::vector<std::int32_t>{9}));
  EXPECT_EQ(in.positions, (std::vector<std::int32_t>{10}));
  EXPECT_EQ(in.cu_seqlens, (std::vector<std::int32_t>{0, 1}));
}

// ------------------------------------------------------------------ empty ----

TEST(BatchAssembler, EmptyBatchIsNoOp) {
  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssemblePrefill({}).ok());
  EXPECT_EQ(asm_.inputs().num_seqs(), 0);
  EXPECT_EQ(asm_.inputs().num_tokens(), 0);
  EXPECT_EQ(asm_.inputs().cu_seqlens, (std::vector<std::int32_t>{0}));

  ASSERT_TRUE(asm_.AssembleDecode({}).ok());
  EXPECT_EQ(asm_.inputs().num_seqs(), 0);
  EXPECT_EQ(asm_.inputs().cu_seqlens, (std::vector<std::int32_t>{0}));
}

// ------------------------------------------------------------- validation ----

TEST(BatchAssembler, RejectsNullCache) {
  const Sampler s = MakeSampler();
  const std::vector<std::int32_t> t = {1};
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = t, .cache = nullptr, .sampler = &s}};
  BatchAssembler asm_;
  EXPECT_TRUE(IsInvalidArgument(asm_.AssemblePrefill(seqs)));
}

TEST(BatchAssembler, RejectsNullSampler) {
  BlockPool pool = MakePool(4);
  PagedKvCache c(&pool);
  const std::vector<std::int32_t> t = {1};
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = t, .cache = &c, .sampler = nullptr}};
  BatchAssembler asm_;
  EXPECT_TRUE(IsInvalidArgument(asm_.AssemblePrefill(seqs)));
}

TEST(BatchAssembler, RejectsEmptyTokens) {
  BlockPool pool = MakePool(4);
  PagedKvCache c(&pool);
  const Sampler s = MakeSampler();
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = {}, .cache = &c, .sampler = &s}};
  BatchAssembler asm_;
  EXPECT_TRUE(IsInvalidArgument(asm_.AssemblePrefill(seqs)));
}

TEST(BatchAssembler, DecodeRejectsMultiToken) {
  BlockPool pool = MakePool(4);
  PagedKvCache c(&pool);
  AppendAll(c, 3);
  const Sampler s = MakeSampler();
  const std::vector<std::int32_t> two = {1, 2};  // T=2 illegal for decode
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{.token_ids = two, .cache = &c, .sampler = &s}};
  BatchAssembler asm_;
  EXPECT_TRUE(IsInvalidArgument(asm_.AssembleDecode(seqs)));
}

// ------------------------------------------------------ ForwardRequest seam --

TEST(BatchAssembler, MakeForwardRequestWiresBatchFields) {
  BlockPool pool = MakePool(16);
  PagedKvCache c0(&pool);
  PagedKvCache c1(&pool);
  std::vector<Sampler> samplers;
  samplers.reserve(2);
  samplers.push_back(MakeSampler());
  samplers.push_back(MakeSampler());
  const std::vector<std::int32_t> ids0 = {5, 6, 7};
  const std::vector<std::int32_t> ids1 = {2, 3};
  const std::vector<BatchSeqInput> seqs = {
      BatchSeqInput{
          .token_ids = ids0, .cache = &c0, .sampler = samplers.data()},
      BatchSeqInput{.token_ids = ids1, .cache = &c1, .sampler = &samplers[1]},
  };

  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssemblePrefill(seqs).ok());
  const BatchInputs& in = asm_.inputs();
  const ForwardRequest req = in.MakeForwardRequest(LogitsMode::kAll);

  EXPECT_EQ(req.cache, nullptr);  // batched path uses `caches`
  EXPECT_EQ(req.logits_mode, LogitsMode::kAll);
  EXPECT_EQ(req.token_ids.data(), in.token_ids.data());
  EXPECT_EQ(req.token_ids.size(), in.token_ids.size());
  EXPECT_EQ(req.positions.data(), in.positions.data());
  EXPECT_EQ(req.cu_seqlens.data(), in.cu_seqlens.data());
  EXPECT_EQ(req.cu_seqlens.size(), 3U);
  ASSERT_EQ(req.caches.size(), 2U);
  EXPECT_EQ(req.caches[0], &c0);
  EXPECT_EQ(req.caches[1], &c1);
}

// -------------------------------------------------- allocation-free staging --

// Repeated same-size (and smaller) batches keep the staging-vector capacities
// stable after the first warm-up call — a steady-state step allocates nothing.
TEST(BatchAssembler, DecodeAllocationFreeAfterWarmup) {
  BlockPool pool = MakePool(64);
  std::vector<PagedKvCache> caches;
  caches.reserve(4);
  for (int i = 0; i < 4; ++i) {
    caches.emplace_back(&pool);
    AppendAll(caches.back(), 20);
  }
  std::vector<Sampler> samplers;
  samplers.reserve(4);
  for (int i = 0; i < 4; ++i) {
    samplers.push_back(MakeSampler());
  }
  const std::vector<std::int32_t> tok = {1};
  std::vector<BatchSeqInput> seqs;
  seqs.reserve(4);
  for (int i = 0; i < 4; ++i) {
    seqs.push_back(
        BatchSeqInput{.token_ids = tok,
                      .cache = &caches[static_cast<std::size_t>(i)],
                      .sampler = &samplers[static_cast<std::size_t>(i)]});
  }

  BatchAssembler asm_;
  ASSERT_TRUE(asm_.AssembleDecode(seqs).ok());
  const std::size_t bytes_first = asm_.staging_bytes();
  EXPECT_GT(bytes_first, 0U);

  for (int step = 0; step < 50; ++step) {
    const std::size_t n = (step % 2 == 0) ? 4U : 2U;
    ASSERT_TRUE(
        asm_.AssembleDecode(std::span<const BatchSeqInput>{seqs.data(), n})
            .ok());
  }
  EXPECT_EQ(asm_.staging_bytes(), bytes_first);
}

}  // namespace
