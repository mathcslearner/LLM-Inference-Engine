#include "runtime/request.h"

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "runtime/channel.h"
#include "sampling/params.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// Request / Sequence tests (M9-T02; design: docs/design/scheduler-runtime.md
// §3). The state machine (every legal transition succeeds, every illegal one
// CHECK-fails), sequence construction (front-loaded sampler/stop validation,
// resolved seed), the progress bookkeeping, and RAII block reclamation. Pure
// bookkeeping over a real BlockPool — no model, kernel, or forward, so no
// SCALAR_PASS. `runtime` label.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::StatusOr;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::runtime::FinishInfo;
using engine::runtime::IsLegalTransition;
using engine::runtime::OutputChannel;
using engine::runtime::OutputItem;
using engine::runtime::Request;
using engine::runtime::SeqState;
using engine::runtime::Sequence;
using engine::sampling::SamplingParams;
using engine::tensor::DataType;

constexpr int kBs = 8;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

// Dereference an optional the analyzer can prove engaged (the explicit-guard
// idiom bugprone-unchecked-optional-access requires).
template <typename T>
[[nodiscard]] T Require(std::optional<T> opt) {
  if (!opt.has_value()) {
    ADD_FAILURE() << "expected an engaged optional";
    return T{};
  }
  return std::move(opt).value();
}

[[nodiscard]] CacheGeometry TinyGeom() {
  return CacheGeometry{.num_layers = 2,
                       .num_kv_heads = 2,
                       .head_dim = 16,
                       .dtype = DataType::kFloat32};
}

[[nodiscard]] BlockPool MakePool(std::int64_t num_blocks = 8) {
  return Unwrap(BlockPool::Create(TinyGeom(), kBs, num_blocks, nullptr));
}

[[nodiscard]] Request GreedyRequest(engine::runtime::RequestId id = 1) {
  Request req;
  req.id = id;
  req.prompt_ids = {1, 2, 3};
  req.params = SamplingParams::Greedy(/*max_tokens=*/8);
  req.eos_ids = {0};
  req.arrival_index = id;
  return req;
}

// --- construction --------------------------------------------------------

TEST(SequenceTest, CreateStartsWaiting) {
  BlockPool pool = MakePool();
  const Request req = GreedyRequest();
  Sequence seq = Unwrap(Sequence::Create(req, &pool));

  EXPECT_EQ(seq.state(), SeqState::kWaiting);
  EXPECT_FALSE(seq.is_terminal());
  EXPECT_EQ(seq.id(), 1U);
  EXPECT_EQ(seq.num_generated(), 0);
  EXPECT_EQ(seq.num_computed_tokens(), 0);
  ASSERT_NE(seq.cache(), nullptr);
  EXPECT_EQ(seq.cache()->length(), 0);
  ASSERT_NE(seq.channel(), nullptr);
  EXPECT_FALSE(seq.channel()->closed());
  // Empty cache holds no blocks yet.
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(SequenceTest, CreateResolvesSeed) {
  BlockPool pool = MakePool();
  Request req = GreedyRequest();
  req.params.temperature = 0.7F;
  req.params.seed = 12345U;
  Sequence seq = Unwrap(Sequence::Create(req, &pool));
  EXPECT_EQ(seq.sampler().seed(), 12345U);
}

TEST(SequenceTest, CreateRejectsInvalidParams) {
  BlockPool pool = MakePool();
  Request req = GreedyRequest();
  req.params.temperature = -1.0F;  // fails ValidateSamplingParams.
  auto seq = Sequence::Create(req, &pool);
  EXPECT_TRUE(IsInvalidArgument(seq.status()));
}

TEST(SequenceTest, CreateRejectsStopStringsWithoutTokenizer) {
  BlockPool pool = MakePool();
  Request req = GreedyRequest();
  req.params.stop_strings = {"STOP"};
  req.tokenizer = nullptr;  // stop strings need a tokenizer (StopChecker).
  auto seq = Sequence::Create(req, &pool);
  EXPECT_TRUE(IsInvalidArgument(seq.status()));
}

// --- state machine -------------------------------------------------------

// The full 6x6 transition matrix matches §3.2 exactly.
TEST(SequenceTest, TransitionMatrixMatchesTable) {
  using S = SeqState;
  constexpr S kStates[] = {S::kWaiting,  S::kRunning,   S::kPreempted,
                           S::kFinished, S::kCancelled, S::kFailed};
  auto legal = [](S from, S to) {
    // The nine legal edges of §3.2.
    if (from == S::kWaiting) {
      return to == S::kRunning || to == S::kCancelled;
    }
    if (from == S::kRunning) {
      return to == S::kRunning || to == S::kFinished || to == S::kPreempted ||
             to == S::kCancelled || to == S::kFailed;
    }
    if (from == S::kPreempted) {
      return to == S::kRunning || to == S::kCancelled;
    }
    return false;  // terminals have no outgoing edges.
  };
  for (const S from : kStates) {
    for (const S to : kStates) {
      EXPECT_EQ(IsLegalTransition(from, to), legal(from, to))
          << static_cast<int>(from) << " -> " << static_cast<int>(to);
    }
  }
}

TEST(SequenceTest, LegalTransitionsSucceed) {
  BlockPool pool = MakePool();
  const Request req = GreedyRequest();

  // waiting -> running -> running -> finished
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kRunning);
    EXPECT_EQ(seq.state(), SeqState::kRunning);
    seq.Transition(SeqState::kRunning);  // decode self-loop.
    seq.Transition(SeqState::kFinished);
    EXPECT_TRUE(seq.is_terminal());
  }
  // waiting -> cancelled
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kCancelled);
    EXPECT_EQ(seq.state(), SeqState::kCancelled);
  }
  // running -> preempted -> running
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kRunning);
    seq.Transition(SeqState::kPreempted);
    seq.Transition(SeqState::kRunning);
    EXPECT_EQ(seq.state(), SeqState::kRunning);
  }
  // running -> preempted -> cancelled
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kRunning);
    seq.Transition(SeqState::kPreempted);
    seq.Transition(SeqState::kCancelled);
    EXPECT_EQ(seq.state(), SeqState::kCancelled);
  }
  // running -> failed
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kRunning);
    seq.Transition(SeqState::kFailed);
    EXPECT_EQ(seq.state(), SeqState::kFailed);
  }
}

TEST(SequenceDeathTest, IllegalTransitionsCheck) {
  BlockPool pool = MakePool();
  const Request req = GreedyRequest();

  // waiting -> finished (must admit first).
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    EXPECT_DEATH(seq.Transition(SeqState::kFinished), "illegal transition");
  }
  // waiting -> preempted.
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    EXPECT_DEATH(seq.Transition(SeqState::kPreempted), "illegal transition");
  }
  // waiting -> waiting (self-loop only legal for running).
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    EXPECT_DEATH(seq.Transition(SeqState::kWaiting), "illegal transition");
  }
  // out of a terminal state (finished -> running).
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kRunning);
    seq.Transition(SeqState::kFinished);
    EXPECT_DEATH(seq.Transition(SeqState::kRunning), "illegal transition");
  }
  // preempted -> finished.
  {
    Sequence seq = Unwrap(Sequence::Create(req, &pool));
    seq.Transition(SeqState::kRunning);
    seq.Transition(SeqState::kPreempted);
    EXPECT_DEATH(seq.Transition(SeqState::kFinished), "illegal transition");
  }
}

// --- progress & finish info ----------------------------------------------

TEST(SequenceTest, AppendGeneratedAdvancesProgress) {
  BlockPool pool = MakePool();
  Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));
  seq.set_num_computed_tokens(3);  // prompt prefilled.
  seq.AppendGenerated(11);
  seq.AppendGenerated(12);
  EXPECT_EQ(seq.num_generated(), 2);
  EXPECT_EQ(seq.num_computed_tokens(), 5);
  ASSERT_EQ(seq.generated_ids().size(), 2U);
  EXPECT_EQ(seq.generated_ids()[0], 11);
  EXPECT_EQ(seq.generated_ids()[1], 12);
}

TEST(SequenceTest, MakeFinishInfoFinished) {
  BlockPool pool = MakePool();
  Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));
  seq.Transition(SeqState::kRunning);
  seq.set_finish(engine::engine::FinishReason::kStop,
                 engine::engine::StopTrigger::kEosId, "");
  seq.Transition(SeqState::kFinished);

  const FinishInfo info = seq.MakeFinishInfo();
  EXPECT_EQ(info.terminal, SeqState::kFinished);
  EXPECT_EQ(info.finish_reason, engine::engine::FinishReason::kStop);
  EXPECT_EQ(info.stop_trigger, engine::engine::StopTrigger::kEosId);
  EXPECT_TRUE(info.error.ok());
}

TEST(SequenceTest, MakeFinishInfoFailedCarriesError) {
  BlockPool pool = MakePool();
  Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));
  seq.Transition(SeqState::kRunning);
  seq.set_error(engine::core::InternalError("boom"));
  seq.Transition(SeqState::kFailed);

  const FinishInfo info = seq.MakeFinishInfo();
  EXPECT_EQ(info.terminal, SeqState::kFailed);
  EXPECT_FALSE(info.error.ok());
}

TEST(SequenceDeathTest, MakeFinishInfoNonTerminalChecks) {
  BlockPool pool = MakePool();
  const Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));
  EXPECT_DEATH((void)seq.MakeFinishInfo(), "non-terminal");
}

// --- cache lifecycle (RAII block reclamation) ----------------------------

TEST(SequenceTest, ReleaseCacheReturnsBlocksAndResets) {
  namespace ops = engine::tensor::ops;
  using engine::tensor::Shape;
  using engine::tensor::Tensor;

  BlockPool pool = MakePool();
  Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));

  // Commit a real prefill through the cache so physical blocks are taken (the
  // append contract itself is covered by paged_cache_test; here we assert
  // ReleaseCache returns whatever was taken — the "no block leaks" basis).
  const CacheGeometry geom = TinyGeom();
  constexpr std::int64_t kT = 4;
  auto kv = [&] {
    return Unwrap(ops::zeros(Shape{kT, geom.num_kv_heads, geom.head_dim},
                             DataType::kFloat32));
  };
  for (int layer = 0; layer < geom.num_layers; ++layer) {
    ASSERT_TRUE(seq.cache()->append(layer, kv(), kv()).ok());
  }
  seq.set_num_computed_tokens(kT);
  EXPECT_EQ(seq.cache()->length(), kT);
  EXPECT_GT(pool.stats().used, 0);

  seq.ReleaseCache();
  EXPECT_EQ(seq.cache(), nullptr);
  EXPECT_EQ(seq.num_computed_tokens(), 0);
  EXPECT_EQ(pool.stats().used, 0);  // every block returned.

  seq.EnsureCache(&pool);
  ASSERT_NE(seq.cache(), nullptr);
  EXPECT_EQ(seq.cache()->length(), 0);
}

TEST(SequenceTest, DestructorReturnsBlocks) {
  namespace ops = engine::tensor::ops;
  using engine::tensor::Shape;
  BlockPool pool = MakePool();
  {
    Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));
    const CacheGeometry geom = TinyGeom();
    auto kv = Unwrap(ops::zeros(Shape{4, geom.num_kv_heads, geom.head_dim},
                                DataType::kFloat32));
    for (int layer = 0; layer < geom.num_layers; ++layer) {
      ASSERT_TRUE(seq.cache()->append(layer, kv, kv).ok());
    }
    EXPECT_GT(pool.stats().used, 0);
  }  // sequence (and its cache) destroyed.
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(SequenceTest, ChannelOutlivesSequence) {
  BlockPool pool = MakePool();
  std::shared_ptr<OutputChannel> channel;
  {
    Sequence seq = Unwrap(Sequence::Create(GreedyRequest(), &pool));
    channel = seq.channel();  // shared handle.
    channel->Push({.token_id = 5, .text_delta = "", .logprobs = {}});
    seq.Transition(SeqState::kRunning);
    seq.Transition(SeqState::kFinished);
    channel->Close(seq.MakeFinishInfo());
  }  // sequence destroyed; channel still alive via the shared_ptr.

  const OutputItem item = Require(channel->Next());
  EXPECT_EQ(item.token_id, 5);
  EXPECT_FALSE(channel->Next().has_value());
  const FinishInfo fin = Require(channel->finish());
  EXPECT_EQ(fin.terminal, SeqState::kFinished);
}

}  // namespace
