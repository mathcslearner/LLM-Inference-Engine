#pragma once

#include "core/status.h"
#include "tensor/tensor.h"

#include <cstdint>

// Workspace (M6-T07; design: docs/design/optimized-cpu-execution.md §6). The
// reused-across-layers scratch the optimized forward pass runs out of: the
// residual-stream tensors, the QKV/context projections, and the MLP gate/up
// buffers, all fp32 and sized for one layer's activations (layer i+1 overwrites
// layer i's — §6.1). `OptimizedModel` owns exactly one and threads its slots
// through every layer, so steady-state decode is allocation-free once the first
// call of a given T has sized it (§6.3, the M12-T05 target reachable now).
//
// Storage policy (§6.3): monotone grow-on-demand with a high-water mark.
// `EnsureCapacity(T)` (re)allocates any slot whose row capacity is below T and
// never shrinks; a growth/OOM failure surfaces as a recoverable Status
// *before* any kernel runs (front-loaded, ADR-003). Logits are NOT held here —
// they are a freshly allocated caller-owned tensor (§6.3), so V does not enter
// the workspace size, and neither does the cache length L (the K/V gather is
// the cache's allocation, §8).
//
// As-built note (design §6.1): the attention kernels keep no per-worker scratch
// (the online-softmax accumulator is `out` itself), so `W_worker = 0` and this
// holds model-level buffers only.

namespace engine::model {

// The per-layer activation slots, all contiguous row-major fp32. The dims
// (E, H, Hkv, d, I) are fixed at construction from the model config; each
// forward calls `EnsureCapacity(T)` and then reads the slots as [T, width]
// prefix views. The names mirror the reference decoder-block orchestration
// (model-execution.md §4.2): `h`/`tmp`/`r` are the residual-stream
// intermediates, `q`/`k`/`v`/`ctx` the attention projections, `gate`/`up` the
// MLP projections. Aliasing-friendly kernels (RmsNormF32, SiluMulF32, AddF32
// all permit y to alias an input) let the graph reuse these tightly.
class Workspace {
 public:
  // Builds the workspace shell for a model with hidden size `e`, `num_heads`
  // query heads, `num_kv_heads` KV heads, head dim `d`, and intermediate size
  // `i`. No storage is allocated until the first `EnsureCapacity` (a T=0
  // workspace holds nothing). All dims must be positive; a non-positive dim is
  // a programmer error (CHECK — the builder validates the config first).
  [[nodiscard]] static Workspace Create(std::int64_t e, int num_heads,
                                        int num_kv_heads, int d,
                                        std::int64_t i);

  // Grows every slot to hold at least `t` rows (monotone; never shrinks). A
  // no-op when `t <= capacity_tokens()`. An allocation failure → the tensor
  // factory's Status (ResourceExhausted/OutOfMemory), leaving the prior
  // (smaller but valid) buffers intact — the caller aborts the forward before
  // touching the cache. `t` must be >= 1.
  [[nodiscard]] core::Status EnsureCapacity(std::int64_t t);

  // The [t, width] contiguous prefix views the forward pass writes into. Each
  // requires `t <= capacity_tokens()` (CHECK — the caller EnsureCapacity's
  // first). Returned by value (cheap Tensor views, shared storage). The four
  // E-width buffers (`x`/`h`/`tmp`/`r`, the `c_stream = 4` of §6.2) hold the
  // residual stream and its per-block intermediates: `x` is the residual stream
  // (embeddings in, layer output back), `h` a norm output, `tmp` a projection
  // output (o_proj / down_proj), `r` the post-attention residual.
  [[nodiscard]] tensor::Tensor x(std::int64_t t) const;     // [T, E]
  [[nodiscard]] tensor::Tensor h(std::int64_t t) const;     // [T, E]
  [[nodiscard]] tensor::Tensor tmp(std::int64_t t) const;   // [T, E]
  [[nodiscard]] tensor::Tensor r(std::int64_t t) const;     // [T, E]
  [[nodiscard]] tensor::Tensor q(std::int64_t t) const;     // [T, H*d]
  [[nodiscard]] tensor::Tensor k(std::int64_t t) const;     // [T, Hkv*d]
  [[nodiscard]] tensor::Tensor v(std::int64_t t) const;     // [T, Hkv*d]
  [[nodiscard]] tensor::Tensor ctx(std::int64_t t) const;   // [T, H*d]
  [[nodiscard]] tensor::Tensor gate(std::int64_t t) const;  // [T, I]
  [[nodiscard]] tensor::Tensor up(std::int64_t t) const;    // [T, I]

  // The high-water mark: the largest T any slot is currently sized for (0
  // before the first EnsureCapacity).
  [[nodiscard]] std::int64_t capacity_tokens() const { return capacity_; }

  // Total bytes currently backing the slots — the §6.2 formula
  // `W_model = 4·[3·T·E + (H + 2·Hkv)·T·d + T·H·d + 2·T·I]` at the current
  // capacity (0 before the first EnsureCapacity). A test asserts this matches
  // the instantiated tiny-llama figure and grows monotonically.
  [[nodiscard]] std::int64_t bytes() const;

 private:
  Workspace(std::int64_t e, std::int64_t q_rows, std::int64_t kv_rows,
            std::int64_t i)
      : e_(e), q_rows_(q_rows), kv_rows_(kv_rows), i_(i) {}

  // Returns a [t, width] fp32 prefix view of `slot` (CHECK t <= capacity_).
  [[nodiscard]] tensor::Tensor Prefix(const tensor::Tensor& slot,
                                      std::int64_t t) const;

  std::int64_t e_ = 0;        // hidden size E
  std::int64_t q_rows_ = 0;   // H*d
  std::int64_t kv_rows_ = 0;  // Hkv*d
  std::int64_t i_ = 0;        // intermediate size I
  std::int64_t capacity_ = 0;

  // Slots, each [capacity_, width]; undefined until the first EnsureCapacity.
  tensor::Tensor x_;
  tensor::Tensor h_;
  tensor::Tensor tmp_;
  tensor::Tensor r_;
  tensor::Tensor q_;
  tensor::Tensor k_;
  tensor::Tensor v_;
  tensor::Tensor ctx_;
  tensor::Tensor gate_;
  tensor::Tensor up_;
};

}  // namespace engine::model
