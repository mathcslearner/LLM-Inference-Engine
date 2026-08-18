# Model execution: modules, forward contract & KV cache

**Milestone:** M5 (design doc: M5-T01; implementation: M5-T02 … M5-T10)
**Governs:** `src/cpu/` (the reference forward pass), `src/model/`'s execution
surface (the model graph — modules, `Model`, architecture registry), the
`kvcache` interface, and `src/engine/`'s generation loop — plus the
forward-pass contract every later execution milestone (M6 optimized backend,
M8 paged cache, M9 batching, M11 prefix caching, M12 fusions, M13 quantized
layers, M14 activation capture, M15 speculative decoding) builds on.
**Cites:** ADR-002 (module boundaries, through **Amendment 5** — this doc's
`model → kvcache` edge), ADR-003 (error handling), ADR-004 (CPU-first pivot),
`docs/design/cpu-backend.md` (threading, dtype/accumulation policy, oracle
chain, numerics classes §6.3), `docs/design/model-loading.md` (`ModelConfig`,
canonical weight names §4, dtype-at-load policy §5, fixtures §7),
`docs/design/tensor.md` (Tensor/Buffer, dtypes, half conversions).

This is the working contract for model execution. Implementation tickets must
conform to it; if implementation reveals a design flaw, this doc is updated in
the same change with a note on what changed and why
(`docs/design/README.md`). M7-T01 will append a sampling-pipeline section here
(§15 placeholder); other milestones amend in place.

---

## 1. Scope & non-goals

M5 turns "weights are tensors in a registry" (where M4 ended) into "the model
computes logits and generates text." It is the **correctness oracle** for
every later execution optimization: a complete, unoptimized, clarity-first
scalar forward pass and greedy generation loop for Llama/Qwen-family
decoder-only transformers, validated against the HuggingFace activation
fixtures (cpu-backend.md §6.1). Speed is an explicit non-goal; obviousness is
the goal. This milestone also **fixes the interfaces** — `Model::forward`, the
module decomposition, the KV-cache API, the backend seam — that M6's optimized
backend and everything after it implement without rewriting.

**In scope (this doc):** the module decomposition (`Linear`, `RmsNorm`,
`Rope`, `Attention`, `Mlp`, `DecoderLayer`, `Model`); how canonical weights
(model-loading.md §4) bind to modules; the exact `Model::forward` contract
(inputs, outputs, preconditions, error posture); batch/sequence/head
dimension conventions and the GQA layout; the numerics/dtype policy the
reference obeys; the RoPE specification; the KV-cache v0 interface and an
explicit account of what M8 (paged) changes; the reference-vs-optimized
backend seam; the architecture registry and builder; the greedy generation
loop; debug/activation hooks; and the golden fixtures M5 adds.

**Non-goals (of this doc, not the project):**

- **No optimized kernels.** Packed GEMM, blocked/flash attention, vectorized
  norms, weight repacking, and workspace strategy are M6-T01's doc
  (`docs/design/optimized-cpu-execution.md`). This doc gives them the `Model`
  and module contracts to implement behind.
- **No sampling beyond greedy argmax.** Temperature, top-k/p, penalties,
  logprobs, and seeded RNG are M7; M7-T01 appends its pipeline to this doc
  (§15). M5's generation loop hard-codes greedy — the smallest thing that
  makes end-to-end generation testable against HF `generate(do_sample=False)`.
- **No paged cache, no batching, no continuous batching.** v0 is a single
  sequence, one contiguous append-only cache. M8 (paged), M9 (ragged batches +
  cu_seqlens) enter behind the interfaces this doc fixes; §6.4 and §5.4 record
  what must not be foreclosed.
- **No quantized weights.** `Linear` is specified as an *interface* precisely
  so M13's `QuantizedLinear` slots in (§4.1); the quant numerics are M13's doc.
- **No chat templates, no HTTP, no scheduler.** The consumer of M5 is a
  test-driven greedy loop, not a server.

---

## 2. Module layout & layering

M5 populates four module directories. Three are new execution surfaces; one
(`model`) already exists from M4 and gains its execution graph.

| Module | Path | Role in M5 | Ticket |
|---|---|---|---|
| `cpu` | `src/cpu/` | **The reference backend**: stateless, clarity-first fp32 ops — GEMM, RMSNorm, SiLU-and-mul, residual add, softmax, embedding lookup, RoPE apply, causal attention. The oracle (cpu-backend.md §2.1). | T02–T05 |
| `model` | `src/model/` | The **model graph**: module abstractions (`Linear`, `RmsNorm`, `Rope`, `Attention`, `Mlp`, `DecoderLayer`), the abstract `Model` and its reference implementation, the architecture registry/builder. | T02–T08, T10 |
| `kvcache` | `src/kvcache/` | The **KV-cache interface** (`KvCache`, §6) and its v0 contiguous implementation (`SimpleKvCache`). | T06 |
| `engine` | `src/engine/` | The **generation loop** (greedy prefill + decode) and the backend-selection seam. | T09 |

Files this milestone creates:

| File | Module | Contents | Ticket |
|---|---|---|---|
| `src/cpu/ops.h` / per-op `.cpp` | cpu | reference op entries (`gemm`, `rmsnorm`, `silu_mul`, `add`, `softmax`, `embedding_lookup`, `rope_apply`, `attention`) | T02–T05 |
| `src/model/modules.h` / `.cpp` | model | `Linear` (interface) + `ReferenceLinear`, `RmsNorm`, `Embedding`, `Rope`, `Attention`, `Mlp`, `DecoderLayer` | T02–T07 |
| `src/model/model.h` | model | abstract `Model`, `ForwardRequest`, `LogitsMode`, `ActivationHook` | T07 |
| `src/model/reference_model.h` / `.cpp` | model | `ReferenceModel` (embedding → N `DecoderLayer` → final norm → lm_head) | T07 |
| `src/model/registry.h` / `.cpp` | model | HF-architecture-string → builder map; `BuildModel` | T08 |
| `src/kvcache/kv_cache.h` | kvcache | abstract `KvCache` interface (§6) + `CacheGeometry`, `KvView` | T05 |
| `src/kvcache/simple_cache.h` / `.cpp` | kvcache | `SimpleKvCache` — per-sequence contiguous append-only storage | T06 |
| `src/engine/generator.h` / `.cpp` | engine | `Generate(...)` greedy loop; per-token callback | T09 |
| `src/engine/backend.h` | engine | `enum class Backend { kReference, kOptimized }`, selection helper (M6 fills the optimized arm) | T09 |

### 2.1 Layering: the edges M5 uses, and the one it adds

Positions in the ADR-002 layer diagram (Amendment 5 is the authority; restated
here per this doc's role):

- `cpu` — **layer 1**, `cpu → tensor, parallel`. Both edges are *already
  listed* (Amendment 4 added `parallel` beside `memory` and named `cpu` among
  its consumers). The M5-T02 threaded reference GEMM links `parallel`
  directly; **no ADR amendment is needed** for it. *(This retires the
  provisional flag in cpu-backend.md §2's third bullet, which anticipated a
  possible amendment before Amendment 4 was finalized. `cpu` links
  `parallel`; the edge was already allowed.)* `cpu` does **not** link
  `kernels` — the reference must not depend on dispatched kernels it exists to
  validate — so on-the-fly fp16/bf16→fp32 conversion uses `tensor/half.h`'s
  value-type conversions (`operator float()` widening), not
  `kernels/convert`.
- `model` — **layer 2**, `model → tensor, memory, core` (M4) **plus a new
  `model → kvcache` edge** (`Model::forward` takes a `KvCache*`; the
  `Attention` module reads and appends through it). This edge does not exist
  in ADR-002 through Amendment 4 and is added by **Amendment 5** (§2.2). No
  cycle: `kvcache` is a sibling in layer 2 and never links `model`.
- `kvcache` — **layer 2**, `kvcache → tensor, memory, core`. v0 stores K/V as
  ordinary CPU `Tensor`s; M8's block pool will reuse M2's `CachingAllocator`
  (`kvcache → memory` covers it).
- `engine` — **layer 3**, `engine → model, kvcache, cpu, tensor, ...` (all
  lower layers). The generation loop depends only on the abstract `Model` and
  `KvCache` interfaces, so it is backend-agnostic by construction (§8, §10).

### 2.2 ADR-002 Amendment 5: `model → kvcache`

**Why the model graph, not the engine, owns the forward pass.** The roadmap
fixes `src/model/registry.h` as the thing that "wires `ModelConfig` + weights
→ `Model`" (M5-T08), and a `Model` that produces logits must consult the KV
cache inside its attention layers. Two placements were considered:

1. **`Model` in `model`, cache passed in (chosen).** `Model::forward` takes a
   `KvCache*`; `Attention` appends new K/V and reads the accumulated K/V
   through the abstract interface. Requires the new `model → kvcache` edge.
2. **`Model` in `engine`, `model` stays pure loading.** Keeps `model` free of
   the cache, but contradicts the roadmap's `src/model/registry.h` placement,
   and splits the model graph across two modules (modules in `model`, the
   `Model` that composes them in `engine`) for no gain.

Option 1 is chosen; it is recorded as **ADR-002 Amendment 5** (dated with this
ticket). The alternative of threading raw K/V *views* through `forward`
(keeping `model` cache-free) was also rejected: M8's paged attention must read
the cache through a block-table abstraction that lives *inside* the attention
kernel, so the cache abstraction has to be reachable from the model graph
regardless — passing it explicitly now is what makes M8-T07's "swap the paged
cache in behind the M5 interface" a drop-in rather than a re-plumbing.

Everything below assumes little-endian and the fp32-accumulation policy
(cpu-backend.md §5); no other ADR edge changes.

---

## 3. Execution model & tensor conventions

### 3.1 The shapes, fixed once

M5 processes **one sequence per `forward` call**, with no explicit batch
dimension. Activations flow as **token-major 2-D tensors** — this is the
convention M9 extends to ragged batches by flattening `[Σ tokens, hidden]`
with `cu_seqlens`, so choosing it now (rather than a rectangular `[B, T, H]`)
means M9 adds a batch concept without reshaping M5's math.

Let `T` = number of tokens in this call, `P` = cache length before the call,
`H` = `num_heads`, `Hkv` = `num_kv_heads`, `d` = `head_dim`, `E` =
`hidden_size`, `I` = `intermediate_size`, `V` = `vocab_size`.

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| hidden states `x` | `[T, E]` | f32 | the residual stream; no batch dim in v0 |
| Q | `[T, H, d]` | f32 | post-projection, pre-RoPE and post-RoPE |
| K, V (new this call) | `[T, Hkv, d]` | f32 | appended to the cache |
| K, V (from cache) | `[Hkv, P+T, d]` | f32 | head-major; see §6.2 |
| attention scores | `[H, T, P+T]` | f32 | per query block, causal-masked |
| logits | `[T or 1, V]` | f32 | §5.2 `LogitsMode` |

`E == H · d` is the common case but **not assumed**: `head_dim` may be
explicit and decoupled from `hidden_size` (model-loading.md §3.2), so `Q`'s
`[T, H, d]` uses `config.head_dim`, and the attention output projection maps
`H · d → E` — the checkpoint's `o_proj` shape `[E, H·d]` is authoritative.

### 3.2 GQA layout & the kv-repeat convention

Grouped-query attention: `H` query heads share `Hkv` key/value heads, with
`num_heads % num_kv_heads == 0` (validated at load, model-loading.md §3.2). The
group size is `g = H / Hkv`. Query head `h` (0-indexed) attends to **kv head
`h / g`** — i.e. `repeat_interleave`, matching HF's `repeat_kv`
(`hidden_states[:, :, None, :]` expand), **not** tiling (`h % Hkv`). This is
stated as the convention because it is invisible in single-KV-head tests and
load-bearing for correctness:

```
query heads:   0    1    2    3    4    5    6    7     (H = 8)
kv head:       0    0    1    1    2    2    3    3     (Hkv = 4, g = 2)  ← h / g
```

The reference **materializes no repeat**: `cpu::attention` takes the `[Hkv,
P+T, d]` K/V and, for query head `h`, indexes kv head `h / g` directly. M6-T04
requires exactly this ("GQA via KV head indexing, no materialized repeat"), so
the convention is set here in a form both backends honor.

### 3.3 Dtype & numerics policy

Per cpu-backend.md §5, restated for execution ops:

- **fp32 accumulation everywhere.** GEMM accumulators, RMSNorm sums, softmax,
  attention score/context sums — all fp32, regardless of weight storage dtype.
- **Weights are storage; convert at the kernel boundary.** A bf16/f16
  checkpoint weight (model-loading.md §5 preserves it) is widened to fp32 as it
  is read, via `tensor/half.h` (`operator float()` — bit-exact, constexpr).
  The reference never up-converts weights at load (that would defeat M4's
  zero-copy property); it converts per element as it multiplies. Activations
  are fp32 throughout the M5 forward pass.
- **Numerics class T** (cpu-backend.md §6.3): every M5 op is
  reference-vs-fixture within a **stated per-test tolerance** (absolute +
  relative), because HF computes in its own op order (FMA contraction,
  different reduction trees). The activation goldens are the *same computation*
  the reference performs (fp32 forward of the bf16 checkpoint, fixtures
  README), so agreement is expected to be far tighter than the tolerance —
  tests record the observed max-abs-diff and set the threshold above it with
  margin, never the reverse.

---

## 4. Modules & weight binding

The forward pass is assembled from small modules, each owning its weights and
exposing a single `forward`-style entry. Modules are **plain classes in
`model`**, holding backend-owned weight handles; the reference implementations
call `cpu::` ops.

### 4.1 `Linear` — an interface, not a struct

`Linear` is the one module that is deliberately an **abstraction**, because
M13 slots `QuantizedLinear` in behind it without touching any layer code
("quantized layers slot in behind the `Linear` interface", M13 overview):

```cpp
// src/model/modules.h
class Linear {
 public:
  virtual ~Linear() = default;
  // y[T, out] = x[T, in] · Wᵀ  (+ bias[out] if present).  Row-major, fp32
  // activations in and out; the weight's storage dtype and layout are the
  // implementation's private business.
  [[nodiscard]] virtual core::Status forward(
      const tensor::Tensor& x, tensor::Tensor& y) const = 0;
  [[nodiscard]] virtual std::int64_t in_features() const = 0;
  [[nodiscard]] virtual std::int64_t out_features() const = 0;
  [[nodiscard]] virtual bool has_bias() const = 0;
};
```

`ReferenceLinear` (M5-T02) holds the canonical weight `Tensor`
(checkpoint-order `[out, in]`, model-loading.md §4) and optional bias as
**zero-copy handles into the mapped checkpoint** — no repacking, no copy. Its
`forward` is `cpu::gemm` with on-the-fly weight conversion. Key properties the
interface fixes for later milestones:

- **Weight representation is opaque and backend-owned.** The reference keeps
  the raw checkpoint tensor; M6's `PackedLinear` repacks into cache-blocked
  tiles at load (M6-T02); M13's `QuantizedLinear` holds qweight/scales/zeros.
  Nothing outside a `Linear` implementation may read `.weight` as an fp32
  matrix — so no call site assumes a layout.
- **One call site serves prefill (GEMM shape, `T` large) and decode (GEMV
  shape, `T == 1`).** The reference does not distinguish them; M13's kernel
  plan (fused dequant-GEMV for decode, dequant-to-tiles for prefill) dispatches
  inside `forward` on `T`.
- **Bias is per-projection and per-config.** Llama default: no bias. Qwen2:
  bias on q/k/v (not o). The builder (§9) constructs `ReferenceLinear` with or
  without bias from `config.attention_bias` — **M5-T10's Qwen2 support is
  wiring, not new layer code**, exactly as the roadmap requires.

### 4.2 The other modules

Simple concrete classes (no interface needed in M5; M6 reuses them or provides
parallel implementations behind `Model`, §8):

- **`RmsNorm`** (M5-T03): holds the `[E]` weight; `forward(x) → y` computes
  `y = x / sqrt(mean(x²) + eps) · weight` in fp32, per HF Llama
  (`x * rsqrt(mean(x*x, -1) + eps)` then scale), `eps = config.rms_norm_eps`.
  Mean is over the hidden dimension per token. Reference calls `cpu::rmsnorm`.
- **`Embedding`** (M5-T04): holds the `[V, E]` table (zero-copy checkpoint
  handle, bf16/f16 storage preserved); `forward(ids) → y` gathers rows widened
  to fp32 via `cpu::embedding_lookup`. Added as a concrete module (the design
  originally treated the lookup as a bare op in `ReferenceModel`) so it is the
  seam for M6-T06's tied-weight sharing — lookup and lm_head projection can hold
  *different physical layouts* of the same logical weight behind separate
  modules. `ids` is the `ForwardRequest.token_ids` span, passed straight
  through.
- **`Rope`** (M5-T04): precomputes cos/sin tables `[num_positions, d/2]` at
  construction from `config.rope_theta` and any `rope_scaling` (§7), exposing
  `inv_freq` for the scaling goldens; `apply(Q, K, positions)` rotates in place
  via `cpu::rope_apply` (once per tensor). Positions are a **per-token vector**,
  not a scalar start (M6-T03/M11 need arbitrary positions). Table construction
  (config interpretation) lives in the module; the stateless `cpu::rope_apply`
  takes the tables as inputs.
- **`Attention`** (M5-T05): owns q/k/v/o `Linear`s (as owning
  `std::unique_ptr<Linear>` handles, so M6/M13 slot `PackedLinear`/
  `QuantizedLinear` in — the class is move-only) and a `Rope`.
  `Create(q,k,v,o, rope, H, Hkv, d)` validates the projection shapes against the
  head counts (`q_proj out == H·d`, `k/v_proj out == Hkv·d`, `o_proj in == H·d`,
  q/k/v share `in == E`, `o_proj out == E`, `rope.head_dim() == d`, `H % Hkv ==
  0`) — `E == H·d` is *not* assumed, so a decoupled head_dim is honored.
  `forward(x, positions, layer, cache, y)` projects QKV, applies RoPE to Q and K,
  **appends new K/V to the cache** (§6), reads the accumulated `[Hkv, P+T, d]`
  K/V back, computes causal-masked GQA attention via `cpu::attention` (scale =
  `1/sqrt(d)` formed in double, cast to fp32 — HF's `head_dim**-0.5`), and
  projects the context through `o_proj`. Input validation is front-loaded so a
  failure never leaves a half-appended layer; cache errors (geometry/capacity)
  propagate as their own `Status`. This is the one module that touches
  `KvCache`. The `cpu::attention` op materializes the scores `[H·T, L]`,
  causal-masks with `-inf`, softmaxes via `cpu::softmax` (reusing its `-inf → 0`
  mask contract), and contracts against V; GQA is by KV-head indexing (query
  head `h` → kv head `h / g`), no materialized repeat.
- **`Mlp`** (M5-T03/T07): owns gate/up/down `Linear`s; SwiGLU —
  `down(silu(gate(x)) ⊙ up(x))`, fp32, via `cpu::silu_mul` + `cpu::gemm`.
- **`DecoderLayer`** (M5-T07): `attn_norm → Attention → residual add →
  mlp_norm → Mlp → residual add`, the pre-norm arrangement (norm names are
  *positions*, model-loading.md §4). Holds two `RmsNorm`s, one `Attention`, one
  `Mlp`.

### 4.3 Weight binding

The builder (§9) resolves every module's weights from `LoadedModel.weights`
(canonical names, shape-validated in M4) by name:

- `embed_tokens.weight [V, E]`, `final_norm.weight [E]`, `lm_head.weight
  [V, E]`.
- Per layer `i`: `layers.{i}.attn_norm.weight`, `layers.{i}.attn.{q,k,v,o}_proj
  .weight` (+ `.bias` iff `attention_bias`), `layers.{i}.mlp_norm.weight`,
  `layers.{i}.mlp.{gate,up,down}_proj.weight`.
- **Tied embeddings** (`config.tie_word_embeddings`): M4 already stores the
  *same* `Tensor` handle under `lm_head.weight` and `embed_tokens.weight`. The
  builder honors this as an **explicit, overridable binding** — the reference
  binds `lm_head`'s `Linear` to the shared handle; M6-T06 overrides the binding
  so lookup and projection can use *different physical layouts* of the same
  logical weight ("share storage across different layouts — resolved in the
  design doc"). So tying is a binding decision the builder makes, never a
  pointer-equality assumption baked into the model.

Modules borrow weight handles (shared `Tensor`, i.e. shared `Buffer`); the
mapped checkpoint stays alive as long as any module holds a handle (M4's
lifetime contract). The `LoadedModel`'s `weights` map is moved into the model
so the model owns the handles for its lifetime.

---

## 5. The `Model::forward` contract

### 5.1 The interface

```cpp
// src/model/model.h
enum class LogitsMode {
  kLast,  // logits for the final position only → [1, V]   (steady-state decode)
  kAll,   // logits for every position          → [T, V]   (perplexity, spec-verify)
};

struct ForwardRequest {
  std::span<const std::int32_t> token_ids;  // [T], each in [0, V)
  std::span<const std::int32_t> positions;  // [T], absolute positions; size == T
  kvcache::KvCache* cache = nullptr;        // non-null; length P on entry, P+T on return
  LogitsMode logits_mode = LogitsMode::kLast;
  ActivationHook* hook = nullptr;           // §11; nullptr = no capture, zero cost
};

class Model {
 public:
  virtual ~Model() = default;
  // Runs the forward pass for one sequence, appending this call's K/V to
  // `cache`. Returns fp32 logits: [1, V] for kLast, [T, V] for kAll.
  [[nodiscard]] virtual core::StatusOr<tensor::Tensor> forward(
      const ForwardRequest& request) = 0;

  [[nodiscard]] virtual const ModelConfig& config() const = 0;
  // The cache geometry this model requires (layers, Hkv, d, dtype). The
  // caller constructs a matching KvCache (§6.1).
  [[nodiscard]] virtual kvcache::CacheGeometry cache_geometry() const = 0;
};
```

`ForwardRequest` is a **struct, not loose parameters**, because M9-T05 grows it
into the batch-assembly bundle (cu_seqlens, slot mappings, block-table tensor,
per-request sampling metadata) "assembled in one pass into preallocated staging
buffers." Choosing a struct now means those become new fields, not a new
signature. v0 uses `token_ids`, `positions`, `cache`, `logits_mode`, `hook`;
the rest arrive with their milestones.

### 5.2 Output & logits modes

- **`kLast`** returns `[1, V]` — the steady-state decode and greedy-loop path
  (M7-T02: "the core sampler over the final-position logits"). The model still
  runs all `T` positions through the layers; only the lm_head projection is
  restricted to the last token's hidden state — a real compute saving the
  reference takes because it is also correct.
- **`kAll`** returns `[T, V]` — a **real, supported mode**, not a test-only
  flag. M14-T03's `eval-ppl` needs all-position logits; M15's verification
  forward needs logits at every speculated position. The reference implements
  `kAll` by projecting every position's hidden state through lm_head. *(M15's
  finer need — logits at a per-sequence subset of positions — is left to
  M15-T04 to add as a selector field on `ForwardRequest`; `kAll` is its
  correctness reference.)*

Logits are always **fp32**, full vocab, contiguous per position (`[·, V]`
row-major) — M7-T06 batches sampling as `[num_seqs, V]`, so per-sequence
contiguity is fixed now. The returned `Tensor` is freshly allocated and owned
by the caller in M5; the contract states only that it is valid until the caller
drops it (M6 may return a view into a reused workspace — the doc's lifetime
promise is deliberately "until you drop the handle or call forward again,"
whichever the backend needs, and M6-T01 pins which).

### 5.3 Preconditions & error posture

Per ADR-003 and M9-T10 ("per-request errors fail only that request"),
malformed *inputs* are recoverable `Status`, never `CHECK`:

- `token_ids.size() == positions.size() == T`, `T ≥ 1` → else `InvalidArgument`.
- every `token_ids[t] ∈ [0, V)` → else `InvalidArgument` naming the offending
  index/value.
- `cache != nullptr` and its geometry equals `cache_geometry()` → else
  `InvalidArgument`. (A null cache is a programmer error at the call site, but
  surfacing it as `Status` keeps the batched M9 path — where one request's bad
  cache must not abort the batch — uniform. `CHECK` is reserved for internal
  invariant violations, e.g. a module receiving a wrong-rank tensor it built
  itself.)
- `max(positions) < config.max_position_embeddings` and
  `P + T ≤ cache.capacity()` → else `InvalidArgument` / `ResourceExhausted`
  (v0's contiguous cache has a fixed capacity set at construction; M8 makes
  exhaustion a block-pool concern with the same status code).

The cache is mutated *as a side effect* of a successful `forward`: on return,
`cache.length()` is `P + T`. On error, the cache is left unchanged (the append
happens per layer inside attention, so a mid-forward failure would be a `CHECK`
bug, not a partial commit — validation is front-loaded per ADR-003).

### 5.4 What later milestones add (recorded, not built)

So the contract is chosen to admit them without a signature break:

- **M9 batching:** token-major activations + `cu_seqlens` field; `forward`
  processes a ragged batch, output `[Σ_selected, V]`. v0's 2-D `[T, E]` flow is
  the single-sequence special case.
- **M12 decode allocation-free:** the `kLast`/`T==1` path must be
  heap-allocation-free once M6 provides workspaces (M12-T05). v0 allocates
  freely — clarity first — but the *interface* (caller owns logits, cache
  append is in-place) already permits it.
- **M14 activation capture / M16 spans:** the `hook` field (§11) is the single
  seam; nullptr is compile-checkable to zero cost.
- **M15 spec-verify + rollback:** `kAll` logits + the cache's `truncate` (§6.3).

---

## 6. KV cache — interface v0

### 6.1 The interface

Per-sequence, all layers, device-agnostic. One `KvCache` object holds the K/V
for **one sequence across all layers**; the engine owns one per active sequence
(M9-T02's `Sequence` holds a per-sequence cache handle, so the object is
per-sequence from day one, not a global store).

**This header (`src/kvcache/kv_cache.h`) landed with M5-T05, not T06**, because
the `Attention` module consumes it — its `forward` takes a `KvCache&`. T06 adds
the concrete `SimpleKvCache` (§6.2) and the token-by-token KV invariant test;
T05 exercised the interface through a test-local `FakeKvCache` double
(head-major storage, append transposes token-major `[T,Hkv,d]` → `[Hkv,len,d]`).

```cpp
// src/kvcache/kv_cache.h
struct CacheGeometry {
  int num_layers = 0;
  int num_kv_heads = 0;      // Hkv
  int head_dim = 0;          // d
  tensor::DataType dtype = tensor::DataType::kFloat32;  // v0: f32; M13 adds int8
};

struct KvView {                // a read view of one layer's accumulated K/V
  tensor::Tensor k;            // [Hkv, len, d], contiguous, head-major
  tensor::Tensor v;            // [Hkv, len, d]
};

class KvCache {
 public:
  virtual ~KvCache() = default;
  [[nodiscard]] virtual CacheGeometry geometry() const = 0;
  [[nodiscard]] virtual std::int64_t length() const = 0;    // committed tokens (all layers agree)
  [[nodiscard]] virtual std::int64_t capacity() const = 0;  // max tokens this cache can hold

  // Appends T new tokens' K/V for one layer. k,v: [T, Hkv, d]. Must be called
  // once per layer per forward, in layer order; advances that layer's fill.
  [[nodiscard]] virtual core::Status append(
      int layer, const tensor::Tensor& k, const tensor::Tensor& v) = 0;

  // Read view of layer's K/V over [0, current fill), including tokens appended
  // this forward. Valid until the next append to this layer.
  [[nodiscard]] virtual core::StatusOr<KvView> view(int layer) const = 0;

  // Drop everything after `new_length` tokens (per layer). truncate(0) == reset.
  // M15 rollback; also the general "current length" mutator.
  [[nodiscard]] virtual core::Status truncate(std::int64_t new_length) = 0;
  void reset() { (void)truncate(0); }
};
```

The four verbs the roadmap names for v0 — **append, view, current length,
reset** — map to `append`, `view`, `length`, and `truncate(0)`. `truncate` is
the one addition beyond the literal list, included now because M15-T04's cache
rollback needs it and v0 can provide it trivially (drop the tail); making
`reset` the `new_length == 0` case keeps the surface minimal.

### 6.2 `SimpleKvCache` (M5-T06)

Two `[num_layers, Hkv, capacity, d]` fp32 tensors (K and V), allocated once at
construction from `capacity` (caller-chosen: prompt length + max new tokens for
the single-request loop). `append(layer, k, v)` copies the `[T, Hkv, d]` block
into `[layer, ·, fill_layer : fill_layer+T, ·]` after transposing to head-major
`[Hkv, T, d]`; `view(layer)` **gathers** the layer's `[Hkv, fill_layer, d]`
history into a fresh contiguous tensor. Head-major storage (`[Hkv, len, d]`) is
chosen because M6-T05's decode attention reads one kv head's full history
contiguously; it matches HF's cache layout after `transpose(1, 2)`.

**Why `view()` gathers rather than slices** (M5-T06, refining an earlier draft
that said "zero-copy slice"): a `[Hkv, fill, d]` window of the
`[…, capacity, d]` store is inner-strided whenever `fill < capacity`, and
`cpu::attention` requires contiguous K/V. Copying keeps the T05 op contract
untouched, and it is exactly the seam §6.4 describes M8 keeping — v0's `view()`
*is* the reference's "gather cached K/V for a layer" helper; M8's is a
block-table gather; the `Attention` code above them is unchanged. The gather is
O(Hkv·fill·d), strictly cheaper than the attention it feeds. `length()` reports
the **minimum** per-layer fill (committed = what every layer agrees on),
well-defined mid-forward and after a failed forward.

**The acceptance invariant** (M5-T06): decoding token-by-token through the
cache produces logits **equal to a full-prompt recompute** at every step
(within fp32 tolerance). This is the fundamental KV-cache correctness property
and the test that makes the cache trustworthy for every later milestone.
Concretely: `forward([t0..tk], positions=[0..k], fresh cache, kAll)` and
`k+1` successive single-token `forward`s through a growing cache must produce
the same logits at each position.

Because `Model::forward` is M5-T07, T06 lands this invariant one level down —
at the **attention chain** (norms/MLP/residuals don't touch the cache, so the
chain fully exercises the cache's contribution) — and T07 elevates the same
check to full-model logits. In the reference it holds **bit-exactly**: each op
reduces a row in a single ascending fp32 accumulator, and a masked (softmax-0)
key contributes `0.0 · v == 0.0` exactly, so the full-prefill and
token-by-token schedules produce identical sums (observed max_abs_diff = 0). The same invariant reappears as M8-T07's
"identical to pre-paging engine," M9-T09's re-prefill equivalence, and
M15-T04's rollback-then-re-decode equality — all are this property under a
different storage backend.

### 6.3 The prefill-continuation contract

`forward` supports **prefill of `T` tokens against a cache of length `P`**
(M5-T05: "prefill against an existing cache of length P"). When `P > 0`,
attention's causal mask lets each of the `T` new queries attend to all `P`
cached positions plus the causal prefix of the new block — query at
new-position `j` (absolute position `P+j`) attends to cached `[0, P)` and new
`[0, j]`. This single path serves:

- **decode** (`T == 1`, `P` = history length),
- **prefill from empty** (`P == 0`),
- **prefix-cache hits** (M11: `P` = shared-prefix length, `T` = suffix; even
  the full-prompt-hit edge, `T == 1` re-computing the last token for logits),
- **chunked prefill** (M12-T06: repeated partial prefill, positions offset, `P`
  growing by the chunk size each time).

None of these need a new forward signature — they are `(P, T)` choices. The doc
states this so M11/M12 inherit it rather than re-deriving it.

### 6.4 What M8 (paged) changes — and what it does not

M5-T01's acceptance criterion demands this note explicitly. The v0 interface
above is the seam M8 keeps; the implementation is what changes.

**Unchanged (the abstract `KvCache` surface):** `geometry`, `length`,
`capacity`, `append`, `truncate`, and the per-sequence ownership model. M8-T07
"swaps the paged cache into the single-request generation path behind the M5
cache interface" and requires "tiny-fixture greedy output identical to
pre-paging" — so `KvCache` stays the polymorphic type; `PagedKvCache` is a
second implementation, and the generator/attention code that consumes the
interface does not change.

**Changed (the implementation, and one accessor):**

- **Storage:** contiguous per-sequence tensors → a shared **block pool**
  (`[num_blocks, 2, …]`, block size 16 default, M8-T01) drawn from M2's
  `CachingAllocator`, with a per-sequence **block table** (logical→physical
  block mapping, M8-T03) and **reference counting** from day one (so M11
  prefix-sharing is an extension, not a rewrite).
- **`append` becomes slot-mapping-driven:** M8-T03 computes a slot mapping (one
  physical slot per new token, allocating blocks on boundary crossings);
  M8-T04's kernel scatter-writes new K/V into paged storage given that mapping,
  "replacing the contiguous append path." So `append`'s *contract* (add `T`
  tokens for a layer) survives; its *mechanism* becomes gather/scatter.
- **`view()` is the accessor that does not survive paging.** A paged cache
  cannot hand back a single contiguous `[Hkv, len, d]` slice — the history is
  scattered across blocks. M8's attention reads through the block table
  directly (decode, M8-T05) or gathers cached K/V into a contiguous workspace
  before the M6 blocked-prefill kernel (prefill, M8-T06). Therefore **v0's
  attention is written to a private helper (`gather cached K/V for a layer`)
  rather than assuming `view()` returns storage it can pointer-walk** — v0's
  helper is `view()`, M8's helper is a block-table gather, and `Attention`'s
  code above them is unchanged. `view()` remains on the interface for tests and
  the reference; it is documented as "may copy" so paging can satisfy it.
- **Exhaustion:** v0's fixed capacity → M8's `ResourceExhausted` when the block
  pool is empty (M8-T08), plus a usage/stats surface (blocks used/free,
  utilization) that M16 reads. v0 already returns `ResourceExhausted` on
  over-capacity append, so the status code is fixed now.
- **Dtype:** v0 is fp32. M13-T07's INT8 KV cache makes `geometry().dtype`
  meaningful — reads no longer promise fp32; scales become cache-resident
  metadata and dequant moves inside the attention kernel. The `dtype` field
  exists in `CacheGeometry` now so that change is additive.

**Immutability (for M11).** v0 is **append-only**: a committed position's K/V
is never rewritten (only appended past, or dropped by `truncate`). This is the
property M11's prefix sharing rests on ("why shared blocks are immutable") — a
shared prefix block is safe to reference from multiple sequences precisely
because no one rewrites it. The interface states append-only as a contract, not
an accident.

---

## 7. RoPE specification (M5-T04)

Rotary position embeddings, matching HF Llama exactly (the fixtures are HF's
output, so "exactly" is the acceptance bar).

**Frequencies.** For `head_dim = d` (even), inverse frequencies
`inv_freq[i] = 1 / theta^(2i/d)` for `i ∈ [0, d/2)`, computed in fp32,
`theta = config.rope_theta`. Per position `p`, angle `p · inv_freq[i]`.

**Layout: half-rotation (HF `rotate_half`).** HF splits each head vector into
halves `[x_first_half | x_second_half]` of size `d/2` and rotates
`(x_j, x_{j+d/2})` as a pair by angle `p · inv_freq[j]`:

```
out[j]        = x[j]        · cos[p,j] − x[j+d/2] · sin[p,j]
out[j+d/2]    = x[j+d/2]    · cos[p,j] + x[j]     · sin[p,j]        for j ∈ [0, d/2)
```

where `cos[p,j] = cos(p · inv_freq[j])`, `sin` likewise. This is the
"half" (GPT-NeoX / HF-Llama) layout, **not** the interleaved-pairs layout —
choosing wrong is a silent correctness bug that only the goldens catch, hence
its explicit statement here.

**Tables.** Precompute `cos`/`sin` as `[max_position_embeddings, d/2]` fp32
tables at model build (or lazily up to the max position seen); `apply` indexes
by each token's absolute position from `ForwardRequest.positions`. Arbitrary
per-token positions (not a contiguous `0..T`) are required for M6-T03 and M11.
Only the first half `d/2` is stored — the half-rotation reuses `cos[p,j]` for
both elements of a pair. `inv_freq` and the angle `p · inv_freq[j]` are formed
in **fp64** for accuracy and stored fp32; HF forms them in fp32, so the tables
agree with the HF goldens as **Class T** (the fp32-`cosf` vs fp64-`cos` range
reduction diverges as the position grows — negligible at the tiny fixture's max
position 127, ~5e-3 at position 131071). `inv_freq` itself agrees to ~1 ulp, so
the scaling goldens (which target `inv_freq` directly) stay tight.

**`rope_scaling`** (parsed in M4, model-loading.md §3.2; interpreted here):

| `rope_type` | Effect on `inv_freq` / angle |
|---|---|
| absent / `"default"` | none (identity) |
| `"linear"` | `inv_freq /= factor` (position interpolation). Equivalent to scaling the angle `p / factor` since `angle = p · inv_freq`; the reference divides `inv_freq` to match HF's `_compute_linear_scaling_rope_parameters` arithmetic exactly. |
| `"llama3"` | HF's piecewise low/high-frequency wavelength scaling: below `low_freq_factor` wavelength unscaled, above `high_freq_factor` divided by `factor`, a smooth ramp between — the exact HF `_compute_llama3_parameters` formula, using `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings` from `RopeScaling` |
| anything else | `Unimplemented` at build (the executor rejects what it can't honor, model-loading.md §3.2) |

The tiny-llama fixture has no scaling (theta 10000, no `rope_scaling`), so the
llama3 path needs its own golden — a `cos`/`sin` table golden generated from
the Llama-3.1 config's scaling block (the config fixture already committed,
model-loading.md §3.2) validates the formula without a full scaled model.

---

## 8. The backend seam: reference vs optimized

`Model` (and `KvCache`) are the polymorphic contracts; **which implementation
runs is a construction-time choice**, invisible to the generation loop and the
tests above the interface.

```cpp
// src/engine/backend.h
enum class Backend { kReference, kOptimized };   // M5 implements kReference; M6 adds kOptimized
```

- **M5** ships `kReference` only: `ReferenceModel` + `ReferenceLinear` + `cpu::`
  ops. It is the **oracle** — deliberately never optimized (cpu-backend.md
  §2.1).
- **M6** ships `kOptimized`: a second `Model` implementation (repacked weights,
  per-thread workspaces, dispatched kernels) selected by the same `Backend`
  enum, validated token-for-token against `kReference` (M6-T07).
- **Shared vs per-backend.** The *module interfaces* (`Linear`, `Model`,
  `KvCache`) are shared. The reference `RmsNorm`/`Rope`/`Attention`/`Mlp`/
  `DecoderLayer` concrete classes are M5's; M6-T01 decides whether it reuses
  them with swapped `Linear`/ops or provides parallel implementations behind
  `Model`. **This doc commits only to the `Model`-level and `Linear`-level
  contracts** being the seam — it does not force M6 to reuse the reference's
  layer classes, because forcing structural reuse could compromise the
  reference's job as a clarity-first oracle. M6-T01 owns that call.
- **Selection is test-friendly:** `BuildModel` (§9) takes the `Backend`, so a
  single test can build both and assert equality (M6-T07's acceptance). A
  future third backend (Metal/MLX, roadmap "Future directions") enters the same
  way — the seam is implementation-shaped, not ISA- or device-shaped.

---

## 9. Architecture registry & builder (M5-T08)

```cpp
// src/model/registry.h
struct BuildOptions {
  Backend backend = Backend::kReference;
};

// Builds an executable Model from a loaded checkpoint. Consumes the weights.
[[nodiscard]] core::StatusOr<std::unique_ptr<Model>> BuildModel(
    LoadedModel model, const BuildOptions& options = {});

// Registration (M5-T08): keyed by the HF architecture string. Adding an
// architecture is one call — no switch to edit (the acceptance criterion).
using ModelBuilder = std::function<core::StatusOr<std::unique_ptr<Model>>(
    LoadedModel, const BuildOptions&)>;
void RegisterArchitecture(std::string_view hf_arch_name, ModelBuilder builder);
```

- **Keyed by `config.architecture_name`** (the raw HF string, e.g.
  `"LlamaForCausalLM"`, `"Qwen2ForCausalLM"`), which `ModelConfig` already
  carries. `Architecture` (the enum) selects config-parse behavior in M4;
  `architecture_name` selects the *builder* here. Unknown → `Unimplemented`
  listing the registered names. *(This is reachable even though
  `ParseModelConfig` rejects unknown architectures at load: the dummy-arch
  registry test constructs a `LoadedModel` directly, bypassing the parser — so
  "adding an architecture requires only a registration call" is verifiable in a
  test-local arch, M5-T08's acceptance.)*
- **Llama and Qwen2 share one family builder.** The only differences are
  `attention_bias` (Qwen2 biases q/k/v) and config field values — both flow
  through the same `DecoderLayer` construction. **M5-T10 registers Qwen2 with
  the same builder**; the diff is config/wiring, not new layer code.
- The builder validates the config↔registry consistency (head-dim, dtype) —
  M17-T04's "model-support matrix validated at startup" reads this as its
  entry point, though M5 only needs the arch-string check.

---

## 10. Generation loop (M5-T09)

```cpp
// src/engine/generator.h
struct GenerateOptions {
  std::int64_t max_new_tokens = 0;                 // hard cap; > 0 required
  std::vector<std::int32_t> eos_ids;               // stop when any is produced
};

// Greedy (argmax) continuation of `prompt_ids`. Returns the generated token
// ids (excluding the prompt). `on_token`, if set, is called with each new id
// as it is produced (the streaming seam; M10 uses it).
[[nodiscard]] core::StatusOr<std::vector<std::int32_t>> Generate(
    Model& model, KvCache& cache, std::span<const std::int32_t> prompt_ids,
    const GenerateOptions& options,
    FunctionRef<void(std::int32_t)> on_token = {});
```

- **Prefill** the whole prompt in one `forward` (positions `0..P-1`, `kLast`),
  argmax the `[1, V]` logits → first new token. **Decode**: one `forward` per
  step (`T == 1`, position = running length, `kLast`), argmax, append, repeat.
- **Greedy = argmax with lowest-index tie-break** (deterministic; the
  determinism-test acceptance requires two runs to be identical — trivially
  true for argmax, but the tie-break is stated so it is not left to
  `std::max_element`'s incidental behavior).
- **Stopping:** any `eos_ids` match, or `max_new_tokens` reached. `eos_ids`
  comes from the caller — the tokenizer's `eos_id()` and/or config. *(M5-T09
  adds `eos_token_id` parsing to `ModelConfig`: HF serializes it as an int or a
  list; the loop accepts a set. This is a small M4-config addition M5-T09 makes
  in passing, noted here so it is not a surprise.)*
- **Backend-agnostic:** the loop touches only `Model` and `KvCache`. M6-T07
  reuses it verbatim ("greedy loop reused from M5") against the optimized
  backend; the acceptance is token-for-token agreement with the reference. The
  golden test (M5-T09) is agreement with HF `generate(do_sample=False)` for ≥32
  tokens on tiny-llama.
- `on_token` is the **per-token callback hook** the roadmap names ("hooks for
  per-token callbacks, streaming later"); M10's SSE streaming and
  cancel-within-one-step build on it.

---

## 11. Debug & activation hooks (M5-T07)

A single seam serves M5's per-layer debug dumping, M14's calibration capture,
and M16's per-layer spans:

```cpp
// src/model/model.h
struct ActivationEvent {
  std::string_view name;   // "embeddings" | "layers.{i}" | "final_norm" |
                           // "logits" | "linear_input:<canonical weight name>"
  int layer = -1;          // -1 for non-layer stages
  const tensor::Tensor& tensor;  // borrowed; valid only during the call
};
class ActivationHook {
 public:
  virtual ~ActivationHook() = default;
  virtual void on_activation(const ActivationEvent&) = 0;
};
```

- **Names match the fixture keys** (`embeddings`, `layers.{i}`, `final_norm`,
  `logits`) so a debug hook can dump exactly the tensors the goldens hold and
  localize a per-layer regression (M5-T07's acceptance: "per-layer debug hook
  allows dumping intermediate activations").
- **`linear_input:<name>` events** expose the *input* to each `Linear` — this
  is what M14-T02's calibration needs ("per-layer input activations," streaming
  running statistics, "never materializing all activations"). The hook is a
  callback invoked with a *view*, not a dump-everything API, so memory-bounded
  accumulation is possible. **These events are deferred to M14-T02, not built in
  M5-T07** (2026-08-17): emitting them requires threading the hook through the
  `Linear`/`Attention`/`Mlp` forwards (signature changes to the modules landed
  in T02–T05), and T07's acceptance ("per-layer debug hook allows dumping
  intermediate activations") is met by the four stage events above. M14-T02
  adds an optional hook parameter to those forwards when it needs the
  linear-input granularity; the `ActivationEvent`/`ActivationHook` shape here is
  the seam it extends, unchanged.
- **Zero cost when off.** `hook == nullptr` (the default) means the forward
  pass makes no hook calls at all — a branch the optimizer removes. M16-T03
  needs the per-layer form to be "compile-time-gated to zero cost when off";
  the nullptr check is that gate for M5, and the event construction is guarded
  behind it so no tensor view or string is built when no hook is attached.

---

## 12. Golden fixtures for M5

M5 validates against HF goldens (the oracle-chain root, cpu-backend.md §6.1).
The existing tiny-llama `expected/activations.safetensors` (embeddings,
`layers.{i}`, final_norm, logits) covers the **end-to-end** M5-T07 test. M5
adds op-level and generation goldens; each is generated by a `tools/gen_fixtures/`
subcommand landed **with the ticket that consumes it** (not all up front), and
each obeys the model-loading.md §7.3 budget (≤ 5 MB/model fixture, ≤ 40 MB
total). Planned additions under `tests/fixtures/models/`:

| Fixture | Consumed by | Contents |
|---|---|---|
| `tiny-llama/expected/ops.safetensors` | T02–T04 | GEMM cases (incl. `k==1`, skinny, wide); RMSNorm on bf16 input → fp32 out; softmax on large-magnitude logits (e.g. ±1e4); SiLU-and-mul; `embedding_edge` — real bf16 embed table gathered at edge ids {0, V−1, duplicate} |
| `tiny-llama/expected/rope.safetensors` | T04 | `tiny_table` cos/sin `[max_pos, d/2]`; `tiny_sparse`/`tiny_contig` apply I/O in token-major `[T, H, d]` at positions {0, 1, 127} and {0..7} (GQA H=4/Hkv=2); `llama3` scaled `inv_freq` + cos/sin from the committed Llama-3.1 config (§7); `linear` scaled `inv_freq` + cos/sin (synthetic factor 4). The end-to-end embedding golden is the existing `activations.safetensors` `embeddings`/`input_ids`. |
| `tiny-llama/expected/attention.safetensors` | T05 | per case (layer, P, T): the module I/O (`x [T,E]`, `positions [T]`, `out [T,E]`) and the op intermediates (`q_rot [T,H,d]`, `k_all`/`v_all [Hkv,P+T,d]`, `ctx [T,H,d]`), plus `past_k`/`past_v [Hkv,P,d]` when P>0. Cases: `l{0,1}_prefill_empty` (P=0,T=8), `l{0,1}_prefill_continue` (P=5,T=6, HF-seeded past-key-values), `l0_decode` (P=7,T=1). GQA (Hkv=2 < H=4) is inherent; both layers appear (distinct weights). Intermediates captured from HF eager attention (a registered capture wrapper over `eager_attention_forward`). |
| `tiny-llama/expected/generate.json` | T09 | greedy `generate(do_sample=False)` continuation (≥ 32 ids) for a few fixed prompts |
| `tiny-qwen2/` (new model fixture) | T10 | a tiny `Qwen2ForCausalLM` mirror of tiny-llama: **q/k/v biases**, its config fields, tied or untied embeddings, with the same `expected/` set (logits + greedy) — exercises the Qwen2 wiring on the shared modules |

`tiny-qwen2` is deliberately small (same dims class as tiny-llama) and, to
exercise the decoupled path, may set `head_dim ≠ hidden_size / num_heads` — a
config the reference must handle (§3.1) and tiny-llama does not cover. The
generator, budget assertion, and `meta.json` follow tiny-llama's pattern
(`tools/gen_fixtures/tiny_llama.py`), and the fixtures README + model-loading.md
§7.1 layout gain the new files when they land.

*(model-loading.md §7.1 and `tests/fixtures/README.md` are updated with this
ticket to point forward to these additions; the files themselves land with
T02–T10.)*

---

## 13. Testing strategy

Per cpu-backend.md §6.2: M5 ops are **Class T** (reference-vs-fixture within a
stated tolerance). The reference `cpu` ops are also the pre-M6 oracle — M6's
kernels validate against them (bit-exact or tolerance per §6.3). M5 suites are
**not** ISA-dispatched (they call `cpu::` directly, not `kernels::`), so **no
`SCALAR_PASS` registration** and no per-ISA concern — ordinary portable tests,
identical on arm64 and x86-64. Every numerical test states its tolerance
explicitly and records the observed max-abs-diff (CLAUDE.md rule).

| Ticket | Tests |
|---|---|
| T02 | `cpu::gemm` vs fixture across shapes incl. `k==1`, skinny (`m≫n`), wide (`n≫m`); a 512×512×512 GEMM < 1 s (sanity); `ReferenceLinear` forward with/without bias; fp16 & bf16 weight → fp32 conversion path exercised |
| T03 | RMSNorm vs fixture (incl. **bf16 input**); SiLU-and-mul; residual add; softmax vs fixture with **large-magnitude logits** (numerical stability) — each tolerance stated |
| T04 | RoPE vs fixture at positions {0, 1, large}, config head dims; embedding lookup vs fixture rows; llama3-scaled table vs its golden |
| T05 | `cpu::attention` op vs fixture `ctx` (all cases, tol 1e-4; observed ≤4e-7) + bit-exact-vs-serial (threading), causal-mask-hides-future, GQA-interleave-not-tiling, 7 op error paths; `Attention` module vs fixture `out` (**prefill from empty**, **prefill from non-empty cache** P>0, decode T==1; tol 2e-4, observed ≤2e-5) with the accumulated cache view vs `k_all`/`v_all` (wiring, looser 1e-3 band) + `Create`/`forward` error paths (incl. over-capacity cache → ResourceExhausted); GQA (Hkv<H) covered throughout |
| T06 | **The KV invariant** (§6.2): token-by-token decode logits == full-prompt recompute at every step, within fp32 tolerance; `truncate`/`reset`/`view`/`length`/`capacity` unit behavior; over-capacity append → `ResourceExhausted` |
| T07 | **End-to-end** logits for tiny-llama vs `expected/activations.safetensors` (report max-abs-diff; threshold documented); per-layer debug hook dumps the layer tensors; `kLast` vs `kAll` consistency (last row of `kAll` == `kLast`) |
| T08 | Registry resolves `LlamaForCausalLM` and `Qwen2ForCausalLM`; unknown → `Unimplemented` listing supported; a test-local dummy arch registered by one call builds and runs (extensibility) |
| T09 | Greedy continuation matches HF `generate(do_sample=False)` token-for-token ≥ 32 tokens on tiny-llama; determinism (two runs identical); EOS and max-new-tokens stopping; `on_token` fires once per generated id |
| T10 | tiny-qwen2 golden logits + greedy generation pass, **reusing the M5 modules** (the diff under review must be config/registration, not new layer classes) |

Labels: an `engine`/`model`/`kvcache` label via `engine_add_tests(… LABELS)`,
mirroring `model`/`tokenizer`, so `ctest -L engine` iterates the execution
suite alone. Error-path tests assert the specific status code *and* that the
message names the offending input (the actionability criterion, tested).

---

## 14. Deferred (known, intentionally not designed here)

- **Optimized kernels, weight repacking, workspaces, allocation-free decode** —
  M6/M12, behind the `Model`/`Linear` contracts above.
- **Sampling** (temperature, top-k/p, penalties, logprobs, seeded RNG) — M7;
  M7-T01 appends §15 here.
- **Batching / ragged forward / cu_seqlens** — M9; `ForwardRequest` grows the
  fields (§5.1, §5.4).
- **Paged storage, block tables, refcounts, INT8 KV** — M8/M13; the `KvCache`
  interface is the seam (§6.4).
- **Per-position logits selector** (a subset of positions per sequence) —
  M15-T04; `kAll` is its reference (§5.2).
- **Cache rollback across block boundaries** — M15; v0's `truncate` is the
  interface, paged truncation is M8+M15's implementation.
- **Fused RoPE + KV-cache write** (M12-T04): the module decomposition keeps
  RoPE and append as separate observable steps *for the reference*; the fusion
  is a M6/M12 kernel behind attention, with the unfused reference path retained
  for testing — so §4.2's separation is a reference property, not a contract
  the optimized path must expose.
- **Startup model-support matrix** (arch/dtype/head-dim validation) — M17-T04
  extends the registry's config check (§9).

---

## 15. Sampling pipeline

*Placeholder — M7-T01 appends the sampling-pipeline design here (SamplingParams,
penalties, top-k/p, stop conditions, logprobs, seeded RNG), extending the
`Model::forward` logits contract of §5.2.*
