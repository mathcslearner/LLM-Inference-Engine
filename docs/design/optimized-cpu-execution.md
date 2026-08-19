# Optimized CPU execution: packed weights, vectorized kernels & workspaces

**Milestone:** M6 (design doc: M6-T01; implementation: M6-T02 … M6-T08)
**Governs:** the new optimized kernels in `src/kernels/` (GEMM/GEMV, norm,
activation, softmax, RoPE, attention, embedding/logits), the optimized `Model`
implementation in `src/model/` (packed `Linear`, workspace-driven layers), the
backend-selection wiring, and the kernel-validation methodology every optimized
kernel obeys — plus the layout/policy contract M8 (paged attention), M9
(batching), M12 (tuning/fusions), and M13 (quantized layers reuse the packed
tile) build on.
**Cites:** ADR-002 (module boundaries — the `model → kernels` edge is a
lower-layer edge, already allowed, §2.2), ADR-003 (error handling), ADR-004
(CPU-first pivot), `docs/design/cpu-backend.md` (threading, SIMD dispatch,
dtype/accumulation policy, oracle chain, numerics classes §6.3),
`docs/design/model-execution.md` (the `Model`/`Linear`/`KvCache` contracts M6
implements behind, §5/§6/§8), `docs/design/model-loading.md` (canonical weight
names §4, dtype-at-load policy §5), `docs/design/tensor.md` (Tensor/Buffer,
dtypes, half conversions, aligned allocation).

This is the working contract for the optimized CPU backend. Implementation
tickets must conform to it; if implementation reveals a design flaw, this doc is
updated in the same change with a note on what changed and why
(`docs/design/README.md`). M6-T02+ tickets amend in place.

---

## 1. Scope & non-goals

M5 delivered a correct, clarity-first scalar forward pass (the oracle). M6 makes
that forward pass **fast** without changing what it computes: packed-weight
multithreaded GEMM/GEMV, vectorized layer kernels, and blocked (flash-style)
attention — every optimized kernel validated against the M5 reference (which M5
validated against HuggingFace). The milestone ends with end-to-end generation on
a real ~1B model at usable speed and the first tokens/second baseline
(BASELINES.md). This is the analog of the retired v1 single-GPU milestone:
**deliberately straightforward optimized kernels behind stable interfaces** — the
aggressive tuning pass (tile autotuning, prefetch, native fp16/bf16 arithmetic,
flash-decoding split) is M12, behind these same interfaces.

**In scope (this doc):** the packed weight-tile layout and load-time repacking;
the dtype/accumulation policy for the optimized path (where conversion happens);
the threading integration (which loops parallelize, one pool, no
oversubscription); the workspace strategy (per-thread scratch and model-level
buffers, sized once, reused across layers); the optimized `Model`/`Linear`
structure and how it reuses the M5 module *interfaces*; the backend-selection
seam; and the kernel-validation methodology with a per-kernel tolerance/invariance
table (the acceptance-critical "bitwise vs tolerance, and why").

**Non-goals (of this doc, not the project):**

- **No tile/grain autotuning, no prefetch, no native-precision arithmetic.** The
  blocking constants (§3.5) are chosen once with a stated cache-budget rationale
  and left as named constants; M12-T02 tunes them behind the same kernels with a
  documented sweep. Native NEON FP16/BF16 FMA (which would change the
  fp32-accumulation policy) is explicitly M12/§dtype-amendment territory.
- **No flash-decoding cache split.** M6 decode attention threads across heads
  only; splitting one long context across threads with a deterministic reduction
  is M12-T03.
- **No kernel fusions.** RoPE + KV-write, residual + RMSNorm, and SwiGLU-combine
  fusions are M12-T04; M6 keeps them as separate observable steps (the reference
  decomposition, model-execution.md §4.2/§14).
- **No paged cache, no batching.** M6 runs one sequence through the abstract
  `KvCache` interface exactly as M5's reference does; `PagedKvCache` (M8) and
  ragged batches (M9) enter behind the interfaces §8 and §5 fix. §11 records what
  they change.
- **No quantized weights.** The packed tile layout (§3) is specified so M13's
  dequant-to-tiles path targets it, but INT8/INT4 numerics are M13's doc.
- **No new sampling.** The greedy loop (model-execution.md §10) is reused
  verbatim; it is already backend-agnostic.

---

## 2. Module layout, layering & ticket map

M6 adds optimized kernels to `kernels`, an optimized `Model` to `model`, and
touches `parallel` once (the worker-index overload, §6.4). No new module.

| Module | Path | Role in M6 | Tickets |
|---|---|---|---|
| `kernels` | `src/kernels/` | Optimized, dispatched (scalar/NEON/AVX2) kernels: packed GEMM/GEMV, RMSNorm, SiLU-and-mul, softmax, RoPE, prefill/decode attention, embedding/logits gather. Follows the cpu-backend.md §4.4/§4.5 per-ISA-TU pattern verbatim. | T02–T06 |
| `model` | `src/model/` | The **optimized `Model`**: `PackedLinear` (weights repacked at load, held behind the M5 `Linear` interface) and `OptimizedModel` (the workspace-driven forward pass). | T02, T06, T07 |
| `parallel` | `src/parallel/` | One additive change: a `parallel_for` overload whose body receives a worker index, so kernels can select per-thread scratch (§6.4). Determinism-safe (index selects scratch, never chunk assignment). | with T04 |
| `engine` | `src/engine/` | Unchanged. The greedy loop already touches only `Model`/`KvCache`; M6-T07 reuses it against `Backend::kOptimized`. | — |
| `benchmarks` | `benchmarks/` | Kernel microbenchmarks (GEMM vs reference) and `bench_generate` (end-to-end prefill/decode tok/s). | T02, T08 |

Files this milestone creates:

| File | Module | Contents | Ticket |
|---|---|---|---|
| `src/kernels/gemm.h` / `.cpp` + `internal/gemm_impl.h` + `{scalar,neon,avx2}/gemm.cpp` | kernels | packed-weight GEMM + GEMV entry points; the pack routine | T02 |
| `src/kernels/norm.h`, `activation.h`, `softmax.h`, `rope.h` (+ per-ISA TUs) | kernels | RMSNorm, SiLU-and-mul, numerically-stable softmax, RoPE-apply | T03 |
| `src/kernels/internal/exp_common.h`, `neon_exp.h`, `avx2_exp.h`, `exp_impl.h` + `exp.cpp` + `{scalar,neon,avx2}/exp.cpp` | kernels | the shared vector-`expf` polynomial (§10): a scalar spec + NEON/AVX2 lane helpers, plus array-form variants exposed only for the ulp sweep. Through M6 `exp` was not a public kernel — softmax/SiLU embed the lane helpers directly. (M7-T06 later added a public *unthreaded* `kernels::ExpF32` entry in `kernels/exp.h`, reused by the `sampling` module's reference softmax/log-softmax; it runs on the calling thread so it can be invoked inside a caller's own parallel region.) Residual add reuses the existing Class-E `kernels::AddF32` (no new kernel). | T03 |
| `src/kernels/attention.h` / per-ISA TUs | kernels | blocked prefill attention (online softmax) + decode attention | T04, T05 |
| `src/kernels/embedding.h` / `.cpp` (no per-ISA TU) | kernels | packed-layout-aware embedding lookup. **As-built (M6-T06): no per-ISA TU** — the only per-element work is the fp16/bf16→fp32 widen, dispatched through the M3-T06 `convert` variants; the gather itself is inherently scalar (§7). The lm_head logits path is `PackedLinear::forward` (T02) — no new kernel. | T06 |
| `src/model/optimized_embedding.h` / `.cpp` | model | `OptimizedEmbedding` — untied (`FromTable`) and tied (`FromPackedLinear`, sharing the packed lm_head's storage) lookup behind the M5 `Embedding` surface (§7) | T06 |
| `src/model/packed_linear.h` / `.cpp` | model | `PackedLinear` (repack at construction; GEMM/GEMV in `forward`) | T02 |
| `src/model/optimized_model.h` / `.cpp` | model | `OptimizedModel` + its workspace; the `kOptimized` family builder | T07 |
| `src/model/workspace.h` / `.cpp` | model | `Workspace` (model-level buffers + per-worker scratch, §6) | T07 |
| `src/main.cpp` (`engine generate`) | server-layer driver | real-model greedy-generation driver (load → tokenize → `Generate` → stream-detokenize); the §10 real-model acceptance harness until M9's server binary replaces it | T07 |
| `benchmarks/bench_generate.cpp` | benchmarks | prefill/decode tokens/sec harness (M6-T08) | T08 |

### 2.1 Layering — the edges M6 uses

Positions in the ADR-002 layer diagram (unchanged from Amendment 5; M6 adds no
new *listed* edge):

- **`model → kernels` is a layer-2 → layer-1 edge — already allowed, no ADR
  amendment.** ADR-002's rule is that a module "may depend on any module in a
  *lower* layer"; only *intra-layer* edges (like `model → kvcache`, Amendment 5)
  require an amendment. `model` (layer 2) depending on `kernels` (layer 1,
  compute substrate) is a downward edge, exactly like the existing
  `model → tensor`. The optimized `Model` calls dispatched kernels; the CMake
  link is **PRIVATE** (no public `model` header exposes a `kernels` type — the
  intrinsic-free kernel signatures are an implementation detail of
  `PackedLinear`/`OptimizedModel`).
- **The reference keeps its `cpu`-only rule.** `src/cpu/` still links
  `tensor`/`parallel` only, never `kernels` (cpu-backend.md §2.1) — the oracle
  must not depend on the kernels it validates. M6 does not touch `cpu`.
- `kernels → tensor, parallel` (existing). The new pack routine and kernels
  take raw pointers + shapes (intrinsic-free signatures), so `kernels`'
  public surface stays as it is; `PackedLinear` owns the `Tensor` that backs the
  packed bytes.
- `parallel → core` only (existing). The §6.4 overload adds no dependency.

### 2.2 Why parallel implementations, not reuse of the M5 layer classes

model-execution.md §8 deliberately left this call to M6-T01: whether the
optimized backend reuses the reference `RmsNorm`/`Rope`/`Attention`/`Mlp`/
`DecoderLayer` classes with swapped ops, or provides parallel implementations
behind the `Model` interface. **Decision: parallel implementations.** M6 reuses
only the two things that are genuinely shared contracts, not compute:

1. **The `Linear` interface** — `PackedLinear` is a new `Linear` implementation
   (model-execution.md §4.1 fixed `Linear` as an interface precisely for this and
   for M13's `QuantizedLinear`). `Attention`/`Mlp`'s `unique_ptr<Linear>` slots
   accept it unchanged.
2. **`Rope::Create`'s table construction** — the cos/sin/inv_freq computation is
   *config interpretation* (theta, `rope_scaling` → tables), not compute, and it
   is already validated against the scaling goldens (model-execution.md §7). The
   optimized attention reuses the produced tables and applies them with the
   vectorized `rope` kernel.

Everything else in the forward pass — the residual-stream orchestration, the
attention body, the MLP — is a **new `OptimizedModel` graph**, for three reasons
the reference cannot serve:

- **The reference allocates freely** (`Mlp::forward` allocates fresh work
  tensors, model-execution.md §4.2); the optimized path must run out of a
  preallocated `Workspace` (§6) so steady-state decode is allocation-free
  (M12-T05's target, reachable now).
- **The oracle must stay untouched.** cpu-backend.md §2.1: "an oracle you
  optimize is an oracle you no longer trust." Threading workspaces and kernel
  calls through the reference classes would optimize them.
- **Fusion-readiness.** M12-T04 fuses residual+norm and RoPE+KV-write; the
  optimized graph is structured (explicit workspace slots, kernels called at the
  layer level) so those fusions are local edits, not a rewrite.

The **cost** — two graph implementations to keep in step — is bounded by the
equality test that is M6-T07's whole point: `OptimizedModel` and `ReferenceModel`
must produce token-for-token identical greedy output and within-tolerance logits
on the fixtures, so a divergence is caught mechanically, not by inspection.

---

## 3. Weight layout policy — the packed tile (M6-T02)

### 3.1 The source of truth is unchanged

cpu-backend.md §7 imposes one constraint M6 inherits: **the checkpoint-order
tensor `[out, in]` (row-major, model-loading.md §4) remains the source of
truth, and any packed form is derived from it** — so the `cpu` oracle and the
optimized path provably read identical weight *values*. `PackedLinear::Create`
takes the same canonical `[N=out, K=in]` weight the reference takes, produces the
packed tensor, and then may drop the checkpoint handle (§4 — the packed copy is
authoritative for the optimized path once built).

### 3.2 The layout: K-major panels of `NR` output rows

For a projection weight `W[N, K]` (`N = out_features`, `K = in_features`), the
packed form `Wp` has shape **`[P, K, NR]`** with:

- `NR` = **16, a fixed constant across all ISAs** (`kNr`),
- `P = ceil(N / NR)` output-row panels,
- `Wp[p, k, r] = W[p·NR + r, k]` when `p·NR + r < N`, else **`0`** (the last
  panel is zero-padded to a full `NR`).

Element `(p, k, r)` sits at flat offset `((p·K) + k)·NR + r`. The **inner
dimension is the panel's `NR` output rows; the middle dimension is `K`**. So for
a fixed panel `p`, walking `k = 0..K−1` yields consecutive `NR`-wide contiguous
vectors `Wp[p, k, :]` — one aligned vector load per `k`, streamed once. This is
the classic "pack B into K-major panels" GEMM layout, transposed to match our
`[out, in]` storage: the panel is `NR` *output* channels, and a vector load
delivers the `NR` partial products that one activation element `a[m, k]`
contributes to.

**Why `NR = 16, fixed across ISAs.** The same rationale as cpu-backend.md §6.3's
16-lane Class-R convention: 16 fp32 lanes is the natural unroll on both targets
(NEON `4 × float32x4_t`, AVX2 `2 × __m256`), and *fixing* it means the
forced-scalar test pass (§9) processes the **same packed bytes** the vector
passes do — the packed layout is not itself ISA-dependent, only the micro-kernel
that reads it is. A per-ISA panel width would fork the packed bytes and make the
scalar pass validate a different memory image than ships.

**Dtype.** `Wp` is held in the **checkpoint's storage dtype** (bf16/f16/f32 —
§4), shape `[P, K, NR]`, allocated 64-byte aligned via `Tensor::empty` (a fresh
buffer; the panel stride `K·NR` and the `NR`-element inner run are what alignment
buys). Zero-pad uses the dtype's zero bit-pattern (`+0.0`), which contributes
`a[m,k]·0 = 0` exactly to the padded output rows, and those rows `[N, P·NR)` are
never read back.

### 3.3 Worked example

*Illustrative (small `NR` for readability):* `W` is `[N=5, K=3]`, packed with
`NR = 4` → `P = ceil(5/4) = 2` panels, `Wp` shape `[2, 3, 4]`:

```
W (row-major [5,3]):                 Wp (packed [P=2, K=3, NR=4]):
  row0: W00 W01 W02                   panel 0 (rows 0..3):
  row1: W10 W11 W12                     k=0: [W00 W10 W20 W30]
  row2: W20 W21 W22                     k=1: [W01 W11 W21 W31]
  row3: W30 W31 W32                     k=2: [W02 W12 W22 W32]
  row4: W40 W41 W42                   panel 1 (rows 4..7, rows 5..7 zero-pad):
                                        k=0: [W40  0   0   0 ]
                                        k=1: [W41  0   0   0 ]
                                        k=2: [W42  0   0   0 ]
```

Flat memory order of `Wp`: `W00 W10 W20 W30  W01 W11 W21 W31  W02 W12 W22 W32
W40 0 0 0  W41 0 0 0  W42 0 0 0`. Reading panel 0 at `k=1` is the single aligned
4-vector `[W01 W11 W21 W31]` — the four output channels' weights for input
channel 1.

*Real (shipping `NR = 16`):* tiny-llama `q_proj` is `[N=64, K=64]` →
`P = 64/16 = 4` panels, `Wp` shape `[4, 64, 16]`, 4096 elements (bf16 → 8 KB),
no padding (`64 % 16 == 0`). tiny-qwen2 `q_proj` is `[N=96, K=64]` (decoupled
`head_dim=24`, model-execution.md §3.1) → `P = 6`, `Wp` `[6, 64, 16]`, again no
pad. Llama-3.2-1B `gate_proj` `[N=8192, K=2048]` → `P = 512`, `Wp`
`[512, 2048, 16]` (bf16 → 32 MB, same bytes as the checkpoint tensor). A weight
whose `N` is not a multiple of 16 (e.g. an lm_head with an odd vocab tail) pads
its last panel; the padded rows never reach the returned logits.

### 3.4 The micro-kernel and blocking

`PackedLinear::forward(x[M, K], y[M, N])` computes `y = x · Wᵀ (+ bias)` with `x`
fp32 activations. The micro-kernel holds an `MR × NR` block of **fp32 register
accumulators** and, for each `k`, broadcasts each of `MR` activation values
`a[m0+mr, k]` across the `NR` lanes of `Wp[p, k, :]` (widened to fp32 in-register,
§4) and FMA-accumulates:

```
for panel p, for m-block [m0, m0+MR):
  acc[mr][nr] = 0                                     # MR×NR fp32 registers
  for k in [0, K):
    wv = widen_to_f32( Wp[p, k, 0:NR] )               # one NR-wide vector load
    for mr in [0, MR):
      acc[mr] += broadcast(a[m0+mr, k]) * wv          # FMA into NR lanes
  store acc[mr][0:NR] -> y[m0+mr, p·NR : p·NR+NR]      # + bias, mask padded rows
```

- `MR` is per-ISA (`kMr = 4` on NEON, `6` on AVX2 — a register-budget choice:
  `MR` broadcast registers + `MR·(NR/lanes)` accumulators must fit the 32×128-bit
  NEON / 16×256-bit AVX2 register files). `NR = 16` is fixed (§3.2). Unlike a
  reduction, **`MR` does not affect results** — each `y[m,n]` still accumulates
  its `K` products in one fp32 register in ascending-`k` order regardless of the
  block shape.
- The **single output element is one fp32 accumulator, ascending `k`, never split
  across threads** — this is what makes the optimized GEMM bit-identical across
  thread counts (§10), though *not* across ISAs (FMA contraction differs).
- **One packed layout serves all three shapes:** GEMM (`M` large, prefill),
  skinny (`M ≤ MR`), and GEMV (`M == 1`, decode). GEMV is the micro-kernel with
  `MR = 1` — one broadcast, `NR`-wide FMA, streaming each output panel's weights
  once. No separate packed layout for decode (the roadmap's "decode-shaped GEMV
  path" is a code path, not a second layout).

**Cache blocking.** For large `M, N, K` the loops are blocked
`kMc × kNc × kKc` (named constants, `model-execution`-style rationale, tuned in
M12-T02) so a `kKc`-tall slab of the packed panel and a `kMc × kKc` slab of `A`
stay resident. Blocking changes *traversal order only*, never the per-element
reduction order (each `y[m,n]` reloads its running fp32 accumulator from `y` when
re-entering a `k`-block, so the sum is still one ascending walk) — bit-identical
regardless of block sizes, exactly as `cpu::gemm`'s tiling is (`src/cpu/gemm.cpp`).
Initial constants target the two hosts: dev machine Apple M2 (128 KB L1d, 16 MB
shared L2), CI x86-64 (typically 32–48 KB L1d, 256 KB–1 MB L2) — `kKc` sized so a
`kKc × NR` bf16 panel stripe (`kKc·16·2` bytes) plus the `MR × kKc` fp32 A slab
sit in L1d on the smaller (x86) target; `kNc`/`kMc` sized to L2. These are
starting points with a stated budget, not tuned numbers (that is M12-T02, with a
sweep).

> **M6-T02 implementation note (amends this section).** As built
> (`src/kernels/gemm.{h,cpp}`, `{scalar,neon,avx2}/gemm.cpp`): the ISA variant
> is a **tile function** — it computes one output tile *rows `[m0,m1)` × panels
> `[p0,p1)` over all K* and is the single-threaded chunk body; the tile grid and
> dtype dispatch live in `gemm.cpp`. Within a tile the micro-kernel holds the
> **full-K accumulation in `MR × NR` registers** (`MR = 4` NEON `kMr`, `6` AVX2
> `kMr`; `NR = kNr = 16`) and stores once — it does **not** re-enter `y` per
> `k`-block. This is the same ascending-`k` single-accumulator order the section
> requires (so still bit-identical across thread counts and tile shapes, §10),
> just realized register-resident rather than via a `kKc`-reload from `y`; the
> reload form the paragraph sketched is one valid alternative, and an explicit
> `kKc` loop stays available to M12-T02's sweep. Blocking is therefore only
> **`kMc = 64` rows × `kNc = 8` panels** per parallel tile (grain
> `kGemmTileGrain = 1`), sized so a panel-block's weights stay hot across the
> m-blocks that reuse them; K is streamed once per tile. GEMV (decode) is
> `PackedGemm` routing `M == 1` to `PackedGemv`, which threads panel chunks
> (`kGemvPanelGrain = 8`) with the same tile variant at `m1 = 1` — one packed
> layout, a code path, not a second layout (§3.4). Measured 8.76× the naive
> `cpu::gemm` at 4096³ bf16 (BASELINES.md), clearing the ≥5× target.

### 3.5 Bias

Bias `[N]` is converted to **fp32 once at pack time** (stored alongside `Wp` as a
fresh fp32 `[N]` tensor) and added after the `K` accumulation completes — matching
`cpu::gemm`'s "added once after the K accumulation, not seeded into the
accumulator" rounding (`src/cpu/ops.h`). Per-projection presence is the builder's
decision, unchanged from the reference (Qwen2 biases q/k/v; model-execution.md
§4.1).

---

## 4. Dtype & accumulation policy

Restates cpu-backend.md §5 for the optimized path, pinning where conversion
happens:

- **fp32 accumulation everywhere** — GEMM/GEMV accumulators, RMSNorm sums,
  softmax, attention score/context sums — regardless of weight storage dtype. No
  fp16/bf16 accumulation (that is the M12 native-precision revisit, a dtype-policy
  amendment with its own error analysis, cpu-backend.md §10).
- **Activations are fp32 throughout**, exactly as in the reference — every buffer
  in the residual stream and every workspace tensor (§6) is fp32.
- **Packed weights keep the checkpoint dtype; widening is in-register at the
  kernel boundary.** A bf16/f16 checkpoint weight stays bf16/f16 in `Wp`
  (model-loading.md §5 preserves it; M4 never up-converts); the micro-kernel
  widens `Wp[p,k,:]` to fp32 *as it loads it*, using the ISA's conversion path
  (NEON `vcvt`/`fcvt`, AVX2 F16C `vcvtph2ps` for f16; a 16-bit left-shift for
  bf16). This conversion must match `tensor/half.h` on all **finite** inputs —
  which the hardware paths do (cpu-backend.md §5, the F16C/`FCVTL` analysis) — and
  weight values are finite by construction (a NaN/Inf weight is garbage-in, a
  stated precondition, not a correctness obligation the pack must police, unlike
  the exhaustive `half.h` NaN-payload contract the M3-T06 *conversion kernels*
  owe).
- **fp32 checkpoints stay fp32-packed — no narrowing at load.** An fp32 checkpoint
  is packed as fp32 (`Wp` dtype = f32); M6 never lossily narrows weights to save
  memory. Consequence: for an fp32 checkpoint the optimized and reference paths
  read *bit-identical* weight bits, so any logit difference is purely
  accumulation-order (FMA/blocking), never a weight-rounding difference — which
  keeps §10's tolerance analysis clean.
- **Small tensors converted once at load, not per-use:** norm scale weights
  `[E]`, biases `[N]` → fp32 at module construction (they are read every token;
  converting once is both faster and removes a per-element widen from the hot
  path). The embedding table is the one exception (§7): it may stay in checkpoint
  dtype and widen at gather time.
- **KV cache stays fp32** (`CacheGeometry.dtype`, model-execution.md §6.1); logits
  are fp32 full-vocab (§5.2 of model-execution.md). INT8 KV (M13-T07) is additive
  behind the same geometry field.

---

## 5. Threading integration

One `parallel::DefaultPool()` (cpu-backend.md §3.1), one level of parallelism, no
nesting (CHECK-enforced), no allocation or `Status`-validation inside a region
(all front-loaded, ADR-003). Each kernel's public entry decides — above its named
grain — to run its ISA variant as chunk bodies inside `parallel_for`; below it,
inline on the caller (cpu-backend.md §4.2). The variant is the single-threaded
chunk body either way.

| Kernel | Parallelized over | Grain constant | Notes |
|---|---|---|---|
| GEMM (prefill) | flattened `(m-block, n-panel-block)` tiles | `kGemmTileGrain` | 2-D tiling; each tile is an independent `MR×NR`-accumulator region |
| GEMV (decode, `M`≤thresh) | output-panel chunks | `kGemvPanelGrain` | each thread streams a disjoint contiguous run of packed panels once (bandwidth-bound; §3.4) |
| RMSNorm, SiLU-mul, residual add | rows (`T`) | `kRowGrain` | rows independent; per-row reduction (RMSNorm) in one fp32 accumulator |
| softmax | rows | `kRowGrain` | per-row max+sum reductions in fp32, in-row |
| RoPE apply | tokens (`T`) | `kRowGrain` | pure per-element map |
| prefill attention | `(head, query-block)` pairs | `kAttnQGrain` | online softmax per (head, q-block); GQA head `h` reads kv head `h/g` |
| decode attention | kv heads (`Hkv`) | `kAttnHeadGrain` | the `g` query heads of a kv group share that head's streamed K/V |
| embedding / logits gather | tokens / output panels | `kRowGrain` / `kGemvPanelGrain` | §7 |

**The determinism rule for the whole backend:** *no reduction is ever split
across threads except through `parallel_reduce`'s fixed tree.* Every kernel above
either has independent outputs (GEMM tiles, rows, heads) or reduces within a
single thread (each `y[m,n]`'s K-dot, each row's norm/softmax sum, each attention
output's context sum). So results are **bit-identical across thread counts** by
construction — the property §10's tests assert, and the invariant M12-T03's
flash-decoding split must preserve (it will use `parallel_reduce` over the cache
split, the one sanctioned way to reduce across threads).

**No caller-participation, no busy-wait assumptions** — M6 inherits the M3-T04
pool as-is (one core briefly idle per region is accepted, cpu-backend.md §3.2/§10;
revisited with the M6-T08 benchmark, not here). *M6-T08 update: the more
consequential pool finding is not the briefly-idle core but the M2's
**heterogeneous cores** — the 4 efficiency cores throttle the memory-bound
decode step (both our engine and llama.cpp decode faster on 4 P-cores than 8
mixed cores; BASELINES.md M6-T08). The default pool sizes to all physical cores;
an E-core-aware or P-core-only decode pool is an M12 tuning lever, not an M6
change.*

---

## 6. Workspace strategy

### 6.1 What the workspace holds

`OptimizedModel` owns one `Workspace`, reused across all layers of a forward
(buffers are sized for one layer's activations; layer `i+1` overwrites layer
`i`'s). Two kinds of storage:

- **Model-level buffers** (shared, single-threaded orchestration): the
  residual-stream tensors and per-layer intermediates —
  - residual stream: `x`/`h`/`attn_out`/`r`/`mlp_norm_out`/`mlp_out`, each `[T, E]`
    fp32 (a small fixed count `c_stream ≈ 4` live at once, reused);
  - projections: `q [T, H·d]`, `k [T, Hkv·d]`, `v [T, Hkv·d]`, `ctx [T, H·d]`;
  - MLP: `gate [T, I]`, `up [T, I]`.
- **Per-worker scratch** — *designed here, but as-built (M6-T04) the prefill
  attention needs none, and §6.4's worker-index overload was therefore not
  added.* The reason: the online-softmax accumulator **is `out` itself**. Each
  `(head, query-block)` unit writes only its own `out[t, h, :]` rows (disjoint
  from every other unit), so the running weighted-V accumulator is rescaled in
  place there and divided by the denominator at the end — no external
  `[kQb, d]` buffer. The only working memory is one query's `kAttnKb`-wide score
  row, a fixed stack array in the variant. Running max/sum are two stack scalars
  per query. Decode (M6-T05) is expected to accumulate in `out` the same way.
  So `W_worker = 0` in practice; the mechanism (a per-thread scratch slice
  selected by worker index) is kept in this section only as the shape a *future*
  kernel that genuinely needs cross-`d` scratch would use — see §6.4.

The **K/V for this call** are formed token-major (`[T, Hkv, d]`) in the `k`/`v`
model-level buffers, appended via `cache.append` (interface unchanged), and the
accumulated K/V are read back via `cache.view` (§8). The **logits** are a
freshly-allocated caller-owned tensor, *not* a workspace buffer (§6.3). The
`cache.view` gather allocates inside the cache in M6 (the `SimpleKvCache` gather,
model-execution.md §6.2); §8 quantifies that cost and §11 records M8 removing it.

### 6.2 Sizing formula

Let `T` = tokens this call, `E` = hidden, `H`/`Hkv` = query/kv heads, `d` =
head_dim, `I` = intermediate, `nthr` = pool threads, and `kQb`/`kKb`/`kMr` the
attention/GEMM block constants. Workspace bytes (fp32 ⇒ ×4):

```
W_model  = 4 · [ c_stream·T·E                      (residual stream)
               + T·(H + 2·Hkv)·d                   (q, k, v)
               + T·H·d                              (ctx)
               + 2·T·I ]                            (gate, up)

W_worker = 0                                        (as built, M6-T04: the
                                                     attention accumulator is
                                                     `out`; §6.1)

Workspace(T) = W_model
```

(The `W_worker = 4·nthr·[kQb·kKb + kQb·(d+2)]` term the first draft carried is
zero as built — §6.1 explains why the prefill/decode attention keeps no
per-worker scratch. It is retained only as the sizing a future scratch-needing
kernel would follow.)

- **Instantiated, tiny-llama** (`T=8, E=64, H=4, Hkv=2, d=16, I=176`,
  `c_stream=4`): `W_model = 4·[2048 + 1024 + 512 + 2816] = 4·6400 ≈ 25.6 KB`;
  worker scratch negligible. The whole forward runs in ~26 KB of workspace.
- **Instantiated, Llama-3.2-1B** (`T=512, E=2048, H=32, Hkv=8, d=64, I=8192`,
  `nthr=8`): `W_model = 4·[4.19M + 1.57M + 1.05M + 8.39M] ≈ 60.8 MB` at a
  512-token prefill; decode (`T=1`) is ~120 KB. Worker scratch is zero as built
  (§6.1). So a 1B-model prefill needs ~61 MB of workspace, dwarfed by the
  ~2.5 GB of bf16 weights — comfortable on the 16 GB dev machine.

The formula is what `Workspace::EnsureCapacity(T, geometry)` computes; `V`
(vocab) and `L` (cache length) do **not** enter workspace size — logits are
caller-owned (§6.3) and the K/V gather is the cache's allocation (§8).

### 6.3 Grow-on-demand with a high-water mark; logits caller-owned

**Sizing policy: monotone grow-on-demand.** The workspace sizes itself on the
first forward of a given `T` (prefill sizes it large; decode's `T=1` fits within
any prior prefill's buffers), grows if a later call needs more, and **never
shrinks**. Growth failure surfaces as `ResourceExhausted` (front-loaded, before
any kernel runs). Steady-state decode therefore performs **zero heap allocations
per step** once the first decode has sized the (tiny) `T=1` workspace — M12-T05's
allocation-free-decode target, reachable now without a config knob. The
alternative — pre-sizing from a `BuildOptions::max_forward_tokens` bound — is
**recorded and deferred to M9**, when `--max-num-batched-tokens` makes the bound a
real config input; M6 has no scheduler to supply it, so grow-on-demand is both
simpler and sufficient.

**Logits are freshly allocated and caller-owned**, identical to the reference
(model-execution.md §5.2 deliberately left the optimized path's logits lifetime
to M6-T01). `forward` returns a fresh `[1, V]` (kLast) / `[T, V]` (kAll) tensor
the caller owns until it drops the handle — *not* a view into the workspace. Two
reasons: (1) it keeps **one lifetime contract for both backends**, so the
generation loop and every test above the interface treat the two identically;
(2) the greedy loop reads the logits and then calls `forward` again, so a
workspace-view would be clobbered by the next call anyway. M12-T05 may later add
an optional caller-supplied output-buffer field to `ForwardRequest` (additive, to
eliminate the per-step logits allocation in a tight server loop) — but that is an
optimization behind an unchanged default, not M6's contract.

**As-built (M6-T07):** `Workspace` (`src/model/workspace.{h,cpp}`) holds ten
model-level slots as separate contiguous fp32 tensors — the four `c_stream`
E-width buffers `x`/`h`/`tmp`/`r` (residual stream, norm output, projection
output, post-attention residual), the projections `q [T,H·d]`/`k`/`v [T,Hkv·d]`/
`ctx [T,H·d]`, and the MLP `gate`/`up [T,I]` — allocated with `Tensor::empty`
(uninitialized: every slot is written by a kernel before it is read) and exposed
as `[T,width]` prefix views. `EnsureCapacity(T)` allocates all ten into locals
and commits only when all succeed (a mid-way OOM leaves the prior buffers intact,
front-loaded before any kernel or cache touch). Per-slot tensors, not one arena
+ offsets, chosen for simplicity — same guarantees. `bytes()` computes the §6.2
formula and is asserted against the instantiated tiny-llama 25.6 KB and its
monotone growth. `W_worker = 0` as designed (the attention accumulator is `out`).

### 6.4 The `parallel_for` worker-index overload

**Status (M6-T04): NOT added — deferred until a kernel actually needs it.** The
draft planned this overload for the attention kernels' per-worker scratch, but
§6.1 shows they keep none (the accumulator is `out`). Adding an unused
`parallel_for` overload + threading a worker index through `ThreadPool::Run`
just to leave it uncalled is speculative surface, so M6-T04 left
`src/parallel/` untouched. The design below is retained as the ready mechanism
for the first kernel that genuinely needs cross-`d` per-thread scratch (an M12
fusion, or a decode split that does not use `parallel_reduce`); that ticket adds
it. The `cpu-backend.md` §3.2 note is updated to match.

The mechanism, when needed: per-thread scratch needs each chunk body to know
*which* worker's scratch slice to use. The current `parallel_for` body is
`void(int64 begin, int64 end)` — no worker identity
(`src/parallel/parallel_for.h`). One overload delivers it:

```cpp
// src/parallel/parallel_for.h  (added with M6-T04)
// worker ∈ [0, pool.num_threads()); the inline (num_chunks ≤ 1) path passes 0.
void parallel_for(ThreadPool& pool, std::int64_t n, std::int64_t grain,
                  FunctionRef<void(int worker, std::int64_t begin,
                                   std::int64_t end)> body);
```

- **Determinism-safe:** the worker index selects *scratch*, never *chunk
  assignment* — chunk boundaries remain a pure function of `(n, grain)`
  (cpu-backend.md §3.2), and the outputs a kernel writes are still disjoint per
  chunk. The index only routes a thread to its private scratch buffer, so results
  are unchanged whatever the schedule. (This is why it is safe even though
  cpu-backend.md §10 may later change the round-robin schedule or add caller
  participation: nothing about correctness depends on *which* index a chunk runs
  under.)
- **Why not the alternatives:** deriving a worker id from `chunk_id % nthr`
  couples kernels to the round-robin schedule §10 reserves the right to change;
  per-chunk scratch makes scratch memory scale with `num_chunks` (unbounded)
  rather than `nthr`; a `thread_local` grown lazily inside the region allocates
  inside a parallel region (forbidden, cpu-backend.md §3.4). A worker index passed
  by the pool — which already assigns `worker w of T` (thread_pool.h) — is the
  minimal, allocation-free mechanism.
- cpu-backend.md §3.2 gets an amendment note pointing here; the existing
  `void(begin, end)` overload stays (most kernels — rows, tiles — need no worker
  id).

---

## 7. Embedding & logits path; tied weights (M6-T06)

The embedding lookup and the lm_head projection are the two ends of the model
that may **share one logical weight** (`config.tie_word_embeddings`). M4 stores
the *same* `Tensor` handle under both `embed_tokens.weight` and `lm_head.weight`
(model-execution.md §4.3); the optimized backend must honor tying without a
`[V, E]` duplicate, because for a tied model that duplicate is expensive
(Qwen2.5-0.5B: `V=151936, E=896`, bf16 → **~272 MB**, ~27% of the model).

**Decision: one physical copy — the packed lm_head — and the lookup gathers from
it.**

- **Tied** (`tie_word_embeddings`): build one `PackedLinear` for `lm_head` (packed
  `[P=ceil(V/16), E, 16]`, §3). The embedding lookup for token id `v` gathers row
  `v` **out of the packed layout**: row `v` lives in panel `v/16` at lane
  `v%16`, i.e. its `E` elements are at `Wp[v/16, k, v%16]` for `k = 0..E−1`
  (stride `NR = 16`, widened to fp32). That is a strided gather of `E` elements
  per token — `~E/lanes` cache lines touched, trivially cheaper than the GEMMs it
  sits between — and it is exactly the roadmap's "ids → packed rows, handling
  repacked layouts." No second copy of the table exists.
- **Untied:** the embedding table is its own weight. It is **not** packed (the
  lookup is a gather, not a matmul — packing buys nothing) and stays in checkpoint
  dtype as a zero-copy `[V, E]` handle; the lookup is the contiguous
  gather-and-widen (bit-exact to `cpu::embedding_lookup`). The lm_head is a
  separate `PackedLinear`.

So there is **one embedding kernel with two source layouts** (packed-strided for
tied, contiguous for untied), selected at build. The alternative — always gather
from the mmap'd checkpoint-order table even when tied — is **rejected**: it keeps
the checkpoint file pages resident purely for the lookup (defeating the point of
dropping the checkpoint handle after packing, §3.1) and forks the tied path from
the untied one for no benefit. The **logits** path is `PackedLinear::forward` in
GEMM shape (kAll) or GEMV shape (kLast, the last token's hidden state only — the
reference's compute saving, model-execution.md §5.2), producing fp32 `[T or 1, V]`.

**Validation:** the lookup is **bit-exact** vs `cpu::embedding_lookup` for random
id sets including repeated ids and edge ids `{0, V−1}` (a pure gather + exact
widen — no accumulation, so no tolerance); the logits match the reference within
the GEMM tolerance (§10). A dedicated test builds a tied model and asserts the
gathered embedding row `v` equals the packed lm_head's logical row `v`
(the shared-storage-across-layouts property, model-execution.md §4.3).

**As-built (M6-T06).**

- **Two kernel entries, one widen path** (`src/kernels/embedding.{h,cpp}`):
  `EmbeddingLookupF32` (row-major `[V, E]` source) and
  `EmbeddingLookupPackedF32` (packed `[PackedPanels(V), E, kNr]` source). Both
  thread over tokens (`kRowGrain = 1`); each output row is written by exactly
  one worker, so the result is bit-identical across thread counts. The
  contiguous (row-major) path widens a whole row in one call; the strided
  (packed) path gathers each row's `kNr`-strided 16-bit lanes into a fixed stack
  buffer a `kGatherChunk`-wide chunk at a time, then widens the chunk — the
  widen is Class E (chunk-invariant), so chunking is bit-exact.
- **No per-ISA TU.** The gather has no ISA-specific control flow; its only
  per-element arithmetic is the fp16/bf16→fp32 widen, which the M3-T06 `convert`
  variants (`scalar/neon/avx2::{Bf16,Fp16}ToFp32`) already provide bit-exact per
  `half.h`. So `embedding.cpp` builds a `KernelTable<WidenFn>` from those
  variants and `Select`s once — `ENGINE_FORCE_ISA` still selects the widen path,
  keeping the forced-scalar pass honest. This amends the §2 file-table entry
  (which sketched per-ISA TUs); a blind AVX2 embedding TU would be a pure
  wrapper for no benefit. f32 tables need no widen (a `memcpy`/strided copy).
- **Storage sharing is a `Tensor` handle, not a raw borrow.** The tied
  `OptimizedEmbedding::FromPackedLinear` holds `PackedLinear::packed_weight()`
  **by value** — the refcounted `shared_ptr<Buffer>` keeps the one physical
  packed copy alive independently of the `PackedLinear` object, so the linear may
  then be moved into a `unique_ptr<Linear>` (what M6-T07's `OptimizedModel` does)
  without dangling. Tested: `source().data() == lm_head.packed_weight().data()`,
  and the lookup still matches after the linear is moved.
- **The id-range pre-scan lives in the module, not the kernel.** The kernel
  entries are raw-pointer, precondition-only (like `PackedGemm`);
  `OptimizedEmbedding::forward` front-loads the `y` shape/dtype checks and the
  `[0, V)` id scan (naming the offending index+value exactly as
  `cpu::embedding_lookup`) so a bad id never reaches the gather (an
  out-of-bounds read), and nothing inside the parallel region does anything but
  move+widen (§5, ADR-003).

---

## 8. Attention (M6-T04 prefill, M6-T05 decode)

Both attention kernels honor the exact `cpu::attention` op contract
(`src/cpu/ops.h`) so the tests are 1:1 against the reference: `q [T, H, d]`,
`k`/`v [Hkv, L, d]` head-major (`L = P + T`), `scale` applied to the completed
dot (HF order), `out [T, H, d]`, GQA by `h/g` KV-head indexing with **no
materialized repeat**, causal mask (new query `t` at absolute position `P+t`
attends keys `[0, P+t]`). The `Attention`-level orchestration (project QKV →
RoPE Q/K → `cache.append` → `cache.view` → attention → o_proj) is unchanged from
model-execution.md §4.2; only the attention *math* is replaced.

- **Prefill (M6-T04): blocked, flash-style online softmax. *Landed
  2026-08-18.*** Tile queries into `kQb`-row blocks and keys into `kKb`-column
  blocks (`kAttnQb = 32`, `kAttnKb = 64`; named constants in
  `internal/attention_common.h`, an M12-T02 tuning seam). For each `(head,
  query-block)` (the parallel unit, grain 1), stream key blocks left-to-right
  maintaining a running row-max `m`, running denominator `l`, and running output
  accumulator, rescaling by `exp(m_old − m_new)` when the max advances — the
  standard CPU flash-attention recurrence, fp32 throughout. No `[T, L]` score
  matrix is materialized; the optimized path holds only one query's
  `kAttnKb`-wide score row at a time (row-at-a-time, so `kAttnQb` controls only
  how many queries reuse each streamed K/V block — the accumulator is `out`
  itself, §6.1, so there is no `[kQb, d]` scratch). **Causal masking is
  expressed as a per-row `n_valid`** rather than writing `−inf`: key blocks
  wholly past the query's boundary are never visited (the causal skip), and the
  diagonal block iterates only its valid keys — the masked keys contribute
  exactly 0, reproducing the reference's `−inf → 0` softmax contract without an
  explicit mask value. Supports prefill continuing from a non-empty cache
  (`P > 0`) — `limit = P + t`, the same kernel (model-execution.md §6.3).
  **Implementation shape (mirrors the GEMM idiom):** the online-softmax control
  flow is written once as `internal::PrefillUnitsImpl<Ops>`
  (`attention_common.h`); the scalar/NEON/AVX2 TUs supply only four arithmetic
  primitives (dot+score, exp+sum, scale, axpy), so the blind-written AVX2 TU is
  a pure ISA swap of those primitives. Observed vs `cpu::attention` (NEON): max
  ~1.3e-6 at realistic magnitude across T∈{1,17,512,2048}, P∈{0,5}, GQA
  {(4,4),(4,2),(8,1)}, d∈{18,24,64,128}, and the block-boundary sweep; ~1.4e-5
  under a large-logit stress (std≈4) — all inside the §10 tolerance.
- **Decode (M6-T05): one query per sequence over the full cache. *Landed
  2026-08-18.*** `q [H, d]` (a `T == 1` slice) attends the cached `k`/`v [Hkv, L,
  d]`; threaded across kv heads (`kAttnHeadGrain = 1`), the `g = H/Hkv` query
  heads of each group processed together so that kv head's K/V stream from memory
  once. **Implementation shape:** the decode recurrence is written once as
  `internal::DecodeUnitsImpl<Ops>` (a second control-flow template alongside
  `PrefillUnitsImpl`) driving the **same four `Ops` primitives** as prefill —
  zero new ISA arithmetic, so the blind AVX2 TU is a one-line
  `DecodeUnitsImpl<Avx2Ops>` instantiation. The recurrence is **key-block-outer,
  query-inner** (`internal::DecodeGroupSlice`, split out to stay under the tidy
  cognitive-complexity threshold, the gemm.cpp idiom): for each `kAttnKb` key
  block, every query head of the group runs its dot/exp/rescale/axpy against that
  block while it is cache-resident, so K/V go through DRAM once per kv head
  regardless of `g`. Per-query online-softmax state (`run_max`/`run_den`) lives in
  fixed `kAttnDecodeGroupChunk = 8` stack arrays; a group larger than the chunk
  (no real model — g ≤ 7 for Llama-3.2-1B / Qwen2.5) is sliced, re-streaming K/V
  once per slice. No per-worker scratch (the accumulator is `out` itself, §6.1)
  and no allocation. **Bit-identical to `PrefillAttentionF32` called with
  `T = 1`** on the same q/k/v (the M6-T05 acceptance cross-check, asserted
  *bitwise* not by tolerance): for each fixed query head the block sequence, the
  per-block `n_valid`, the first-block `alpha = exp(−inf) = 0`, and the four Ops
  calls are the identical arithmetic in the identical order — interleaving the
  blocks across the group's other query heads never touches this head's
  accumulator. This bitwise equivalence is what M8-T05's paged decode kernel
  ("matches the M6-T05 contiguous decode kernel results exactly") will be
  validated against. **This bit-identity constrains the paged block size**
  (`docs/design/paged-kv-cache.md` §4): a `kAttnKb = 64`-key online-softmax unit
  must be an integer number of whole physical blocks, so the paged kernel gathers
  a unit's blocks into the same 64-wide score row and runs the identical
  max/expsum/scale/axpy — hence `bs | kAttnKb`, i.e. `bs ∈ {8, 16, 32, 64}`
  (default 16), enforced at pool construction. The paged kernel reuses the same
  four `Ops` primitives with block-table indirection replacing the `k_head +
  s·d` contiguous walk. **Parallel width is `Hkv`** (not `H·qblocks` as in prefill):
  decode threads over kv heads only, so a model with `Hkv < cores` leaves cores
  idle on the decode step — the concrete motivator for M12-T03's flash-decoding
  cache split (§11, §12). Observed vs `cpu::attention` (NEON): max **8.6e-6**
  across cache lengths {1, 63, 64, 65, 127, 128, 129, 2048}, GQA
  {(4,4),(4,2),(8,1)} + a `g = 12 > chunk` case, d∈{18,24,64,128}, and the
  large-logit stress (std 2) — inside the §10 tolerance; decode-vs-prefill is 0
  (bitwise).

The online-softmax rescaling changes the *arithmetic order* vs the reference's
materialize-then-`cpu::softmax`, so attention is **Class T** vs the oracle (a
stated tolerance, §10) — but still **bit-identical across thread counts** (each
output row's recurrence runs entirely within one thread). Vectorized `exp` (§10)
is the numerical-algorithm change here, tolerance-bounded per §10.

**Reads through `cache.view`.** M6 reads the accumulated K/V through the abstract
`KvCache::view` (the `SimpleKvCache` gather, model-execution.md §6.2). The gather
is `O(Hkv·L·d)` per layer per forward — for Llama-3.2-1B decode at `L=4096`:
`8·4096·64·2` (K+V) `·4 B ≈ 33 MB` copied per layer, ~530 MB across 16 layers per
decode step, vs the ~2.5 GB of weights streamed — roughly a **12% memory-traffic
overhead** on decode. Accepted for M6 (the interface is the seam that keeps the
kernel simple and paged-ready); **M8-T05 removes it** by reading through the block
table directly (§11). The doc records the number so M6-T08's baseline is
interpreted with it in view, and so M8's improvement is measured against a known
starting point.

> **M8-T05 landed** (`kernels::PagedDecodeAttentionF32`, paged-kv-cache.md
> §9.2): decode reads K/V directly through the block table, so the paged decode
> path pays no `view()` gather. The kernel is **bit-identical** to
> `DecodeAttentionF32` on the same logical K/V (asserted bitwise). The consumer
> swap — `OptimizedModel::forward` trying `paged_view` first and falling back to
> `view()` + the contiguous kernel — is M8-T07, not the kernel ticket; until
> then nothing above the `KvCache` interface changes.

---

## 9. Backend selection & test wiring

- **`BuildOptions.backend`** already exists (`src/model/registry.h`); the
  Llama/Qwen2 family builder dispatches on it: `kReference` →
  `ReferenceModel::Create` (M5), `kOptimized` → `OptimizedModel::Create` (M6-T07,
  replacing the current `Unimplemented`). `engine/backend.h`'s
  `Backend`/`ParseBackend`/`BackendName` need no change — `kOptimized` is already
  named.
- **Both backends buildable in one test:** `BuildModel` takes the `Backend`, so
  M6-T07's acceptance (`OptimizedModel` token-for-token vs `ReferenceModel`) is a
  single test that builds the same `LoadedModel` twice. (Since `BuildModel`
  consumes the `LoadedModel`, the test loads twice or the fixture helper re-maps —
  a test-harness detail, not an interface change.)
- **`ENGINE_FORCE_ISA` / `ENGINE_NUM_THREADS`** unchanged (cpu-backend.md
  §4.3/§3.1). The optimized kernels are dispatched, so their test binaries
  **register with `SCALAR_PASS`** (cpu-backend.md §8.1) — every optimized-kernel
  suite runs twice (host-best ISA + forced scalar), and the forced-scalar pass on
  both hosts is what proves the scalar variant (cpu-backend.md §6.2). The
  `OptimizedModel` end-to-end suite likewise opts into `SCALAR_PASS`. This is the
  first milestone where `SCALAR_PASS` covers a full model forward, not just leaf
  kernels.
- **`model` label** (like M5) for `ctest -L model`; kernel suites keep the
  `kernels` label pattern.

**As-built (M6-T07).** The `BuildReferenceFamily` builder was renamed
`BuildFamily` and now dispatches on `options.backend`: `kOptimized` →
`OptimizedModel::Create`, else `ReferenceModel::Create` (the M5 `Unimplemented`
guard is gone; `registry.h`/its docs updated, and the stale
`registry_test` "kOptimized is Unimplemented" case is now
"kOptimized builds through the registry"). `optimized_model_test` (the
`SCALAR_PASS` model suite — the first full-model forward under forced-scalar)
builds *both* backends on the same fixture — loading it once per backend, since
`BuildModel` consumes the `LoadedModel` — and asserts optimized == reference. The
`OptimizedModel::forward` validation block is copied from `ReferenceModel::forward`
verbatim (same order, same messages with the `OptimizedModel::` prefix), so the
error-path tests match 1:1. `OptimizedModel` builds one shared `Rope` for the
whole model, not one table per layer as the reference does — the cos/sin tables
are position-only, so a single copy suffices and a real model avoids hundreds of
MB of duplicated tables; the vectorized `kernels::RopeApplyF32` reads that shared
table's `cos()`/`sin()` pointers directly rather than going through the
reference's `Rope::apply` (which calls `cpu::rope_apply`).

---

## 10. Kernel-validation methodology & tolerance table

The oracle chain (cpu-backend.md §6.1), now at full strength — M6 is the first
milestone whose kernels are validated against the `cpu` reference at scale:

```
HuggingFace fixtures ──tol(Class T)──▶ src/cpu/ reference ──per-kernel──▶ src/kernels/ optimized
        (M4/M5)                            (M5, the oracle)                  (scalar/NEON/AVX2, M6)
```

Every optimized kernel is tested against the corresponding `cpu::` op (never
re-derived against HF directly — the reference is the proximate oracle), across
the model shapes and the size/alignment sweep (cpu-backend.md §9), on the host's
best ISA **and** the forced-scalar pass (§9). Every numerical test states its
tolerance and records the observed max-abs-diff, setting the threshold above it
with margin (the M5 practice, model-execution.md §3.3 — never the reverse).

**The acceptance question — where is bitwise vs tolerance equality expected, and
why:**

| Kernel | Oracle | Across **thread counts** | Across **ISAs** (scalar/NEON/AVX2) | vs the **oracle** |
|---|---|---|---|---|
| GEMM / GEMV (`PackedLinear`) | `cpu::gemm` | **bit-identical** (one fp32 accumulator per output, ascending k, never split) | **tolerance** (FMA contraction + block traversal differ per ISA) | tolerance (Class T) |
| RMSNorm | `cpu::rmsnorm` | **bit-identical** (per-row single accumulator) | tolerance (vector `rsqrt` refinement / horizontal sum order) | tolerance |
| SiLU-and-mul | `cpu::silu_mul` | bit-identical (pure elementwise) | tolerance (vector `exp` polynomial ≠ `std::expf`) | tolerance |
| residual add | `cpu::add` (`AddF32`) | **bit-identical** | **bit-identical** (Class E — one rounding per element) | **bit-identical** |
| softmax | `cpu::softmax` | bit-identical (per-row) | tolerance (vector `exp`, horizontal max/sum) | tolerance |
| RoPE apply | `cpu::rope_apply` | bit-identical (per-element map) | tolerance (FMA in the rotate; the tables are shared fp32) | tolerance |
| prefill / decode attention | `cpu::attention` | **bit-identical** (each output row's recurrence within one thread) | tolerance (online-softmax rescale order + vector `exp`) | tolerance (Class T) |
| embedding lookup | `cpu::embedding_lookup` | **bit-identical** | **bit-identical** (gather + exact `half.h`-matching widen — no accumulation) | **bit-identical** |

**Reading the table:**

- **Bit-identical across thread counts is a hard guarantee for the whole
  backend** (§5's determinism rule) — asserted in tests at counts {1, 2, 8} over
  the same inputs, exactly as `parallel_reduce` is (cpu-backend.md §9). A kernel
  that is not bit-identical across thread counts is a bug, not a tolerance.
- **Bit-identical across ISAs holds only for the two pure-map ops** (residual add,
  Class E; embedding gather-and-widen) — no cross-lane reduction, one rounding per
  element, so IEEE-754 makes same-input-same-rounding exact (cpu-backend.md §6.3
  Class E). The `half.h`-matching widen is what makes the embedding gather
  bit-exact.
- **Everything with a multiply-accumulate is Class T across ISAs** — GEMM,
  norms, softmax, RoPE, attention. Three legitimate sources of divergence: FMA
  contraction (`a·b+c` fused on NEON/AVX2, and even in scalar C++ it depends on
  `-ffp-contract`, which differs between clang `on` and GCC `fast`), horizontal
  vector reductions (lane-fold order ≠ scalar left-to-right), and vector
  transcendentals (§below). These are *rounding* differences, not correctness
  differences, so the contract is a stated absolute+relative tolerance, observed
  and recorded per shape.

**Tolerance recipes (stated, not "within tolerance"):**

- **GEMM/GEMV:** per element, `|opt − ref| ≤ atol + rtol·|ref|` with the atol
  scaled by `K` (accumulation length) — a longer dot admits more rounding.
  Starting point `rtol = 1e-4, atol = 1e-5·K^{1/2}`; the test records the observed
  max across every model shape (qkv/o/gate/up/down/lm_head, `K∈{1, 64, …, 8192}`,
  skinny/wide) and sets the threshold above it. Never loosened by later tickets
  (M12-T02's tuning "may not loosen tolerances", roadmap M12-T02).
- **RMSNorm:** exact `sqrt`+division required (**not** a raw `rsqrte` estimate —
  the Newton-refined `rsqrt` or a true `1/sqrtf` must land within ~1 ulp of the
  reference's `1/sqrtf`); tolerance `rtol = 1e-5`. The one place the doc forbids a
  fast-but-loose intrinsic in M6 (a raw NEON `vrsqrte` is ~2^-8 accurate — far
  outside tolerance; M12 may revisit with an error budget). *Landed (M6-T03): the
  shipped variants use a scalar `1.0f/sqrtf`; observed max-abs-diff vs the oracle
  2.4e-6 at E=4096 (`rtol 1e-5, atol 1e-6`).*
- **Vector `exp` (softmax, SiLU):** the one *new numerical algorithm* M6
  introduces — a polynomial `expf` approximation replacing `std::expf`,
  **shared verbatim by all three ISAs** (scalar spec in `internal/exp_common.h`,
  NEON/AVX2 lane helpers mirroring it), so the only cross-ISA difference is FMA
  contraction. It gets its **own dedicated sweep test** (`vector_exp_test`) vs a
  correctly-rounded reference (`std::exp` in double → float, stricter than
  `std::expf`), with a stated **ulp bound (≤ 2 ulp)**, independent of the kernels
  that use it. *Landed (M6-T03): the Cephes/`avx_mathfun` lineage — two-part
  (Cody-Waite) `ln2` range reduction `x = n·ln2 + r`, a degree-5 minimax for
  `eʳ`, `2ⁿ` by exponent-field write. Domain contract: `[kExpLo, kExpHi] =
  [-87.3365, 88.3762]` → ≤ 2 ulp (**observed max 1 ulp**, scalar and NEON);
  `x < kExpLo` (incl. `-inf`) → exactly `+0.0` (`n` stays ≥ -126 so no denormal
  trick breaks, and this makes softmax's `-inf → 0` mask contract exact — the
  flushed band's true value is ≤ FLT_MIN); `x > kExpHi` → finite saturation
  (immaterial: softmax inputs are ≤ 0 post-max-subtraction, SiLU's `exp(-v)`
  tail underflows in `1/(1+e)`).* Softmax/SiLU tolerances vs the reference are
  derived from that bound.

  **The scalar variants embed the polynomial too** (not `std::exp`): keeping one
  exp algorithm across all ISAs means the forced-scalar pass actually exercises
  the shipped numerical code, and the only scalar-vs-vector difference is
  FMA/lane order. The (small) cost is that scalar softmax/SiLU are Class T vs the
  oracle rather than near-exact — well inside the stated tolerances (observed
  below).
- **Attention:** `rtol = 1e-4, atol = 1e-5` on the context output vs
  `cpu::attention` (the reference's observed op-level agreement with HF is ≤4e-7,
  model-execution.md §13 T05, so the reference is effectively exact and the
  optimized-vs-reference band is the online-softmax + vector-exp rounding);
  cache lengths `{1, 63, 64, 65, 2048}` and GQA configs (M6-T05), sequence lengths
  `{1, 17, 512, 2048}` (M6-T04). *Landed (M6-T04, NEON): observed max
  **1.3e-6** across the acceptance sweep (T∈{1,17,512,2048}, P∈{0,5}, the
  decode-shaped T=1/P=2047, GQA {(4,4),(4,2),(8,1)}×d∈{18,24,64,128}, and the
  `kAttnQb`/`kAttnKb` block-boundary straddles), rising to ~1.4e-5 only under an
  out-of-range large-logit stress (q/k std 2 ⇒ logit std ≈4). The stated
  threshold holds; a logit std in the tens (no trained model's regime) is where
  the online rescale's incremental rounding exceeds it — recorded, not
  papered over.* **Decode (M6-T05, NEON): observed max 8.6e-6** across cache
  lengths {1, 63, 64, 65, 127, 128, 129, 2048}, GQA {(4,4),(4,2),(8,1)} + a
  `g = 12` (> `kAttnDecodeGroupChunk`) slicing case, d∈{18,24,64,128}, and the
  large-logit stress (std 2, observed 8.6e-6 at L=2048) — all inside the stated
  threshold. The decode kernel is additionally asserted **bit-identical to
  `PrefillAttentionF32(T=1)`** (max-abs-diff 0), the M6-T05 acceptance
  cross-check and the M8-T05 paged-decode oracle.
- **End-to-end:** `OptimizedModel` logits vs `ReferenceModel` logits within a
  stated tolerance (observed recorded); greedy generation **token-for-token
  identical** on the tiny fixtures. The M5-T09/T10 prompt selection (min top-2
  logit gap > 1e-2, model-execution.md §10) is what makes token-for-token robust
  for the optimized backend too — the reference-vs-optimized logit gap is orders
  of magnitude below that separation, so the argmax never flips. *Landed
  (M6-T07, NEON dev machine; band `rtol 2e-4, atol 2e-4`): optimized vs the
  reference **2.4e-7** (kAll) / **1.8e-7** (kLast) on both tiny-llama and
  tiny-qwen2; optimized vs the HF `activations.safetensors` logits golden
  **3.7e-6** (llama) / **3.9e-6** (qwen) — indistinguishable from the reference's
  own HF agreement, so the optimized path is as accurate against HF as the oracle;
  greedy generation token-for-token identical to both the reference backend and
  the `generate.json` goldens on every committed prompt of both fixtures. The KV
  invariant (full prefill vs token-by-token, chunked prefill) is **bit-exact**
  (max-abs-diff 0), matching the reference — GEMV≡GEMM-row (T02) and
  decode≡prefill(T=1) (T05) are bitwise, and each row's online-softmax recurrence
  is T-independent within one thread.*

**M6-T03 observed (NEON, dev machine; thresholds set above these):** softmax
max-abs-diff 6.0e-7 at n=4096 (`atol 1e-6, rtol 1e-4`); SiLU-and-mul 9.5e-7 at
n=32771 (`rtol 1e-5, atol 1e-6`); RoPE 2.4e-7 across head-dims {24, 64, 128} and
GQA head counts (`rtol 1e-5, atol 1e-6`); RMSNorm and vector-`exp` as noted
above. Every M6-T03 kernel is **bit-identical across thread counts** — asserted
by comparing the threaded public entry against a single serial variant call over
the whole problem (the M6-T02 idiom), plus, for softmax, an arbitrary manual
row-chunking. The **residual add is the pre-existing `kernels::AddF32`** (Class E,
bit-identical to `cpu::add`), so no new kernel — the ticket's residual-add box is
closed by a bit-equality parity test in `activation_kernel_test`.

**Recorded alternative (rejected for M6): cross-ISA bit-identity via
FMA-everywhere.** One could force every scalar multiply-add through `std::fma` and
demand the vector paths contract identically, making GEMM/attention bit-identical
across ISAs. Rejected: `std::fma` on an x86 target without FMA lowers to a slow
libm call (a correctness-neutral but large perf cost on exactly the fallback path
that must stay usable), the vector horizontal reductions still would not match
scalar order without further contortion, and the tolerance path is the honest
CPU-inference norm. Revisit only as an M12 tightening *with numbers*, if a
determinism requirement ever needs cross-ISA bit-identity (it does not today —
the merge gate is cross-*thread* bit-identity + cross-ISA tolerance, cpu-backend.md
§6.2).

**Benchmark obligations (no perf claim without a delta, CLAUDE.md):**

- M6-T02: packed GEMM ≥ 5× the M5 naive `cpu::gemm` at 4096³ on the dev machine
  (advisory), in BASELINES.md; a sanity comparison vs Accelerate/BLAS on the same
  shape recorded for context (not a target). Any BLAS comparison lives in
  `benchmarks/` behind an optional CMake `find_package` — **never linked into
  `src/`** (ADR-002; the engine ships no BLAS dependency). *Landed 2026-08-18 at
  8.76× (bf16) / 8.48× (f32) on the M2, ~112 GFLOP/s (BASELINES.md). The BLAS
  context number could not be captured on the dev machine: Homebrew LLVM 20 (the
  pinned toolchain) rejects the CommandLineTools SDK's Accelerate headers with
  `-Welaborated-enum-base` (the broken-CLT situation); the `-DENGINE_BENCH_BLAS`
  wiring is in place for a capable host and the number is context-only, so it is
  deferred rather than blocking.*
- M6-T04: prefill attention time vs the reference at 2k context, in BASELINES.md.
- M6-T08: `bench_generate` prefill tok/s and decode tok/s on the 1B model, ±5%
  run-to-run, with a hardware + thread-count fingerprint, and a same-model
  llama.cpp number alongside for context (parity is M12, not here). *Landed
  2026-08-18 (BASELINES.md M6-T08): Qwen2-0.5B, 8-thread NEON optimized —
  prefill ~133 tok/s (best), decode ~31 tok/s (median), decode run-to-run
  **±3.9% (PASS)**. Same-machine llama.cpp @ `6d05498` (CPU-only NEON): f16
  192/34 tok/s (prefill 8t / decode 4t) — ballpark parity; its ARM CPU bf16 path
  is ~20× slower than f16, so our bf16-first design compares against llama's
  f16. The M2 efficiency cores throttle memory-bound decode in both engines
  (llama decode is faster at 4 threads than 8) — an E-core-aware pool is an M12
  lever this measurement motivates, compounding the `Hkv`-only decode parallel
  width (§8, the M12-T03 motivator). Full quiescing discipline + thread/ISA
  sweep is M12-T01; the 8-thread baseline is the recorded number, 4t/2t are
  advisory.*

---

## 11. What later milestones change — and what they do not

- **M8 (paged cache):** the `KvCache` interface is unchanged (`append`/`view`/
  `truncate`/geometry); `PagedKvCache` replaces `SimpleKvCache` behind it. The
  optimized attention's read seam is the only thing that moves: M6 reads a
  contiguous gather from `cache.view`; M8-T05 decode reads through the block table
  directly, M8-T06 prefill gathers to a workspace before this doc's blocked
  kernel. So **the M6 attention kernels are written to consume a contiguous
  `[Hkv, L, d]` K/V slab** (from `view` today, from an M8 gather tomorrow) — the
  kernel signature does not change, only who fills the slab. This is why §8 quantifies
  the `view` gather cost now: M8 removes exactly that.
- **M9 (batching):** `ForwardRequest` grows `cu_seqlens`/slot-mapping fields
  (model-execution.md §5.4); the optimized forward loops over sequences *above*
  the kernels (the kernels already take per-sequence `q`/`k`/`v`), and varlen
  prefill (M9-T06) tiles across the ragged batch. The packed weights, workspace,
  and micro-kernels are unchanged.
- **M12 (tuning/fusions):** tile/grain constants tuned with a sweep (M12-T02);
  flash-decoding cache split via `parallel_reduce` (M12-T03, preserving §5's
  cross-thread bit-identity); RoPE+KV-write / residual+norm / SwiGLU fusions
  (M12-T04) as local edits to the `OptimizedModel` graph (the reason §2.2 chose a
  separate graph); allocation-free decode already delivered by §6.3, so M12-T05 is
  audit + `ForwardRequest` output-buffer field; mmap-release hygiene
  (`MADV_DONTNEED` after packing, since the checkpoint pages are dead once `Wp` is
  built — §3.1).
- **M13 (quantized):** `QuantizedLinear` is a third `Linear` implementation
  alongside `PackedLinear`; its **dequant-to-tiles path targets this doc's packed
  `[P, K, NR]` layout** (dequantize a group into the packed fp16/bf16 tile, then
  feed the M6 GEMM), and its fused dequant-GEMV mirrors the decode path. The
  packed layout is chosen (fixed `NR`, K-major panels) so that INT8/INT4 dequant
  writes into it naturally.

---

## 12. Deferred (known, intentionally not designed here)

- **Tile/grain autotuning, software prefetch, native fp16/bf16 FMA** — M12-T02
  (the last would amend the fp32-accumulation policy, cpu-backend.md §10).
- **Flash-decoding (cache-split decode attention)** — M12-T03, via
  `parallel_reduce`.
- **Kernel fusions (RoPE+KV-write, residual+norm, SwiGLU-combine)** — M12-T04; the
  unfused reference path is retained for testing (model-execution.md §14).
- **Allocation-free decode as an enforced invariant** (counting-allocator hook) —
  M12-T05; §6.3 delivers the behavior, M12 adds the assertion + the optional
  `ForwardRequest` output buffer.
- **Paged K/V read path, scatter-write kernel** — M8; §8/§11 record the seam.
- **Ragged/batched forward, `cu_seqlens`** — M9.
- **Quantized weight tiles (INT8/INT4 dequant into the packed layout)** — M13.
- **A macOS-arm64 CI job** — still deferred (cpu-backend.md §8.3); NEON coverage
  stays the per-ticket dev-machine workflow, and M6's optimized kernels rely on it
  more than any prior milestone, so the per-ticket "full ctest incl. forced-scalar
  on the dev machine" discipline (CLAUDE.md) is load-bearing for M6.
