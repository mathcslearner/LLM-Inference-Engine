# Paged KV cache & block manager

**Milestone:** M8 (design doc: M8-T01; implementation: M8-T02 … M8-T08)
**Governs:** `src/kvcache/`'s paged storage (the block pool, per-sequence block
table, and the `PagedKvCache` implementation of the M5 `KvCache` interface), the
new paged KV-write (scatter) and paged decode-attention kernels in
`src/kernels/`, the `--kv-cache-memory` / `--kv-block-size` capacity wiring in
the engine, and the block-pool statistics surface — plus the memory
architecture every later milestone (M9 continuous batching, M11 prefix caching,
M12 flash-decoding/fusions/chunked prefill, M13 INT8 KV cache, M15 speculative
rollback) builds on.
**Cites:** ADR-002 (module boundaries — this doc adds the downward `kvcache →
kernels` edge, §2), ADR-003 (error handling — recoverable `Status` vs. `CHECK`),
ADR-004 (CPU-first pivot), `docs/design/model-execution.md` (the `KvCache`
interface v0 §6, the KV correctness invariant §6.2, the prefill-continuation
contract §6.3, and §6.4's "what M8 changes" account this doc discharges),
`docs/design/optimized-cpu-execution.md` (the M6 attention kernels §8/§10 whose
block structure the paged kernels reuse — the source of the block-size
constraint in §4), `docs/design/cpu-backend.md` (threading, SIMD dispatch,
dtype/accumulation policy, numerics classes §6.3), `docs/design/tensor.md`
(Tensor/Buffer, aligned allocation), and the `memory::CachingAllocator` (M2-T06)
that backs the pool.

This is the working contract for the paged KV cache. Implementation tickets must
conform to it; if implementation reveals a design flaw, this doc is updated in
the same change with a note on what changed and why
(`docs/design/README.md`). M8-T02+ tickets amend in place; M11 (prefix caching)
gets its own doc (`docs/design/prefix-caching.md`) that extends the refcount and
index seams reserved here (§6, §11).

---

## 1. Scope & non-goals

M5 gave every sequence a private contiguous cache (`SimpleKvCache`, two
`[L, Hkv, capacity, d]` fp32 tensors sized at construction). That is correct but
un-shareable: capacity is reserved per sequence whether used or not, so batching
`N` sequences reserves `N × max_len` even when most are short, and two sequences
that share a prompt prefix cannot share storage. M8 replaces the contiguous
storage with a **paged** cache: a single pool of fixed-size token **blocks**,
allocated on demand, indexed per sequence by a **block table** (logical
position → physical block). Attention reads through that indirection.

The redesign is done **once, with reference counting from day one**, so M11
(prefix caching) is an *extension* — adopt cached blocks into a new sequence's
table and bump their refcount — not a rewrite. The paged cache is the memory
architecture that makes continuous batching (M9) and prefix caching (M11)
possible.

**In scope (this doc):**

- the physical block layout (§3) and the block-size choice (§4),
- the pool capacity calculation from a `--kv-cache-memory` budget (§5),
- the block pool with refcounting and stats (§6),
- the per-sequence block table and slot mapping (§7),
- the `PagedKvCache` implementation of the M5 `KvCache` interface, including the
  one additive accessor paging needs (§8),
- the paged KV-scatter and paged-decode kernels, and the gather-based paged
  prefill path (§9),
- the engine integration and exhaustion posture (§10),
- how M9/M11/M12/M13/M15 attach to these seams (§11),
- the per-ticket testing strategy (§12).

**Non-goals (of this doc, not the project) — each named with its milestone:**

- **No continuous batching / scheduler.** The pool is a shared resource M9's
  scheduler admits against, but the scheduling policy, the batched block-table
  tensor, and preemption *mechanics* are M9 (`docs/design/scheduler-runtime.md`).
  This doc reserves the accessors M9 needs (`free_blocks()`, `blocks_needed(n)`,
  `release_all`) and sketches the batched kernel shape (§9.4, §11).
- **No prefix-cache index, hashing, or eviction policy.** The refcount states
  and the "shared blocks are immutable" invariant are designed here (§6.3) so
  M11 attaches without a storage change; the content-hash index, LRU eviction,
  and the copy-on-write edge are M11 (`docs/design/prefix-caching.md`).
- **No fully paged prefill kernel.** M8-T06 prefill *gathers* the paged K/V to a
  contiguous workspace and reuses the M6 blocked prefill kernel unchanged
  (simple, correct). A prefill kernel that walks the block table directly is
  M12's tuning pass.
- **No flash-decoding cache split.** The paged decode kernel (M8-T05) threads
  across kv heads exactly like M6-T05; splitting one long context across threads
  with a deterministic reduction is M12-T03.
- **No kernel fusions.** RoPE + KV-write stays two observable steps; the fusion
  (and the raw-slot write accessor it needs) is M12-T04 (§11).
- **No INT8 KV cache.** The pool is fp32 (`geometry().dtype == kFloat32`). The
  layout reserves where per-block-per-head scales live so M13-T07 is additive
  (§3.4, §11).

---

## 2. Module layout & layering

New files, all under existing modules:

| File | Module | Ticket | Responsibility |
|---|---|---|---|
| `src/kvcache/block_pool.{h,cpp}` | kvcache | M8-T02 | Block storage + free-list allocate/free, per-block refcounts, stats |
| `src/kvcache/block_table.{h,cpp}` | kvcache | M8-T03 | Per-sequence logical→physical map, append/slot-mapping/truncate/free |
| `src/kvcache/paged_cache.{h,cpp}` | kvcache | M8-T04/T07 | `PagedKvCache : KvCache` — the paged implementation of the M5 interface |
| `src/kernels/kv_scatter.{h,cpp}` | kernels | M8-T04 | `KvScatterF32` — write new K/V into paged storage given a slot mapping |
| `src/kernels/paged_attention.{h,cpp}` (+ per-ISA TUs) | kernels | M8-T05 | `PagedDecodeAttentionF32` — decode reading K/V through a block table |
| `src/kernels/kv_gather.{h,cpp}` | kernels | M8-T06 | `KvGatherF32` — read paged K/V into a contiguous head-major slab (the scatter's mirror) |
| `src/kvcache/paged_gather.{h,cpp}` | kvcache | M8-T06 | `GatherLayerKV` — one layer's paged K/V → contiguous `KvView` for M6 prefill (calls `KvGatherF32`) |

**Layering.** `kvcache` is a layer-2 domain module (ADR-002 §layers). Today it
links only `tensor`/`memory`/`core`. M8-T04 adds a **downward** edge
**`kvcache → kernels`** (both the scatter and paged-decode kernels live in the
layer-1 `kernels` module; `PagedKvCache` and `paged_gather` call them). This is a
lower-layer edge of exactly the kind ADR-002 already permits without amendment —
identical in status to `model → kernels` (optimized-cpu-execution.md §2.2),
`sampling → kernels` (M7-T06), and `cpu`/`kvcache → tensor`. No ADR amendment is
required; the CMake `target_link_libraries(engine_kvcache … PRIVATE
engine::kernels)` and the stale "never links `kernels`" comment in `kv_cache.h`
are updated when T04 lands, and this doc records the edge so the change is not a
silent divergence.

The kernels stay **layout-agnostic**: they never see `BlockPool` or
`BlockTable`. They take raw fp32 base pointers, an `int32_t*` block-id array, and
explicit strides (§9). The physical layout is `kvcache`'s private decision, so
M12/M13 can change it (fusion accessors, INT8 tiles) without touching a kernel
signature beyond adding parameters.

The `model → kvcache` edge (Amendment 5) is unchanged: `Attention` /
`OptimizedModel` hold a `KvCache&` and are oblivious to whether it is
`SimpleKvCache` or `PagedKvCache` — except for the single optional fast-path
accessor in §8.3, which is a virtual on the shared interface, not a
`dynamic_cast`.

---

## 3. Physical layout

### 3.1 The choice: per-layer K/V slabs of head-major block tiles

The pool holds `num_blocks` physical blocks. A block stores `block_size` (`bs`,
§4) contiguous token slots. For **each layer** the pool keeps **two slabs** — one
for K, one for V — each shaped

```
[num_blocks, Hkv, bs, d]   fp32, contiguous, row-major
```

So the pool owns `2 · num_layers` tensors (K and V per layer). A physical block
id `b` names the same slot region in every layer's slabs; a sequence's block
table (§7) is **layer-independent** — one table drives all layers, and layer `ℓ`
reads block `b` from *its* slab pair. The innermost tile

```
tile(ℓ, K/V, b, h) = slab[ℓ][K/V] + (b·Hkv + h)·bs·d      // [bs, d] contiguous
```

is a contiguous `[bs, d]` region: `bs` token rows of one kv head, each `d`
floats. Address of layer `ℓ`, kv head `h`, block `b`, in-block token `p`,
element `e`:

```
K: slab_k[ℓ] + ((b·Hkv + h)·bs + p)·d + e
V: slab_v[ℓ] + ((b·Hkv + h)·bs + p)·d + e
```

with the strides the kernels receive as parameters:

```
block_stride = Hkv·bs·d      // one block id advances this many floats
head_stride  = bs·d          // one kv head within a block
row_stride   = d             // one token slot within a (block, head) tile
```

### 3.2 Why this layout

- **Matches the M6 kernels' access pattern.** M6-T05 decode and M6-T04 prefill
  stream *one kv head's keys contiguously* (`k + hk·L·d`, key `s` at `+s·d`) —
  head-major, matching HF's cache after `transpose(1,2)` (model-execution.md
  §6.2). Our `[bs, d]` tile is exactly a run of that stream; a full head's
  history is the concatenation of its per-block tiles in block-table order. The
  paged decode kernel (§9.2) walks blocks and, within a block, reads the same
  contiguous `[n_valid, d]` the contiguous kernel reads — this is what makes
  M8-T05's *bitwise* match to M6-T05 achievable (§4, §9.2).
- **Separate K and V slabs** match every kernel's `(k, v)` pointer-pair
  signature (M6 `PrefillAttentionF32(q, k, v, …)` / `DecodeAttentionF32`), and
  the scatter kernel writes K and V from separately-projected `[T, Hkv, d]`
  tensors. Interleaving K and V in one slab would force a stride the kernels do
  not take and buys nothing on CPU.
- **The roadmap's `[num_blocks, 2, layer…]` sketch, resolved.** The literal `2`
  is the K/V axis; we realize it as *two slabs* rather than an inner `2` dim
  because the K and V tiles are consumed by different kernel arguments and are
  never addressed as a unit — folding them into one tensor would only add a
  stride. **Per-layer slabs** (rather than a single `[num_blocks, 2, L, Hkv, bs,
  d]` tensor) keep one layer's blocks in a dense, separately-allocatable address
  range: layer `ℓ`'s decode touches only `slab_k[ℓ]`/`slab_v[ℓ]`, so the working
  set per layer is `used_blocks · Hkv·bs·d` contiguous-per-block floats rather
  than being interleaved with 23 other layers' data at a `2·L·Hkv·bs·d` block
  stride (better prefetch and TLB behavior; also lets M9/M13 vary per-layer
  allocation later). This is a deviation from the literal roadmap sketch,
  recorded here per `docs/design/README.md`.

### 3.3 Worked example — concrete offsets

**tiny-llama** (`L=2, Hkv=2, d=16`, `bs=16`, fp32):

- tile bytes `= bs·d·4 = 16·16·4 = 1024 B = 1 KiB` per (block, head).
- one block, one layer, K = `Hkv·bs·d·4 = 2·1024 = 2 KiB`; K+V = 4 KiB.
- one block, all layers = `2 · L · Hkv·bs·d·4 = 2·2·2·16·16·4 = 8 KiB`.
- Layer 1, V, block `b=3`, kv head `h=1`, in-block token `p=5`, element `e=2`:
  `slab_v[1] + ((3·2 + 1)·16 + 5)·16 + 2 = slab_v[1] + (7·16+5)·16 + 2 =
  slab_v[1] + 117·16 + 2 = slab_v[1] + 1874` (floats), i.e. byte `7496` into
  layer 1's V slab.

**Qwen2-0.5B** (`L=24, Hkv=2, d=64`, `bs=16`, fp32 — the dev-machine real model):

- tile bytes `= 16·64·4 = 4096 B = 4 KiB` per (block, head).
- one block, one layer, K = `2·4 KiB = 8 KiB`; K+V = 16 KiB.
- one block, all layers = `2·24·2·16·64·4 = 393216 B = 384 KiB`.
- a 1 GiB KV budget ⇒ `⌊1073741824 / 393216⌋ = 2730` blocks ⇒ `2730·16 =
  43680` cached tokens across the pool (§5).

**Llama-3-8B class** (`L=32, Hkv=8, d=128`, `bs=16`): one block, all layers =
`2·32·8·16·128·4 = 4 MiB`; a 40 GiB budget ⇒ `10240` blocks ⇒ `163840` tokens.

### 3.4 Dtype policy and the INT8 seam

The pool is fp32: `geometry().dtype == kFloat32`, matching M5's v0 and the M6
kernels' fp32 accumulation policy. `CacheGeometry.dtype` already exists on the
interface, so M13-T07's `--kv-cache-dtype int8` is additive: the slabs become
INT8 `[num_blocks, Hkv, bs, d]`, and **per-block-per-head scales** live in a
parallel `[num_blocks, Hkv]` (or `[num_blocks, Hkv, 2]` for K and V) fp32 array
the pool allocates alongside the slabs. Dequant moves inside the attention
kernels (they gain a `dtype` + `scales` parameter). This doc only reserves the
shape; M13 designs the calibration. The scatter/decode signatures in §9 are
written so the INT8 variant adds parameters, not a new function family.

---

## 4. Block size

**Default `bs = 16` tokens.** Runtime-configurable via `--kv-block-size`;
constrained to a **power of two that divides `kAttnKb = 64`** — i.e.
`bs ∈ {8, 16, 32, 64}`. Rationale:

- **Fragmentation.** A sequence wastes at most `bs − 1` slots in its last
  (partial) block: `≤ 15` tokens at `bs=16`, negligible against typical prompt +
  generation lengths.
- **Tile size.** `bs·d·4` at `bs=16, d=64` is a 4 KiB tile — one page, a
  friendly unit for the head-major stream the M6 kernels want (§3.2).
- **Prefix-cache granularity (M11).** Prefix caching shares only *full* blocks
  (M11-T01). Smaller `bs` ⇒ finer sharing (a 20-token shared prefix shares one
  16-block, not zero 32-blocks) but more blocks, more hash-chain work, and more
  table entries. 16 is the vLLM-proven middle.
- **The divisibility constraint — load-bearing for M8-T05.** M6-T05's decode
  recurrence rescales the online softmax once per **`kAttnKb = 64`-key unit**
  (`DecodeGroupSlice` in `attention_common.h`: `DotScoreRow` over the unit into a
  fixed 64-wide `scores` row → running-max update → `ExpRowSum` → `ScaleRow` →
  per-key `AxpyRow`). To reproduce that arithmetic **bit-for-bit** over paged
  storage, one 64-key unit must be an integer number of whole physical blocks:
  `kAttnKb % bs == 0`. With `bs ∈ {8,16,32,64}` a unit spans `64/bs ∈ {8,4,2,1}`
  contiguous blocks; the paged kernel gathers those blocks' keys into the same
  64-wide `scores` row and runs the identical max/expsum/scale/axpy sequence (the
  running max is order-free; the row-sum and axpy run over the identical
  contiguous score row), so the fp32 accumulation order per output element is
  unchanged. This is why M8-T05 can promise "matches the M6-T05 kernel
  **exactly**" and not merely "within tolerance." A `bs` that did not divide
  `kAttnKb` would force a different reduction grouping and only a Class-T match;
  the constraint is enforced at pool construction (`InvalidArgument`) and
  restated in optimized-cpu-execution.md §8.

---

## 5. Capacity: `--kv-cache-memory` budget

### 5.1 The budget flag

`--kv-cache-memory` sets the **KV pool's byte budget** (K/V slabs + scale arrays
only — not weights, not workspace, not the pool's own bookkeeping vectors). Two
spellings:

- **Absolute:** a byte count with an optional unit — `2GiB`, `1500MiB`,
  `8000000000`.
- **Fraction:** a value in `(0, 1]` — `0.6` — meaning that fraction of *host
  RAM* is the ceiling for **weights-resident + workspace + KV**, from which the
  KV budget is derived (§5.3).

Default: `0.9` (fraction) when unset, matching vLLM's `gpu_memory_utilization`
analog for host RAM. The single-request `engine generate` path (M8-T07) also
accepts the pre-paging `--cache-capacity N` *tokens* spelling as a convenience,
translated to `num_blocks = ⌈N / bs⌉`; the memory-budget flag is the primary
knob and the one the M9 server uses.

**M8-T07 as built (CLI default divergence).** `engine generate` defaults to a
**token-sized pool** (`⌈(prompt + max_new_tokens) / bs⌉` blocks), *not* the
`0.9` fraction, when neither `--kv-cache-memory` nor `--cache-capacity` is
given. Rationale: the `0.9`-of-host-RAM default is for the M9 server, which
holds an unknown number of concurrent sequences and wants to claim the machine;
a single-request CLI generation needs only this request's worst case, and a
`0.9` default would zero-fill ~13 GB of slabs on the 16 GB dev box (and in the
ctest smoke runs) for a 5-token prompt. `--kv-cache-memory 0.9` still works and
is honored exactly as specified — it is just not the CLI default. The M9 server
carries the `0.9` default (§5.1 unchanged for it). The flag also gained a
sibling `--kv-cache {paged,simple}` (default `paged`) selecting the backend —
`simple` keeps the pre-paging `SimpleKvCache` as an escape hatch and the A/B
baseline for `bench_generate` — and `--kv-block-size N` (default 16).

### 5.2 The blocks-per-pool formula

```
bytes_per_block = 2 · num_layers · Hkv · bs · head_dim · itemsize
                    │                                      └ 4 (fp32); 1 (+scales) for INT8, M13
                    └ K and V

num_blocks      = ⌊ kv_budget_bytes / bytes_per_block ⌋
pool_tokens     = num_blocks · bs
```

`num_blocks < 1` after the floor ⇒ `Create` returns `ResourceExhausted` (the
model does not fit the budget). Worked numbers in §3.3.

### 5.3 Deriving the KV budget from a fraction

When `--kv-cache-memory` is a fraction `f`:

```
kv_budget_bytes = f · host_ram_bytes − weights_resident_bytes − workspace_bytes
```

- `host_ram_bytes`: `sysctl hw.memsize` on macOS, `sysconf(_SC_PHYS_PAGES) ·
  _SC_PAGE_SIZE` on Linux (a tiny `core`/util helper; documented per platform).
- `weights_resident_bytes`: reported by the loader/`OptimizedModel` — the packed
  weight bytes actually allocated (known after model build), **not** process RSS.
- `workspace_bytes`: the model's `Workspace::bytes()` high-water estimate for the
  configured max batch/sequence (the M6 workspace already reports this).

A non-positive result ⇒ `ResourceExhausted` naming the three terms, so the
operator sees *why* nothing is left for KV. Absolute budgets skip this
subtraction (the operator owns the arithmetic) but still `ResourceExhausted` if
`num_blocks < 1`.

---

## 6. Block pool (`block_pool.h`, M8-T02)

Pure bookkeeping over the pre-allocated slabs — fully unit-testable without a
model. One pool is shared by all sequences (M9); in the M8 single-request path
the engine still owns exactly one.

### 6.1 Storage and construction

`BlockPool::Create(CacheGeometry geom, int block_size, int64_t num_blocks,
memory::Allocator* allocator) → StatusOr<BlockPool>`:

- validates `geom` (positive dims, `dtype == kFloat32` until M13), `block_size`
  ∈ {8,16,32,64} (§4), `num_blocks ≥ 1`;
- allocates the `2·L` slabs (§3.1) from `allocator` (the engine passes the M2
  `CachingAllocator`; tests pass the plain `DefaultCpuAllocator`), 256-byte
  aligned (`CachingAllocator::kMaxAlignment`), **zero-filled once** at
  construction. Zero-fill matters for correctness parity: a partial last block's
  unused slots read as `0.0`, and — as in v0 (model-execution.md §6.2) — a
  masked/out-of-range key contributes `0.0·v == 0.0` exactly, so the KV
  invariant stays bit-exact. It also makes the pool's RSS resident and
  predictable up front rather than faulting in mid-generation.
- **The whole pool is allocated at construction; blocks are never individually
  allocated upstream.** "Backed by the M2 caching allocator" means the pool's
  *slabs* come from it (one big allocation per slab, likely a cache miss then a
  reused block on restart); the per-block free list is the KV block allocator and
  never calls upstream on the token hot path.

Not copyable/movable while `Buffer`s are live is the allocator's rule
(`caching_allocator.h`); the pool holds its slab `Tensor`s by value and is
move-constructed out of `Create` before any block is handed out.

### 6.2 Free list, allocate/free, refcounts

State: `refcount_[num_blocks]` (`int32`), a `free_list_` (a `vector<int32_t>`
used as a stack — LIFO reuses recently-freed, cache-warm blocks), and running
`used_/free_` counts. Thread-safe under one mutex (allocation happens only at
block-boundary crossings, never per token — off the hot path, matching the
`CachingAllocator` stance).

```cpp
struct BlockPoolStats { int64_t total, used, free; double utilization; };

class BlockPool {
 public:
  static core::StatusOr<BlockPool> Create(CacheGeometry, int block_size,
                                          int64_t num_blocks,
                                          memory::Allocator*);

  // Allocate one fresh block: pop free list, set refcount 1, return its id.
  // Empty free list → ResourceExhausted (M8-T08 posture; M11 eviction retries
  // before this fires).
  [[nodiscard]] core::StatusOr<int32_t> Allocate();

  // Bump refcount (M11 prefix sharing adopts an existing block). CHECK on a
  // free (refcount-0) block — sharing what nobody owns is a programmer error.
  void Share(int32_t block);

  // Drop one reference. refcount 1→0 returns the block to the free list.
  // Releasing a refcount-0 block is the "double-free" the acceptance names →
  // CHECK (programmer error, not recoverable).
  void Release(int32_t block);

  [[nodiscard]] int32_t refcount(int32_t block) const;   // tests/stats
  [[nodiscard]] BlockPoolStats stats() const;            // used/free/util
  [[nodiscard]] int64_t free_blocks() const;             // M9 admission
  // Blocks a sequence at `cur_tokens` needs to admit `add_tokens` more
  // (boundary-crossing count) — M9 scheduler admission, pure arithmetic.
  [[nodiscard]] int64_t blocks_needed(int64_t cur_tokens, int64_t add) const;

  // Kernel-facing slab pointers + strides for layer ℓ (used by PagedKvCache
  // and the gather; never by a kernel directly — kvcache passes them down).
  [[nodiscard]] float* k_slab(int layer);
  [[nodiscard]] float* v_slab(int layer);
  [[nodiscard]] int64_t block_stride() const;  // Hkv·bs·d
};
```

`stats()` accuracy through scripted alloc/free/share sequences, `ResourceExhausted`
on exhaustion (no crash), and the double-free `CHECK` are M8-T02's acceptance.

**M8-T02 implementation notes** (as built, `src/kvcache/block_pool.{h,cpp}` —
additive clarifications of the sketch above, no divergence):

- **Slabs are allocated directly from the `Allocator` at `kSlabAlignment = 256`
  (== `CachingAllocator::kMaxAlignment`) then `memset` to zero**, rather than
  through `tensor::ops::zeros`/`Tensor::empty` (which pin a 64-byte alignment).
  This honors the §6.1 256-byte guarantee for any allocator; the slabs are held
  as `tensor::Tensor` values (`Tensor::from_buffer` over the raw `Buffer`).
- **The pool is returned by value (`StatusOr<BlockPool>`) and holds its mutex
  behind a `std::unique_ptr<std::mutex>`** so it stays movable (`std::mutex` is
  not). The move-constructor `CHECK(used_ == 0)` enforces "move out of `Create`
  before any block is handed out" (§6.1) — the M8-T03 `BlockTable` holds a raw
  `BlockPool*`, so a post-handout move would dangle it. There is no destructor
  `CHECK`: nothing dangles into a dead pool except that non-owning pointer,
  whose lifetime rule (§10.1) already covers it.
- **The §5.2 capacity formula ships as two static pure helpers on `BlockPool`** —
  `BytesPerBlock(geom, block_size)` and `NumBlocksForBudget(geom, block_size,
  budget)` (overflow-guarded; `< 1` block → `ResourceExhausted`). Host-RAM
  detection and the fraction-of-RAM subtraction (§5.3) stay with M8-T07, their
  first consumer (§13).
- **`block_size` is validated to `{8,16,32,64}` at construction *and* in the
  capacity helpers** (the §4 `bs | kAttnKb=64` constraint), `InvalidArgument`
  with the rationale otherwise; a non-fp32 `geom.dtype` is `Unimplemented` (the
  M13 seam), matching `SimpleKvCache::Create`.
- **Kernel-facing accessors** expose all three §3.1 strides — `block_stride()`
  (`Hkv·bs·d`), `head_stride()` (`bs·d`), `row_stride()` (`d`) — plus
  `slab_bytes()`/`total_bytes()` for the M8-T07 stats log and
  `block_size()`/`num_blocks()`/`geometry()`.

### 6.3 Refcount lifecycle

```
                 Allocate()                Share()                Share()
   ┌────────┐  (pop free list)  ┌────────┐  (M11)    ┌────────┐  (M11)  ┌────────┐
   │  FREE  │ ────────────────► │ OWNED  │ ────────► │ SHARED │ ──────► │ SHARED │
   │ rc = 0 │                   │ rc = 1 │           │ rc = 2 │         │ rc = n │
   │ on the │ ◄──────────────── │        │ ◄──────── │        │ ◄────── │        │
   │free list│   Release()      └────────┘  Release() └────────┘ Release └────────┘
   └────────┘   (rc 1→0,                    (rc 2→1)            (rc n→n-1)
     ▲          push free list)                 │                   │
     │                                          ▼                   ▼
     │  Release() on rc=0  ══► CHECK       (any rc>0 block is immutable — §6.4)
     │  (double free, programmer error)
     │
     └─────  M11 extension (dashed, not built here):  ┌──────────────┐
             a released prefix block does not go       │   CACHED     │
             straight to FREE but to CACHED (rc 0,      │ rc = 0, in   │
             still in the content index, evictable);    │ prefix index │
             a later hit Share()s it back to OWNED,      └──────────────┘
             LRU eviction moves CACHED → FREE.           (M11-T03/T05)
```

M8 uses only FREE/OWNED/SHARED. M11 inserts CACHED between "refcount hit 0" and
"returned to the free list" — a pure extension of the `Release` path (the pool
gains a `on_release` hook or the index wraps `Release`), which is exactly why the
refcount lives in the pool from day one.

### 6.4 The immutability invariant (M11 rests on this; stated now)

**A block with refcount ≥ 1 is never rewritten in place.** Sequences only ever
*append*: a write targets a slot in the sequence's **current tail block**, and
that block always has **refcount 1** (exclusively owned) because:

1. a freshly `Allocate()`d tail block has refcount 1, and
2. prefix sharing (M11) shares only **full** blocks — a shared block is by
   definition complete and never the append target; the first append past a
   shared prefix `Allocate()`s a new exclusive tail block.

So the "write only your own refcount-1 tail" rule holds without a check on the
hot path, and a `SHARED` block's bytes are stable for every reader. This is the
immutability M11's correctness proof invokes ("why shared blocks are safe to
share"). `truncate` into a shared block (M15 rollback of a shared prefix — rare)
must copy-on-write: `Allocate` a private block, copy the surviving prefix,
release the shared one. M8's `truncate` only ever shrinks a sequence's own
exclusive tail (no CoW needed); the CoW edge is flagged for M11/M15 (§11), not
built here.

---

## 7. Block table (`block_table.h`, M8-T03)

Per-sequence logical→physical mapping. Owned by a `PagedKvCache` (one table
drives all layers, §3.1).

### 7.1 State and slot mapping

```cpp
class BlockTable {
  BlockPool* pool_;            // non-owning; must outlive the table
  std::vector<int32_t> blocks_;  // logical block i → physical block id
  int64_t num_tokens_ = 0;       // committed tokens in this sequence
};
```

Logical block `i` covers positions `[i·bs, (i+1)·bs)`. Token at logical position
`pos` lives in logical block `pos / bs`, in-block offset `pos % bs`, physical
block `blocks_[pos/bs]`. Its **flat slot** (the scatter kernel's addressing unit)
is:

```
slot(pos) = blocks_[pos / bs] · bs + (pos % bs)
```

a single `int64` naming "which physical block, which row within it." A batch of
`T` new tokens produces a `slot_mapping[T]` array — one slot per new token — that
the scatter kernel consumes (§9.1).

### 7.2 `append` (grow) and slot mapping for a batch

`AppendTokens(int64_t count) → StatusOr<std::vector<int64_t>>` (the slot mapping
for the `count` new tokens, at positions `[num_tokens_, num_tokens_ + count)`):

1. `need = blocks_needed(num_tokens_, count)` new physical blocks (boundary
   crossings).
2. **All-or-nothing:** `Allocate()` `need` blocks into a scratch vector; if any
   `Allocate` returns `ResourceExhausted`, `Release` the ones already taken and
   return `ResourceExhausted` — **table and pool are left exactly as before**
   (no half-grown table, mirroring v0's "a rejected append leaves the layer
   untouched"). Only on full success append the new ids to `blocks_`.
3. Build `slot_mapping[count]` from `slot(pos)` for `pos` in the new range;
   advance `num_tokens_`.

**Worked slot mappings** (`bs = 8` — the smallest size `BlockPool::Create`
accepts, §4; the M8-T03 tests reproduce exactly these numbers by priming the
pool's LIFO free list so `Allocate` returns `[5, 2]`):

- **Prefill, `T = 12` from empty** (`num_tokens_ 0 → 12`): needs
  `⌈12/8⌉ − 0 = 2` blocks. `Allocate` returns `[5, 2]` → `blocks_ = [5, 2]`.
  Positions 0..7 → block 5 (slots `40..47`); positions 8..11 → block 2 (slots
  `16..19`). The block-boundary straddle (pos 7→8) crosses from block 5 to
  block 2.
- **Decode, `T = 1`** at `num_tokens_ = 12` (→ 13): `⌈13/8⌉ − ⌈12/8⌉ = 2 − 2 = 0`
  new blocks; pos 12 → slot `blocks_[1]·8 + 4 = 2·8+4 = 20`.
- **Decode, `T = 1`** at `num_tokens_ = 16` (→ 17): `⌈17/8⌉ − ⌈16/8⌉ = 3 − 2 = 1`
  new block; `Allocate` → say `b`, `blocks_ = [5,2,b]`; pos 16 → slot
  `b·8+0`. This is the boundary-crossing decode that allocates.

### 7.3 `truncate` and `free`

- `Truncate(int64_t new_len)`: drop tokens past `new_len`. Physical blocks that
  become wholly empty (`new_len ≤ i·bs`) are `Release()`d and popped from
  `blocks_`; the partial surviving tail block is kept (its stale slots are
  overwritten on the next append, and read masked-out until then). `new_len` in
  `(0, num_tokens_]` only; `new_len == 0` releases all blocks. Under M8's
  exclusive-tail rule no released block is shared, so no CoW (§6.4).
- Destructor / `FreeAll()`: `Release()` every block, clear the table (RAII — a
  dropped `PagedKvCache` returns its blocks; the basis for M8-T08 / M9-T10 "no
  leaks, stats zero at end").

M8-T03 acceptance: growth across boundaries, hand-verified prefill/decode slot
mappings (§7.2), and blocks returned to the pool on free (stats-checked).

**M8-T03 implementation notes** (as built, `src/kvcache/block_table.{h,cpp}` —
additive clarifications of the §7.1–§7.3 sketch, no divergence):

- **`AppendTokens(count)` rejects `count <= 0` with `InvalidArgument`** (a
  forward always appends ≥ 1 token; the design named only the exhaustion path).
  All-or-nothing rollback is as specced: the new blocks are `Allocate`d into a
  scratch vector, and on any `ResourceExhausted` the taken blocks are `Release`d
  before returning — `blocks_`/`num_tokens_` and the pool are left byte-identical
  (a recovery test proves a smaller append then succeeds).
- **The block list is exposed as `std::span<const std::int32_t> blocks()`** —
  the contiguous, logical-order `const int32_t*` the M8-T05
  `PagedDecodeAttentionF32` reads (§8.3/§9.2), zero-copy, valid until the next
  append/truncate. A `slot(pos)` query (CHECK-guarded to committed positions)
  and `num_blocks()`/`num_tokens()`/`block_size()`/`pool()` round out the
  surface.
- **Move-only, mirroring `BlockPool`** — move-construct (leaving the source
  empty so its destructor releases nothing), move-assign deleted (a table is
  built in place, never reseated). The destructor calls `FreeAll` (RAII).
- **Not thread-safe by design** — one table per sequence on the engine thread;
  the pool's mutex covers the cross-sequence contention. `Truncate` releases
  wholly-empty blocks tail-first (the lowest logical block lands on top of the
  pool's LIFO free list, reused first) and keeps the partial surviving tail.
- **Tests use `bs = 8`** (the smallest `BlockPool`-valid size) with a primed
  free list to reproduce the §7.2 `[5, 2]` ordering, since the original `bs = 4`
  example is not an allocatable block size (§4). +23 tests.

---

## 8. `PagedKvCache` — the M5 interface, paged (`paged_cache.h`, M8-T04/T07)

`PagedKvCache` implements the abstract `KvCache` (model-execution.md §6.1)
unchanged, so `Attention` / `OptimizedModel` / `Generate` consume it exactly as
they consume `SimpleKvCache`. It holds a `BlockPool*` (shared, non-owning,
outlives the cache) and one `BlockTable`.

### 8.1 The verbs

- `geometry()` — the pool's `CacheGeometry`.
- `length()` — committed tokens. Because one block table drives all layers, all
  layers agree by construction; `length() == num_tokens_` after a completed
  forward (v0 reports the per-layer *min* because it fills layer-by-layer within
  a forward — the paged cache grows the table once per forward-token-batch, so
  it exposes the same "committed = all layers agree" semantics; see §8.2 for the
  within-forward ordering).
- `capacity()` — **advisory** under paging. Reports `num_tokens_ +
  free_blocks()·bs` (what *this* sequence could still grow into given the pool's
  current free blocks). It is a hint, not a guarantee: with a shared pool another
  sequence may consume free blocks first, so `append`'s `ResourceExhausted` is
  the *binding* capacity check (M8-T08). `Generate`'s up-front worst-case check
  (M5-T09, uses `capacity()`) becomes a best-effort early rejection; the real
  guard is per-append. Documented so M8-T07 does not mistake it for a hard bound.
- `append(layer, k, v)` — see §8.2.
- `view(layer)` — **gathers** the layer's `[Hkv, len, d]` head-major history into
  a fresh contiguous tensor by walking the block table (`paged_gather`, §9.3).
  This *keeps* v0's "view may copy" contract (model-execution.md §6.4) — it is
  the M8-T06 prefill read path and the reference/test read path. It is **not** on
  the decode hot path (§8.3).
- `truncate(new_len)` — `BlockTable::Truncate` (§7.3).

### 8.2 `append` under paging: grow-table + scatter

`append(layer, k, v)` with `k, v : [T, Hkv, d]` token-major:

1. **Layer-0 grows the table.** The block table is layer-independent, but
   `append` is called once per layer per forward (in layer order,
   model-execution.md §6.1). Growing on **layer 0** and reusing the resulting
   slot mapping for layers 1..L−1 keeps the table advance atomic per
   forward-token-batch. Implementation: `PagedKvCache` remembers the current
   forward's slot mapping — computed (and blocks allocated, all-or-nothing) on
   the `layer == 0` call, reused for `layer > 0`. A layer-0 `ResourceExhausted`
   aborts the whole forward before any K/V is written (no partial state); layers
   never disagree because only layer 0 mutates the table. (The `Attention`
   modules already call layer 0 first; the contract is documented, and an
   out-of-order or repeated layer-0 call is `InvalidArgument`.)
2. **Scatter.** `KvScatterF32` (§9.1) writes the `[T, Hkv, d]` K and V into the
   layer's slabs at the slot mapping. Front-loaded shape/dtype/contiguity
   validation (like v0's `append`) *before* the write, so a rejected append
   leaves storage untouched.

`length()` advances only when layer 0 grows the table; within a forward, layers
1..L−1 write into slots already counted, so "committed = all layers agree" holds
after the final layer's append — identical observable semantics to v0.

### 8.3 The one additive accessor: `paged_view` (the decode fast path)

A paged cache **cannot** hand back a single contiguous `[Hkv, len, d]` slice —
the history is scattered across blocks — so decode must read *through* the block
table. Gathering a full `[Hkv, len, d]` copy per layer per decode step (what
`view()` does) would double KV memory traffic every step and blow M8-T07's ≤10%
regression bound. Therefore the interface gains **one additive virtual**:

```cpp
struct PagedKvView {
  const float* k_slab;      // layer's K slab base
  const float* v_slab;      // layer's V slab base
  const int32_t* block_table;  // logical→physical, this sequence
  int64_t num_blocks;       // valid entries in block_table
  int block_size;           // bs
  int64_t length;           // committed tokens (last block possibly partial)
  int64_t block_stride;     // Hkv·bs·d
  // Hkv, d come from geometry()
};

// On the base KvCache interface:
[[nodiscard]] virtual core::StatusOr<PagedKvView> paged_view(int layer) const {
  return core::UnimplementedError("paged_view: cache is not paged");
}
```

`SimpleKvCache` inherits the default `Unimplemented`. `PagedKvCache` overrides it
to return the pointers/strides for the layer with **zero copy**. The optimized
decode path (§9.2) tries `paged_view` first and, on `Unimplemented`, falls back
to `view()` + the M6 contiguous `DecodeAttentionF32` (so `SimpleKvCache` keeps
working unchanged, and the paged path is exercised only when present). This is
the *only* change to consumer code (`OptimizedModel::forward`'s attention step);
everything else in generator/attention is untouched — satisfying
model-execution.md §6.4's "the code above the interface does not change" for all
paths but this deliberate, tested fast path. The default-implemented virtual
(vs. a `dynamic_cast`) keeps the seam explicit, testable, and layering-clean.

`kv_cache.h`'s class comment is amended (T04) to document `paged_view` and the
now-permitted `kvcache → kernels` edge.

**M8-T04 implementation notes** (as built, `src/kvcache/paged_cache.{h,cpp}` —
additive clarifications of §8, no divergence):

- **Construction cannot fail.** `PagedKvCache(BlockPool* pool)` is a plain
  constructor (CHECK non-null pool), not a `StatusOr` factory — the slabs were
  allocated when the pool was created, so nothing here allocates. Move-only,
  mirroring `BlockTable`/`BlockPool`: move-construct (moved-from left empty via
  the block table's own move), move-assign deleted.
- **The layer protocol is enforced, not merely assumed.** `next_layer_` (0
  between forwards) tracks the expected layer: an append whose `layer !=
  next_layer_` is `InvalidArgument`, and a `layer > 0` append whose `T` differs
  from the layer-0 batch is `InvalidArgument` — so an out-of-order, repeated, or
  short append is rejected rather than corrupting the cache. Layer 0 grows the
  table (all-or-nothing) and records the slot mapping in `pending_slots_`;
  layers 1..L−1 reuse it; `next_layer_ = (layer + 1) % L` marks the forward
  complete on wrap. `truncate` resets `pending_slots_`/`next_layer_` (a truncate
  between or mid-forward re-opens the protocol at layer 0). This is stricter
  than §8.2's "documented contract" — the design left it to the caller; the
  implementation checks it, since the cost is one integer compare off the write.
- **`view(layer)` is `Unimplemented` in T04**, deferred to the M8-T06 paged
  gather (§9.3) whose body it will become. The decode path never calls it — it
  reads through `paged_view` (§8.3) with zero copy — so nothing consumes `view`
  until T06 wires the prefill read path. `SimpleKvCache::view` is unaffected.
- **`capacity()` reports `num_blocks·bs + free_blocks·bs`** (owned-slot capacity,
  which includes the partial tail block's still-free slots, plus the free pool),
  a small correction to §8.1's `num_tokens + free_blocks·bs`: the latter
  under-counts the tail block's free slots and would make `Generate`'s up-front
  check (and the models' `length()+T > capacity()` guard) spuriously reject
  tokens that fit in an already-owned block. It stays **advisory** (a shared
  pool's free blocks may be taken by another sequence first — the binding check
  is `append`'s per-forward `ResourceExhausted`, M8-T08).

---

## 9. Kernels

All kernels are layout-agnostic raw-pointer entries (§2): the caller
(`PagedKvCache` / `paged_gather`) front-loads every precondition, so nothing
inside a parallel region does anything but arithmetic (cpu-backend.md §5).

### 9.1 `KvScatterF32` (M8-T04) — the paged append

```cpp
// Writes T new tokens' K/V into paged slabs. src_k/src_v: [T, Hkv, d]
// token-major (as the projections produce them). slot_mapping[T]: flat slot
// per new token (§7.1). k_slab/v_slab: layer slab bases. Pure copy + widen-free
// (fp32→fp32). Class E (bit-exact), no reduction, thread-safe (disjoint slots).
void KvScatterF32(const float* src_k, const float* src_v,
                  const int64_t* slot_mapping, int64_t t_dim,
                  int64_t kv_heads, int64_t d, int64_t block_size,
                  float* k_slab, float* v_slab);
```

For token `t`, kv head `h`: source row `src_k + (t·Hkv + h)·d`; destination
`slot = slot_mapping[t]`, block `= slot / bs`, in-block `p = slot % bs`, dest
`k_slab + (block·Hkv + h)·bs·d + p·d` (§3.1). A pure `memcpy` of `d` floats per
(token, head) — no arithmetic, so **bit-exact** and ISA-independent; a **single
TU** (no per-ISA variant), like the M6 embedding gather. It replaces the
contiguous `SimpleKvCache::append` transpose-copy. M8-T04 acceptance: scattered
writes land in the exact expected block/offset, verified by reading back and
comparing against an **independently-simulated** paged layout (a test-local
plain-array model), across boundary-straddling prefills and single-token
decodes.

**As built** (`src/kernels/kv_scatter.{h,cpp}`): the signature is §9.1 verbatim.
Tokens are the unit of `parallel_for` work (grain 16 = one `bs=16` block's
tokens per worker), each writing disjoint slots — race-free and thread-count
bit-identical trivially (a copy has no reduction). Because there is no
dispatched (per-ISA) path — a single scalar TU ships — `kv_scatter_test`
registers **no `SCALAR_PASS`** (the forced-scalar pass would rerun the identical
bytes); the design's §12 M8-T04 entry is read accordingly.

### 9.2 `PagedDecodeAttentionF32` (M8-T05) — decode through the block table

```cpp
// out[H, d] = softmax(scale · q·Kᵀ) · V over the whole cache, reading K/V
// through a block table. Bit-identical to DecodeAttentionF32 on the same
// logical K/V (§4 constraint). Threaded across Hkv kv heads (M6-T05 parity).
void PagedDecodeAttentionF32(const float* q,               // [H, d]
                             const float* k_slab, const float* v_slab,
                             const int32_t* block_table, int64_t num_blocks,
                             int64_t length,               // cached keys
                             int64_t heads, int64_t kv_heads, int64_t d,
                             int64_t block_size, int64_t block_stride,
                             float scale, float* out);      // [H, d]
```

The recurrence is M6-T05's `DecodeUnitsImpl`/`DecodeGroupSlice` (the four `Ops`
primitives — `DotScoreRow`, `ExpRowSum`, `ScaleRow`, `AxpyRow` — reused
verbatim), with one change: instead of indexing a contiguous `k_head + s·d`, it
walks the block table. Because `bs | kAttnKb` (§4), a `kAttnKb`-key unit maps to
`kAttnKb/bs` consecutive block-table entries; the kernel gathers those blocks'
`[bs, d]` key tiles into the same fixed 64-wide `scores` row (or, since each
block tile is already contiguous, runs the dot per block into the correct
`scores` offset) and performs the identical max/expsum/scale/axpy over the unit.
Per fixed query head the block sequence, the `n_valid` per unit, the
first-unit `alpha = exp(−inf) = 0`, and the four Ops calls are identical to
`DecodeAttentionF32` — so the result is **bit-identical by construction**
(M8-T05 acceptance: matches the M6-T05 contiguous kernel *exactly* for cache
lengths crossing many blocks, including length exactly at a block boundary),
and bit-identical across thread counts (each kv head's whole recurrence in one
thread). Per-ISA TUs (`scalar/neon/avx2`) instantiate the one template, AVX2
written blind + proven by CI, exactly like M6. `SCALAR_PASS` registered.

**As built** (`src/kernels/paged_attention.{h,cpp}` +
`internal/paged_attention_common.h`) — additive clarifications of §9.2, no
divergence:

- **The signature is §9.2 verbatim.** `PagedDecodeAttentionF32` builds a
  `PagedDecodeArgs` and threads over `kv_heads` (grain 1), exactly like
  `DecodeAttentionF32` — the same parallel width (`Hkv`), so the §8 idle-core
  note (decode leaves cores idle when `Hkv < cores`, the M12-T03 flash-decoding
  motivator) carries over unchanged.
- **The recurrence is single-sourced but a separate template.**
  `PagedDecodeUnitsImpl` / `PagedDecodeGroupSlice` live in the new
  `internal/paged_attention_common.h` (not in `attention_common.h`, which stays
  byte-untouched so M6's includer set is unperturbed). Per unit it scores
  **block-by-block** — one `DotScoreRow(q, k_tile, n_b, d, scale, scores +
  b·bs)` per physical block, folding the per-block maxes with `>` — and axpys
  in ascending key order walking the block table once per block. Because each
  physical `[bs, d]` tile is already contiguous, this is the "runs the dot per
  block into the correct `scores` offset" variant of §9.2; the resulting
  `scores[0..n_valid)` row, the unit max, and the ascending V accumulation are
  byte-identical to the contiguous call, hence the bit-identity.
- **The per-ISA variants live in the existing `{scalar,neon,avx2}/attention.cpp`
  TUs**, each a one-line `internal::PagedDecodeUnitsImpl<XxxOps>(…)` reusing the
  *same* anonymous-namespace `Ops` structs as `PrefillUnits`/`DecodeUnits` —
  reusing those primitives **literally** is the bit-identity guarantee, so there
  is no separate `{isa}/paged_attention.cpp`. `paged_attention.cpp` holds only
  the `KernelTable` dispatch + `parallel_for`, and the `detail::
  PagedDecodeAttentionVariant(Isa)` test seam.
- **`bs | kAttnKb` is CHECKed at the public entry** (`block_size > 0 && kAttnKb %
  block_size == 0`), re-asserting the pool-construction invariant a violating
  `bs` would silently break (ADR-003 CHECK territory, death-tested); `num_blocks`
  is a caller-side bound only (`ceil(length/bs) ≤ num_blocks`), never read past.
- **Tests** (`tests/unit/paged_decode_attention_kernel_test.cpp`, `kernels`
  label, `SCALAR_PASS`): **bitwise** equality (`allclose atol=rtol=0`) to
  `DecodeAttentionF32` on the same logical K/V materialized into a *test-local*
  paged layout (independent of `KvScatterF32`) — a **reverse-permuted** block
  table with **poisoned** unused physical blocks and tail slots, so a
  contiguous-fallback or an over-read fails — across `bs ∈ {8,16,32,64}`,
  many-block and exact-boundary lengths, GQA ratios (incl. `g=12>chunk`), head
  dims {18,24,64,128}, thread counts (threaded == serial == manual-chunked), the
  HF attention goldens (Class T vs the oracle), and an **end-to-end pass through
  a real `PagedKvCache`/`BlockPool`** (append token-by-token, feed `paged_view`
  to the kernel — bit-exact vs contiguous, proving the `PagedKvView` field
  semantics). 8 cases ×2 (SCALAR_PASS) → +16 ctest.
- **Bench** (`benchmarks/kernels/attention_bench.cpp` `paged` mode,
  BASELINES.md M8-T05): the isolated paged kernel is ~0.87–0.89× the contiguous
  decode kernel per call (the block-indirection cost) — but it **replaces the
  `view()` gather** the contiguous path needs in the engine (optimized-cpu-
  execution.md §8: ~12% decode memory-traffic overhead), so the whole-step
  comparison is M8-T07's job, not this microbenchmark's.

### 9.3 `paged_gather` (M8-T06) — prefill read path

`GatherLayerKV(pool, block_table, layer, length) → StatusOr<KvView>` walks the
block table and copies each block's `[Hkv, bs, d]` tile (clipping the last block
to `length % bs`) into a fresh contiguous `[Hkv, length, d]` head-major tensor —
exactly the shape the M6 `PrefillAttentionF32` reads. This *is* `PagedKvCache::view`'s
body; prefill (T > 1, including prefill-continuation with `P > 0` cached tokens)
gathers once per layer and calls the **unchanged** M6 blocked prefill kernel.
Simple and correct; a block-walking prefill kernel is M12. The gather is
independently unit-tested (M8-T06 acceptance) and, fed to prefill, matches the
reference backend on prefill-with-existing-cache.

**M8-T06 implementation notes** (as built — additive clarifications of §9.3/§8,
no divergence):

- **The copy is a layout-agnostic kernel, not inline `kvcache` code.** The tile
  copy lives in the layer-1 kernel `KvGatherF32(k_slab, v_slab, block_table, bs,
  length, Hkv, d, out_k, out_v)` (`src/kernels/kv_gather.h`) — the exact mirror
  of `KvScatterF32`. `GatherLayerKV` (`src/kvcache/paged_gather.h`) is the thin
  `kvcache` wrapper: it validates, allocates the two `[Hkv, length, d]` outputs
  with `Tensor::empty` (every element is overwritten — no zero-fill), and calls
  the kernel over the same PRIVATE `kvcache → kernels` edge as the scatter. This
  keeps `parallel` out of `kvcache` (the kernel threads over `(head, block)`
  tiles) and lets M12/M13 change the physical layout without touching the
  wrapper. Per-forward work unit: one `memcpy` of `rows·d` floats per (head,
  block), `rows = min(bs, length − b·bs)`; **Class E (bit-exact across ISAs and
  thread counts)** — a single scalar TU ships, no per-ISA variant, no
  `SCALAR_PASS` (like the scatter and the embedding gather).
- **Fresh tensor, not a workspace slot.** The gather owns its output storage (the
  `KvView` contract), so it allocates per call rather than reusing an
  `OptimizedModel::Workspace` slot. A workspace-resident gather buffer (to make
  prefill allocation-free) is a T07/M12 option, not needed for correctness; decode
  never gathers (it reads through `paged_view`, §8.3).
- **`PagedKvCache::view` exposes a per-layer visible length.** Because layer 0
  grows the table (bumping `length()`) while layers `≥ next_layer_` have their new
  slots allocated but unwritten mid-forward, `view(layer)` gathers only the
  layer's *committed* length — `length()` when the forward is complete or `layer <
  next_layer_`, else `length() − pending`. This reproduces `SimpleKvCache::view`'s
  per-layer-fill semantics exactly (a not-yet-appended layer sees the prior
  committed history, never stale slot bytes) and costs one integer compare. The
  real consumers always call `view(layer)` right after `append(layer)`, so they
  always see the full committed length; the rule only matters for tests that view
  an un-appended layer.

### 9.4 Batched shape (reserved for M9)

M9-T07's batched decode needs a `[B, max_blocks]` int32 block-table tensor
(rows padded with `−1`) and a `[B]` lengths vector, with the kernel looping
sequences and skipping `−1` entries. The single-sequence signature above is the
`B = 1` case; the batched entry (`PagedDecodeAttentionBatchedF32`) is added in
M9, not here. Flagged so the single-sequence layout does not paint M9 into a
corner (the block-table pointer + strides already generalize to a padded 2-D
tensor).

---

## 10. Engine integration (M8-T07) & exhaustion (M8-T08)

### 10.1 Where the pool lives

The `BlockPool` is owned by the engine/driver (M8-T07: `engine generate`; M9: the
runtime), **outliving every `PagedKvCache`** that borrows it (the allocator
lifetime rule, §6.1). In the M8 single-request path: build model → compute the
KV budget (§5) → `BlockPool::Create` → per request a `PagedKvCache(pool)` →
`Generate` as today. `BuildOptions` / CLI gain `--kv-cache-memory` and
`--kv-block-size`; the pre-paging `--cache-capacity N` (tokens) still works
(§5.1).

**M8-T07 acceptance:** tiny-fixture greedy output **identical** to the pre-paging
engine (the KV invariant under a new storage backend — model-execution.md §6.2's
"same invariant reappears as M8-T07's identical-to-pre-paging"), because the
scatter is a bit-exact copy and the paged decode kernel is bit-identical to the
contiguous one; the real Qwen2-0.5B generates coherent text with cache stats
logged; and `bench_generate` shows **≤ 10%** decode-throughput regression vs the
M6 baseline (the `paged_view` fast path — §8.3 — is what keeps it under, by
avoiding a per-step full-history copy; number recorded in BASELINES.md).

**M8-T07 as built.** The decode fast path is wired in
`OptimizedModel::ForwardLayer` (the *only* consumer change, §8.3): decode
(`T == 1`) calls `cache.paged_view(layer)` first and, on `Unimplemented`, falls
back to `view()` + the contiguous decode kernel so `SimpleKvCache` is untouched;
a non-`Unimplemented` status propagates (a real cache failure is not masked).
Prefill always gathers via `view()`. The budget plumbing landed as pure,
unit-tested helpers: `core::host_memory_bytes()` (the §5.3 host-RAM helper,
macOS `sysctl` / Linux `sysconf`), `model::weight_resident_bytes(loaded)` (the
`weights_bytes` term, computed from the checkpoint before `BuildModel` moves
`loaded`; dedups the tied embedding), `Workspace::BytesFor(config, T)` (the
`workspace_bytes` term), and `kvcache::{ParseKvCacheMemory, ResolveKvBudgetBytes,
BlocksForTokens}` (`kv_budget.h` — flag parse → absolute budget → block count).
The driver owns `unique_ptr<BlockPool> pool` (declared first) and
`unique_ptr<KvCache> cache` (destroyed first), and logs
`BlockPool::stats()` after generation. **Allocator:** the M8 single-request
pool draws its slabs from the **default CPU allocator**, not the M2 caching
allocator — a single-request pool allocates once and frees at teardown, so the
caching allocator (built to amortize churn) buys nothing here; it wires in with
the M9 runtime, where pools churn across requests. **Measured:** paged decode is
1.6–4.2% below the same-machine `--kv-cache simple` baseline (BASELINES.md
M8-T07), inside the ≤10% bound and inside the machine's own run-to-run noise.

### 10.2 Exhaustion (M8-T08)

Until the M9 scheduler brings preemption, a pool that runs dry mid-generation is
a **graceful error**: `BlockPool::Allocate` → `ResourceExhausted` → `BlockTable::AppendTokens`
propagates it → `PagedKvCache::append` returns it → `Generate` surfaces it as the
request's failure. Crucially, **all of that sequence's blocks are still reclaimed**
when its `PagedKvCache` drops (RAII, §7.3) — M8-T08 acceptance asserts pool stats
return to zero (no leaked blocks) after a generation that exceeds capacity. The
cache-usage stats API (`BlockPool::stats()` — blocks used/free, utilization,
§6.2) is the surface M9's scheduler admits against and M16's metrics endpoint
exports.

---

## 11. Interactions with later milestones

- **M9 (continuous batching).** The pool is the shared resource the scheduler
  admits against: `free_blocks()` / `blocks_needed()` (§6.2) drive
  block-availability admission; a batched `[B, max_blocks]` block-table tensor
  (§9.4) drives batched decode; **preemption** frees a victim's blocks
  (`BlockTable::FreeAll`), sets it PREEMPTED, and re-prefills prompt+generated on
  resume (the re-prefill equivalence is the KV invariant again). No storage
  change — M9 adds a scheduler and the batched kernel entry, both on seams
  reserved here.
- **M11 (prefix caching).** `Share`/`Release` refcounts and the immutability
  invariant (§6.3/§6.4) are the whole foundation: M11 adds a content-hash index
  (`prefix_index.h`), hashes full blocks as they fill (a hook in
  `BlockTable::AppendTokens` when a block completes), adopts the longest cached
  prefix into a new sequence's table (`Share` the blocks, prefill only the
  suffix), and evicts refcount-0 CACHED blocks in LRU order before the pool
  reports exhaustion. The `truncate`-into-shared copy-on-write edge (§6.4) is
  built when M11/M15 first need it.
- **M12 (tuning/fusions/chunked prefill).** Flash-decoding (M12-T03) splits one
  sequence's blocks across threads with a deterministic reduction — the paged
  decode kernel's per-kv-head loop is the split point. RoPE+KV-write fusion
  (M12-T04) needs a **raw-slot write accessor** on `PagedKvCache` (write K/V for
  a slot mapping without a separate projection tensor) — reserved, not built.
  Chunked prefill (M12-T06) is just the prefill-continuation `(P, T)` path
  (model-execution.md §6.3) repeated over the paged cache — already supported.
- **M13 (INT8 KV).** `--kv-cache-dtype int8` makes the slabs INT8 + per-block-
  per-head fp32 scales (§3.4); scatter quantizes, the attention kernels dequant
  inside the loop (added parameters, not new functions).
- **M15 (speculative rollback).** Verification scores k+1 positions in one
  forward; on rejection at position j, `truncate(j)` rolls back the paged cache —
  block-table truncation + releasing emptied blocks (§7.3), including rollbacks
  across block boundaries. The cache-integrity invariant (rollback-then-redecode
  == never-speculated) is the KV invariant under `truncate`.

---

## 12. Testing strategy (per ticket)

The correctness chain is unchanged (cpu-backend.md §methodology): HF fixtures
validate the reference; the reference validates the paged path. Concretely:

- **M8-T02 (`block_pool`).** Unit tests: exhaustion → `ResourceExhausted` (no
  crash); double-`Release` → `CHECK` (death test); `stats()` accurate through
  scripted `Allocate`/`Release`/`Share` sequences; `Share` on free block →
  `CHECK`; free-list LIFO reuse observable.
- **M8-T03 (`block_table`).** Growth across block boundaries; hand-verified slot
  mappings for prefill (T tokens) and decode (1 token), incl. the §7.2 straddle
  and boundary-crossing-decode-allocates cases; all-or-nothing append leaves
  pool untouched on exhaustion; blocks returned to pool on `Truncate`/free
  (stats-checked).
- **M8-T04 (`KvScatterF32` + `PagedKvCache::append`).** Readback vs an
  independently-simulated paged layout (test-local plain array), across
  boundary-straddling prefills and single-token decodes; append validation
  (shape/dtype/contiguity) front-loaded (rejected append leaves storage
  untouched).
- **M8-T05 (`PagedDecodeAttentionF32`).** **Bitwise** equality to
  `DecodeAttentionF32` on the same logical K/V materialized both ways, for cache
  lengths crossing many blocks and length exactly at a block boundary; GQA
  ratios; bit-identical across thread counts; `SCALAR_PASS`.
- **M8-T06 (`paged_gather` + prefill).** Gather independently tested vs a
  simulated layout; prefill-with-existing-paged-cache matches the reference
  backend (prefill-continuation `P > 0`).
- **M8-T07 (integration).** Tiny-fixture greedy output **identical** to the
  pre-paging engine on both fixtures (the regression guarantee); the KV
  invariant (token-by-token == full-recompute) bit-exact through the paged
  cache; real-model generation smoke; `bench_generate` decode regression ≤ 10%
  recorded.
- **M8-T08 (exhaustion).** A generation exceeding capacity fails gracefully with
  `ResourceExhausted`; pool stats zero after (no leaked blocks — RAII
  reclamation asserted).

Suites touching a kernel or a full forward register `SCALAR_PASS` so the
forced-scalar pass (`ENGINE_FORCE_ISA=scalar`) covers the shipped bytes on both
CI (x86-64) and the arm64 dev machine.

---

## 13. Deferred / open items (recorded)

- **Copy-on-write `truncate` into a shared block** — needed only when M11/M15
  first truncate a shared prefix; the mechanism is stated (§6.4) but unbuilt.
- **Per-layer variable allocation** — the per-layer slab split (§3.2) permits
  later per-layer block budgets (e.g. M13 mixed-precision layers); not exploited
  in M8.
- **Block-walking prefill kernel** — M8-T06 gathers; a direct paged prefill
  kernel is M12.
- **Batched block-table tensor / batched decode kernel** — shape reserved
  (§9.4), built in M9-T07.
- **Host-RAM detection helper** — landed in M8-T07 as `core::host_memory_bytes()`
  (§5.3, §10.1 as-built), the fraction spelling's first consumer.
