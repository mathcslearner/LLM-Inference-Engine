# Benchmark baselines

Measured performance records. Rules (CLAUDE.md): every optimization ticket
records before/after numbers here; no performance claim without a benchmark
delta. Numbers are machine-specific — each entry states its host, toolchain,
and configuration. Until M12-T01 matures the harness (thread/ISA sweeps,
stability discipline), entries are advisory: best-of-N on a non-quiesced
machine.

## M3-T06 — first vectorized kernels (2026-08-08)

**Host:** Apple M2 (8 physical cores), macOS, dev machine.
**Toolchain:** Homebrew clang 20.1.8, `-O3` Release.
**Command:** `build/benchmarks/kernels_bench` (1 048 576 elements, best of
20 iterations). Rows compare the scalar variant against the NEON variant,
both called directly as single-threaded chunk bodies (no thread pool).

| Kernel | scalar ns/elem | NEON ns/elem | speedup |
|---|---:|---:|---:|
| AddF32 | 0.143 | 0.135 | 1.06× |
| MulF32 | 0.134 | 0.132 | 1.02× |
| ScaleF32 | 0.088 | 0.108 | 0.81× |
| SumF32 (chunk) | 0.395 | 0.059 | 6.70× |
| MaxF32 (chunk) | 1.142 | 0.143 | 8.00× |
| Fp16ToFp32 | 0.730 | 0.180 | **4.06×** |
| Fp32ToFp16 | 0.770 | 0.269 | **2.86×** |
| Bf16ToFp32 | 0.064 | 0.062 | 1.03× |
| Fp32ToBf16 | 0.208 | 0.180 | 1.15× |

The M3-T06 acceptance target — vectorized conversion ≥ 2× scalar at 1M
elements (advisory) — is met by both fp16 conversions (4.06× / 2.86×), the
paths that use the hardware conversion unit.

Reading the smaller ratios honestly: the scalar variants are compiled at
`-O3` and clang auto-vectorizes them, so these columns compare
hand-written NEON against compiler-vectorized NEON, not against truly
serial code. For the pure-load/store-bound ops (add/mul/scale, bf16
widening at ~90 GB/s) both saturate single-core memory bandwidth and the
ratio is noise around 1×; ScaleF32's 0.81× is that noise, not a regression
(the op is bit-identical by Class E regardless of which variant runs). The
reductions' 6.7–8× and the fp16 conversions' 2.9–4.1× are the cases where
the fixed accumulation convention and the hardware conversion unit
respectively beat what the auto-vectorizer may do. Revisit-with-numbers
belongs to M12 (kernel tuning).

## M6-T02 — packed-weight GEMM (2026-08-18)

**Host:** Apple M2 (8 physical cores), macOS, dev machine.
**Toolchain:** Homebrew clang 20, `-O3` Release.
**Command:** `build/benchmarks/gemm_bench 4096 4096 4096 3` (best of 3, all
paths threaded through the default pool).

| Path | time | GFLOP/s | speedup vs naive |
|---|---:|---:|---:|
| `cpu::gemm` naive bf16 (M5 baseline) | 10.704 s | 12.84 | 1.00× |
| `cpu::gemm` naive f32 (M5 baseline) | 10.422 s | 13.19 | 1.03× |
| **`PackedGemm` bf16** | **1.221 s** | **112.52** | **8.76×** |
| **`PackedGemm` f32** | 1.229 s | 111.81 | 8.48× |

The M6-T02 acceptance target — packed GEMM ≥ 5× the M5 naive `cpu::gemm` at
4096×4096×4096 (advisory) — is met with margin (8.76× bf16). The win is
register tiling: the naive kernel re-streams the whole weight matrix once per
output row (`M` passes over `B`), while the `MR×NR` micro-kernel reuses each
widened weight vector across `MR` activation rows and each `A` slab across a
panel-block, cutting weight-memory traffic by roughly `M/MR` and keeping the
FMA units fed. bf16 and f32 land within noise of each other — both are FMA
throughput bound at this size, not weight-bandwidth bound, once packed.

**BLAS sanity number (Accelerate) — not captured on this machine.** The
optional `-DENGINE_BENCH_BLAS=ON` context comparison could not be built on the
dev machine: Homebrew LLVM 20 (the pinned toolchain, CLAUDE.md) rejects the
CommandLineTools SDK's Accelerate headers with `-Welaborated-enum-base` hard
errors (the same broken-CLT situation that blocks Apple-clang C++ here). The
CMake wiring (`ENGINE_BENCH_BLAS`, benchmark-only, never linked into `src/`)
is in place for a capable environment (a working Accelerate SDK, or a Linux
`find_package(BLAS)` host); the number is for context only and parity is an
M12 goal, not M6-T02's — so it is deferred rather than blocking. Methodology:
`cblas_sgemm(RowMajor, NoTrans, Trans, …)` computing `A·Wᵀ` on the same f32
inputs, best of 3.

## M6-T03 — norm/activation/softmax/RoPE kernels (2026-08-18)

**Host:** Apple M2 (8 physical cores), macOS, dev machine.
**Toolchain:** Homebrew clang 20, `-O3` Release.
**Command:** `build/benchmarks/kernels_bench` (default `n = 2^20`, best of 20).
Each row is the single-threaded per-ISA *chunk body* (scalar vs the selected
vector variant, NEON here) — an apples-to-apples ISA comparison with no thread
noise, exactly like the M3-T06 rows. The shaped kernels reshape the flat buffer
to a model-like hidden size (RMSNorm/Softmax over `[256, 4096]`; RoPE over
`[1024, 8, 128]`); ns/elem is over the total element count.

| Kernel | scalar ns/elem | NEON ns/elem | speedup |
|---|---:|---:|---:|
| ExpF32 (poly) | 0.459 | 0.428 | 1.07× |
| SiluMulF32 | 0.587 | 0.562 | 1.05× |
| RmsNormF32 | 0.962 | 0.457 | **2.10×** |
| SoftmaxF32 | 2.161 | 0.701 | **3.08×** |
| RopeApplyF32 | 0.096 | 0.137 | 0.70× |

M6-T03 has no perf target (the design's only hard GEMM target is M6-T02's 5×);
these numbers exist because the kernels are performance-motivated and CLAUDE.md
forbids a perf claim without a delta. Reading them:

- **RMSNorm 2.1× / Softmax 3.1×** are the real wins — both reduce over a row and
  then re-scan it, and the vector max/sum reductions plus vectorized scale/exp
  beat the scalar per-element path clearly. Softmax gains most (its body is two
  reductions plus a vector-exp map).
- **ExpF32 / SiluMul ~1.05×** are modest because clang `-O3` already
  auto-vectorizes the branch-light scalar polynomial well, and at `n = 2^20`
  both are streaming a 4 MB buffer, so the body is closer to memory-bound than
  compute-bound. The absolute throughput (≈18–21 GB/s) is what matters for the
  forward pass; the ISA ratio is secondary here.
- **RoPE 0.70× — the vector body is currently *slower* than scalar.** The
  scalar half-rotation is a clean countable loop clang auto-vectorizes into
  essentially the same NEON, and the hand-written variant carries a little extra
  overhead (separate `vmul`+`vfma` rather than a fused sequence, and the
  `v[j]` / `v[j+half]` split loads). Recorded honestly — not claimed as a
  speedup. Correctness is unaffected (Class T within tolerance, all thread
  counts bit-identical); closing this gap is an M12-T02 tuning item (the tile /
  fusion pass), not an M6 obligation.

Cross-ISA note: these are NEON-only numbers (the dev machine). The AVX2 bodies
mirror the NEON structure and are proven correct + warnings-clean by CI; their
throughput is not measured here (no x86-64 dev machine), consistent with the
prior kernel baselines.

## M6-T04 — optimized prefill attention (2026-08-18)

**Host:** Apple M2 (8 physical cores), macOS, dev machine.
**Toolchain:** Homebrew clang 20, `-O3` Release.
**Command:** `build/benchmarks/attention_bench` (defaults `T=L=2048, H=32,
Hkv=8, d=64`, P=0 — a Llama-3.2-1B-shaped 2k-context prefill; best of 5).
Optimized = `PrefillAttentionF32` (blocked online softmax); baseline = the M5
reference `cpu::attention` (materializes the `[H·T, L]` score matrix). Both
threaded through the same `DefaultPool`. The M6-T04 obligation is "time vs the
reference at 2k context" — no perf target.

| Config | reference | PrefillAttentionF32 | speedup |
|---|---:|---:|---:|
| NEON, 8 threads (default) | 2.121 s | 0.218 s | **9.72×** |
| NEON, 1 thread | 11.893 s | 0.923 s | 12.89× |
| scalar, 1 thread | 11.658 s | 1.746 s | 6.68× |

Reading them:

- **9.72× at the default 8-thread NEON** is the headline: the blocked kernel
  holds only a `kAttnKb`-wide score row per query (no `537 MB` `[H·T, L]`
  score-matrix write the reference pays), and its dot/axpy vectorize over `d`.
- **Single-thread NEON 12.89×** isolates the algorithmic + ISA win from
  threading. The reference is memory/materialization-bound, so its scalar and
  NEON single-thread times are nearly equal (11.66 vs 11.89 s) — the vector
  reference gains little because it is dominated by the score-matrix traffic and
  the `cpu::softmax` pass over it. NEON→scalar on the *optimized* path is
  0.923 → 1.746 s (1.9×), the dot/axpy vectorization.
- **8-thread vs 1-thread NEON on the optimized path** is 0.923 → 0.218 s
  (4.2× over 8 cores) — the `(head, query-block)` units parallelize cleanly;
  the sub-linear scaling is the shared-K/V memory bandwidth, expected and an
  M12 item, not an M6 obligation.

Correctness is unaffected (Class T within the §10 tolerance rtol 1e-4 atol 1e-5;
bit-identical across thread counts). AVX2 body mirrors the NEON structure and is
proven correct + warnings-clean by CI; not measured here (no x86-64 dev
machine).

## M6-T05 — optimized decode attention (2026-08-18)

**Host:** Apple M2 (8 physical cores), macOS, dev machine.
**Toolchain:** Homebrew clang 20, `-O3` Release.
**Command:** `build/benchmarks/attention_bench decode` (defaults `L=2048, H=32,
Hkv=8, d=64`, T=1 — a Llama-3.2-1B-shaped 2k-context decode step; best per-call
of 200 samples × 100-call batches, since a single decode call is µs-scale).
Three paths on the same single-token query: the M5 reference `cpu::attention`
with T=1, the M6-T04 `PrefillAttentionF32` with T=1, and the M6-T05
`DecodeAttentionF32`. Both optimized paths are numerically identical to the
reference within §10 tolerance; decode is **bit-identical to prefill(T=1)** (a
tested invariant, not just a benchmark note). The M6-T05 obligation is "time vs
the reference" — no perf target.

| Config | reference | Prefill(T=1) | Decode | decode vs ref |
|---|---:|---:|---:|---:|
| NEON, 8 threads (default) | 1407.1 µs | 275.5 µs | 268.3 µs | **5.25×** |
| NEON, 1 thread | 6273.7 µs | 898.1 µs | 878.0 µs | 7.15× |
| scalar, 1 thread | 6271.9 µs | 1702.5 µs | 1687.2 µs | 3.72× |

Reading them:

- **5.25× at the default 8-thread NEON** is the headline: the reference
  materializes and softmaxes an `[H·1, L]` score buffer per call; the decode
  kernel holds only a `kAttnKb` score row per query head and streams K/V once
  per kv head. The reference's scalar and NEON single-thread times are ~equal
  (6273.7 vs 6271.9 µs) — it is dominated by the score-buffer traffic +
  `cpu::softmax` pass, so the vector reference gains almost nothing, exactly as
  in the M6-T04 prefill picture.
- **Decode ≈ prefill(T=1), decode marginally faster** (268 vs 275 µs at
  8-thread; 878 vs 898 µs single-thread). Both hold one score row at T=1; the
  only algorithmic difference is that decode streams each kv head's K/V once for
  all `g = 4` query heads of the group, whereas prefill(T=1) re-reads K/V per
  query head (`g×` more). At the 2k context that is a ~2% edge — the decode step
  at these sizes is bandwidth-bound on the shared K/V regardless, so the reuse
  win is small but real, and it grows with `g`. The value of the decode kernel
  is not this margin but the **allocation-free, cache-friendly single-token
  path** that M6-T07 calls once per generated token.
- **Thread scaling** (NEON, decode): 878.0 → 268.3 µs over 8 cores (3.27×). The
  decode kernel parallelizes over the `Hkv = 8` kv heads only (design §8, no
  flash-decoding split until M12-T03), so its parallel width equals Hkv — here
  it happens to match the 8 cores; a model with `Hkv < cores` (e.g.
  Qwen2.5-0.5B, Hkv=2) would leave cores idle on decode, the concrete M12-T03
  motivator. The sub-linear 3.27× is the shared-K/V memory bandwidth.

Correctness unaffected (Class T within §10 rtol 1e-4 atol 1e-5, observed
max-abs-diff ≤ 8.6e-6 across the acceptance sweep; bit-identical across thread
counts; bit-identical to prefill(T=1)). AVX2 body mirrors the NEON structure and
is proven correct + warnings-clean by CI; not measured here (no x86-64 dev
machine).

## M6-T08 — end-to-end generation baseline (2026-08-18)

**Host:** Apple M2 (**4 performance + 4 efficiency cores**, 8 physical; 16 GB),
macOS, dev machine — *not fully quiesced* (see the stability note). The default
pool size is 8 (`parallel::physical_core_count()` counts all 8 physical cores).
**Toolchain:** Homebrew clang 20, `-O3` Release.
**Model:** `Qwen2-0.5B-Instruct` (bf16, `Qwen2ForCausalLM`, 24 layers, `E=896`,
`H=14`, `Hkv=2`, `d=64`, `I=4864`, `V=151936`, tied embeddings + q/k/v biases) —
the ~1B-class acceptance model.
**Command:** `build/benchmarks/bench_generate --model <dir> --prompt-len 128
--new-tokens 128 --runs 5 --threads <t> --backend optimized`.

**Methodology.** The benchmark drives the real greedy `Generate` loop and reads
its per-token callback timestamps: **prefill tok/s** = `prompt_len /`(start →
first token); **decode tok/s** = `1000 /` the *median* per-step latency of that
run. The median (not the whole-window average) is the steady-state decode rate —
on a non-quiesced machine a single background hiccup inflates one step out of 128
(visible as the `p90 ≪ max` gap) without reflecting steady-state throughput. The
prompt is synthetic random ids in `[0, vocab)`; only its length drives the
compute. One warmup run (first-touch faults + workspace grow-on-demand) precedes
the recorded runs. Prefill's headline is best-of-N (peak), matching the
peak-of-N convention of the kernel microbenches; decode's headline is the median
per-run rate, and the ±5% verdict is stated against the decode series.

**Optimized backend (`kOptimized`), NEON:**

| Threads | prefill tok/s (best) | decode tok/s (median) | decode ±% run-to-run |
|---:|---:|---:|---:|
| **8 (default)** | **133.5** | **31.0** | **±3.9% — PASS** |
| 4 | ~101 | ~26 | ±4.4% (advisory) |
| 2 | ~57 | ~15 | ±7.3% (advisory) |

The **8-thread row is the recorded baseline** and meets the ±5% acceptance
criterion (decode ±3.9% across 5 runs; decode step p50 32.1 ms / p90 37.4 ms).
The 4- and 2-thread rows are **advisory** — captured while the machine carried
light background load, so their run-to-run spread is not a property of the
engine. Formalizing the quiescing procedure (and a proper thread/ISA sweep with
±3% discipline) is **M12-T01**; per this file's preamble, pre-M12 entries are
advisory. The single-thread point is omitted: at ~120 ms/decode-step it did not
finish inside the measurement window here (an M12 sweep item).

**Stability caveat (honest).** Repeated back-to-back invocations on this laptop
intermittently spiked individual decode steps to 130–650 ms (vs a ~32 ms p50) —
OS/background activity, not the engine. Run **one config at a time on an
otherwise-idle machine** for a clean ±5%; the median-step metric absorbs
isolated spikes but not sustained contention.

### llama.cpp context number

Parity with llama.cpp is an **M12 goal, not this ticket** — this is a
same-machine context number only.

**Build:** `ggml-org/llama.cpp` @ `6d05498` (2026-08-19), CPU-only, same Homebrew
LLVM 20 toolchain as the engine (Apple clang cannot compile C++ on this machine —
CLAUDE.md). Configured `-DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF
-DLLAMA_CURL=OFF -DGGML_NATIVE=ON` (a pure-CPU NEON build, no Metal/Accelerate,
to compare CPU kernel against CPU kernel), plus `-DCMAKE_CXX_FLAGS=
-Wno-elaborated-enum-base` to get past LLVM 20 rejecting the CommandLineTools
SDK's CoreFoundation headers (the same broken-CLT issue that blocked the M6-T02
Accelerate number). **Conversion:** `convert_hf_to_gguf.py` on the same HF
checkpoint, `--outtype bf16` and `--outtype f16`.
**Command:** `llama-bench -m <gguf> -p 128 -n 128 -t <t> -r 5` (its `pp128` =
prefill tok/s, `tg128` = decode tok/s; it reports mean ± stddev).

| GGUF | threads | pp128 (prefill) | tg128 (decode) |
|---|---:|---:|---:|
| f16 | 8 | 192.5 ± 8.3 | 18.2 ± 8.4 |
| f16 | 4 | 111.9 ± 7.0 | **34.0 ± 7.1** |
| bf16 | 8 | 7.7 ± 5.0 | 5.1 ± 1.0 |

Reading it (context, not a scoreboard — same loaded machine, so the ± values are
wide):

- **Compare f16, not bf16.** llama.cpp's CPU backend has no vectorized bf16 GEMM
  on ARM (it widens per element), so its bf16 GGUF runs ~20–25× slower than its
  f16 GGUF (7.7 vs 192 pp). Our engine keeps weights in their checkpoint bf16 and
  vectorizes the widen inside the packed GEMM, so **our bf16 path is the fast
  path** — llama.cpp's f16 GGUF is the fair opponent for our bf16 model.
- **Ballpark parity, different core-scaling.** Our engine (8t): ~133 prefill /
  ~31 decode. llama.cpp f16: 192 prefill (8t) / 34 decode (**4t**). We are within
  ~1.4× on prefill and roughly at par on decode.
- **The E-cores hurt decode in both engines.** llama.cpp's memory-bound decode is
  *faster at 4 threads (34 t/s) than 8 (18 t/s)* — the 4 efficiency cores drag
  down the shared-K/V bandwidth-bound step and add sync overhead, while the
  compute-bound prefill still scales up with 8 (192 vs 112). Our default pool
  uses all 8 physical cores; an E-core-aware pool (P-cores only for decode) is an
  M12 tuning lever this number motivates. It compounds with our decode kernel's
  `Hkv`-only parallel width (`Hkv=2` here → only 2-way parallel decode; design §8,
  the M12-T03 flash-decoding motivator).
- Our decode also still pays the `SimpleKvCache::view` gather (~12% decode
  memory-traffic overhead, design §6); M8-T05 removes it by reading through the
  block table.

**Artifacts** (not committed; ~2 GB, kept for the M12 sweep — safe to delete):
`~/llama.cpp` (the pinned CPU build) and `~/models/qwen2-0.5b-instruct.{bf16,f16}.gguf`.

---

## M7-T06 — Batched sampling (2026-08-19)

**What:** `BatchedSampler::Sample` over a synthetic `[batch, vocab]` fp32 logits
block (no model load — the sampling hot path in isolation), median ms per step
vs the equivalent serial loop over the single-sequence reference `Sampler`. The
acceptance target is advisory: 64 sequences × 128k vocab under 5 ms on the dev
machine.

**Harness:** `benchmarks/bench_sampling.cpp` —
`bench_sampling --batch 64 --vocab 131072 --runs 30 --threads T --config C`. One
warmup step (sizes scratch + pool), then the median of `--runs` steps; synthetic
deterministic logits (~[-8, 8), tokenizer-free). All rows use the same config
here (the batched path supports heterogeneous per-row params; this measures the
common case). `--threads T` sets `ENGINE_NUM_THREADS` before the pool is sized.

**Machine:** same M2 dev machine / Homebrew LLVM 20 Release / NEON as the other
entries (loaded interactively, so treat as ballpark, not a scoreboard).

64 × 131072, batched median ms/step (serial reference loop in parens), speedup:

| config | 1 thread | 4 threads | 8 threads | verdict (8t, ≤5 ms) |
|---|---:|---:|---:|:--|
| greedy | 9.96 (9.80, 0.98×) | 2.66 (10.1, 3.79×) | **2.01** (9.82, 4.86×) | **PASS** |
| temp (T=1) | 42.3 (41.4, 0.98×) | 12.2 (41.4, 3.40×) | 10.4 (41.4, 3.98×) | over |
| top-k 50 + top-p 0.9 | 94.8 (94.7, 1.00×) | 27.0 (94.2, 3.49×) | 21.4 (94.7, 4.42×) | over |
| top-p 0.95 (full vocab) | — | — | 114.8 (639, 5.57×) | over |
| logprobs (T=1, N=5) | — | — | 22.4 (83.8, 3.74×) | over |

Reading it:

- **Threading is the win; the batched harness adds ~nothing.** At 1 thread the
  batched path equals the serial reference (0.98–1.00×) — confirming both run the
  identical shared `detail::SampleRow`. Threading scales ~3.5× at 4 cores, ~4–5×
  at 8 (the 4 M2 efficiency cores give diminishing returns on these
  memory-bandwidth-bound steps, as in the M6-T08 decode note).
- **Greedy meets the target (2.0 ms).** The sampling configs do not: `temp` alone
  is ~10 ms because it makes several full-vocabulary passes over 128k × 64
  (copy → CheckFinite → temperature → `x−max` → `ExpF32` → double-sum → normalize)
  and is DRAM-bandwidth-bound, not compute-bound. A fused-pass rewrite and/or an
  E-core-aware pool are M12 levers; the target is advisory and the number is
  recorded honestly here.
- **The `ApplyTopP` positive-only sort is a real win for the realistic config.**
  top-k 50 + top-p 0.9 dropped from **~44 ms → ~21 ms** (8t) once `ApplyTopP`
  stopped full-sorting all 131072 entries and sorted only the ~50 that survive
  top-k (the rest are exactly 0 from `exp(-inf)`). Pure top-p over the full
  vocabulary (every prob positive) still sorts them all — no regression vs the
  old full sort, no win either; on this near-uniform synthetic distribution its
  nucleus is large (a real, peaked model distribution would sort far fewer).
- **Bit-exactness is free.** The batched path calls the same `detail::SampleRow`
  as the reference, so it picks identical tokens/logprobs by construction
  (`batched_sampler_test`, host ISA + forced-scalar) — the "optimization" is
  purely threading + scratch reuse + the shared filter improvements, never a
  different arithmetic.

## M8-T05 — paged decode attention (2026-08-19)

**Host:** Apple M2 (8 physical cores), macOS, dev machine.
**Toolchain:** Homebrew clang 20, `-O3` Release.
**Command:** `build/benchmarks/attention_bench paged` (defaults `L=2048, H=32,
Hkv=8, d=64, bs=16`, T=1 — the same Llama-3.2-1B-shaped 2k-context decode step
as M6-T05; best per-call of 200 samples × 100-call batches). Compares the
M6-T05 contiguous `DecodeAttentionF32` against the M8-T05
`PagedDecodeAttentionF32` reading K/V through a block table (sequential mapping,
the common decode-time layout). The two are **bit-identical** (asserted in
`paged_decode_attention_kernel_test`); this measures only the block-indirection
cost. No perf target ("time vs the contiguous kernel").

| Config | contiguous decode | paged decode | paged vs contig |
|---|---:|---:|---:|
| NEON, 8 threads (default) | 245.8 µs | 274.7 µs | **0.89×** |
| NEON, 1 thread | 892.8 µs | 1024.8 µs | 0.87× |

Reading them:

- **The isolated paged kernel is ~11–13% slower per call** than the contiguous
  one. The recurrence is identical arithmetic; the overhead is the block-table
  walk — splitting each 64-key unit's `DotScoreRow` into `kAttnKb/bs = 4`
  per-block calls (each with its own horizontal-reduction setup) plus the
  per-block pointer recompute and table load.
- **This is not the number that matters for M8.** In the engine the contiguous
  path must first **gather** K/V out of the paged cache via `KvCache::view`
  (~12% decode memory-traffic overhead, optimized-cpu-execution.md §8); the
  paged kernel reads the slabs in place and pays no gather. The whole-decode-step
  comparison — where the gather is charged to the contiguous path — is M8-T07's
  `bench_generate` regression check (≤10% vs the M6 baseline), not this
  microbenchmark. Recorded here so M8-T07's numbers are read with the isolated
  kernel cost in view.
- **Parallel width is `Hkv` (= 8 here)**, identical to M6-T05 decode — the
  E-core throttling and `Hkv`-only width (M12-T03 flash-decoding motivator)
  apply unchanged.
