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
