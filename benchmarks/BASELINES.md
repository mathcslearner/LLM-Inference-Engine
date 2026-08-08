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
