# ADR-004: CPU-first pivot

## Status

Accepted (2026-08-07)

## Context

The project began (ADR-001, v1 roadmap) as a multi-GPU CUDA inference engine
targeting NVIDIA hardware on Linux x86-64. Two facts collided with that plan:

1. **The development environment has no NVIDIA hardware.** The primary dev
   machine is an Apple Silicon Mac (arm64, no CUDA toolkit); CI runs CPU-only
   GitHub-hosted runners. Through all of M2 — the CUDA backend foundation —
   no CUDA compiler ever compiled the device-side code and no GPU ever ran a
   test.
2. **The post-M2 audit (2026-08-07) demonstrated what that costs.** A
   compile-breaking access-control bug (`stream.cpp` calling a private
   constructor from a free function) survived five "done" tickets, because
   the one tool that catches it in a second — any CUDA compiler — was never
   in the loop. Every acceptance criterion involving GPU execution was
   validated by inspection only. The milestones ahead (attention kernels,
   CUDA graphs, NCCL) would have compounded this: thousands of lines of
   unverifiable code, with the correctness ladder (CPU reference validates
   GPU kernels) unable to actually run its top rungs.

The project's goals are unchanged: a production-grade inference engine with
clean architecture and demonstrable, *measured* performance — an engine that
verifiably runs. "Keep writing CUDA blind" preserves the letter of the plan
while forfeiting the goal; doing nothing was not an option.

## Decision

We pivot to a **CPU-first engine**: hand-vectorized SIMD kernels (NEON on
arm64, AVX2 on x86-64, runtime-dispatched over an always-present scalar
reference), thread-pool parallelism, and a performance target of
llama.cpp-class CPU inference. The feature set is otherwise unchanged —
paged KV cache, continuous batching, prefix caching, quantization
(deepened: we implement RTN/AWQ/GPTQ ourselves, M14), speculative decoding
(deepened: a measurement study, M15-T07), OpenAI-compatible serving,
metrics.

Rules and scope:

- **Platform matrix:** macOS arm64 is the primary dev platform; Linux
  x86-64 is CI and the deployment target. Both are first-class: every
  kernel ships scalar + NEON + AVX2 variants, and the test suite runs the
  host's best ISA plus a forced-scalar pass, so the two platforms jointly
  cover every backend.
- **The correctness ladder survives, re-rooted:** HuggingFace fixtures
  validate the scalar reference; the scalar reference validates every
  vectorized/optimized kernel. Nothing is "validated by inspection" —
  every line of kernel code is executable on the machines we have. This is
  the property whose absence forced this ADR.
- **Retirement, not erasure (M3-T02):** `src/cuda/`, the CUDA allocators
  and transfer path, the CUDA kernel infrastructure, and the planned
  `distributed/` module are removed from the tree (ADR-002 Amendment 4).
  Kept: the device-agnostic caching pool allocator (the future KV block
  pool), the tensor library, and `Device` with its reserved `kCUDA` value
  — so a future GPU backend is additive, not an API break. History stays
  in git and in `docs/design/retired/`; living docs (README and the active
  design docs) describe only the CPU-first engine.
- **The project follows a v2 CPU-first plan** (the pre-pivot v1 plan is
  retired).

### Alternatives considered

- **Rent cloud GPUs at milestone boundaries.** Genuinely viable and cheap
  (~$0.30–0.70/hr for a 4090/A10; likely under $100 for the whole project),
  and it preserves the CUDA skill-building goal. It lost on cadence, not
  cost: the roadmap's kernel-heavy middle (M5, M11–M14 in v1) needs a tight
  edit-compile-run loop, not an afternoon of validation per multi-week
  milestone — the M2 audit showed how much silently rots between GPU
  sessions. This remains the recommended path if the project ever revives
  the CUDA backend ("future directions").
- **Metal/MLX backend.** Uses the dev machine's actual GPU with a local
  loop, and the backend seams would admit it cleanly. It lost because the
  market/ecosystem signal is niche next to either CUDA or CPU inference,
  and it would trade the CUDA problem (can't run it) for a portability
  problem (CI can't run Metal). Kept as a future direction — the pivot's
  reference/optimized backend split is designed so a third backend slots
  in.
- **Continue writing CUDA blind.** Zero immediate cost. Rejected outright:
  the audit measured the compounding risk, and an engine whose core has
  never executed cannot demonstrate anything — including the engineering
  judgment this project exists to demonstrate.
- **Abandon the project.** Rejected: M0–M2's host-side infrastructure
  (build/test/CI discipline, tensor library, allocators, error handling)
  transfers to the CPU-first plan nearly whole, and the architectural
  substance (scheduler, paged cache, serving) was never GPU-specific.

## Consequences

- Everything the engine claims, it can prove on hardware we own: benchmarks
  are honest, sanitizers cover the *entire* surface (no unsanitizable
  device code), and determinism is strengthened (fp32 + deterministic
  partitioning makes batch-invariance plausibly guaranteeable).
- We forfeit the CUDA-specific portfolio signal for now; the compensating
  depth is SIMD/threading engineering, the quantization-algorithms toolkit,
  and measured comparisons against llama.cpp on identical hardware.
- A 7B-class model at INT4 is comfortably servable in host RAM, but
  CPU decode throughput is memory-bandwidth-bound: absolute tokens/sec will
  not approach GPU figures, and the performance narrative must be framed
  against CPU peers, never GPUs.
- M2's audited work is retired weeks after completion — sunk cost accepted
  consciously; its transferable pieces (caching allocator, testing
  patterns, source-seam discipline) are inherited by M3, and the retired
  design remains the seed of any future GPU backend.
- Follow-up work created: M3 (excision + CPU substrate), the doc
  restructuring in M3-T01, and ADR-002 Amendment 4.
