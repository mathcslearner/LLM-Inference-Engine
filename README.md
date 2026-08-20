# LLM Inference Engine

A production-grade, **CPU-first** inference engine for decoder-only
transformer models (Llama, Qwen, and similar families), written in C++20 with
hand-vectorized SIMD kernels — NEON on Apple Silicon, AVX2 on x86-64 — behind
runtime dispatch over an always-present scalar reference. Inference only — no
training or fine-tuning.

The design prioritizes clean module boundaries, a strict correctness
methodology, and measured performance: on the dev machine it reaches rough
throughput parity with a same-machine `llama.cpp` build (see
`benchmarks/BASELINES.md`).

## What's implemented

- **Model loading** — `config.json` parsing, single-file and sharded
  safetensors checkpoints, HuggingFace weight-name mapping, tied embeddings,
  zero-copy weight registry (Llama and Qwen2 families).
- **Tokenizer** — byte-level BPE (`tokenizer.json`), byte-identical to
  HuggingFace on the committed vectors, with incremental (streaming)
  detokenization.
- **Two execution backends behind one interface** — an unoptimized scalar
  *reference* engine (the correctness oracle) and an *optimized* engine built
  on custom SIMD kernels with runtime ISA dispatch (scalar / NEON / AVX2).
- **Custom SIMD kernels** — packed-weight GEMM/GEMV, flash-style attention
  (prefill, single-token decode, paged, and ragged batched variants with an
  online softmax), RMSNorm, SwiGLU, softmax, RoPE, and embedding — each
  validated against the scalar reference within a stated numerical tolerance.
- **Paged KV cache** — a refcounted block pool and per-sequence block tables,
  with zero-copy paged attention on the decode path.
- **Continuous batching runtime** — an asynchronous engine thread, a
  continuous-batching scheduler (decode-first admission, preemption with
  evict-and-recompute), request cancellation, and per-request failure
  isolation.
- **Sampling** — temperature / top-k / top-p, repetition/presence/frequency
  penalties, stop tokens and stop strings, log-probabilities, and a batched
  sampler; deterministic under a fixed seed.

The engine loads and generates coherent text from a real ~0.5B checkpoint
(Qwen2-0.5B-Instruct) on both backends.

## Planned

An OpenAI-compatible HTTP API with SSE streaming, weight-only quantization
(INT8/INT4, AWQ/GPTQ, INT8 KV cache) with our own quantizer and evaluation
harness, prefix caching, chunked prefill, speculative decoding, and Prometheus
metrics. These are built incrementally on the foundation above.

## Design principles

- **Clean module boundaries** — enforced dependency layering
  ([ADR-002](docs/adr/ADR-002-repository-layout-and-module-boundaries.md)).
- **`Status`-based error handling** — no exceptions across module boundaries
  ([ADR-003](docs/adr/ADR-003-error-handling.md)).
- **A strict correctness ladder** — HuggingFace golden fixtures validate the
  scalar reference; the scalar reference validates every vectorized kernel.
  The suite runs the host's best ISA plus a forced-scalar pass, so CI (x86-64)
  and the dev machine (arm64) jointly cover every backend. Over 1,300 unit and
  integration tests.
- **Measured performance only** — no optimization claim without a recorded
  benchmark delta (`benchmarks/BASELINES.md`).

Each subsystem has a design doc under [`docs/design/`](docs/design); each
architectural decision an ADR under [`docs/adr/`](docs/adr).

## Building

Requires CMake ≥ 3.26 and a C++20 compiler (GCC 12+/Clang 16+).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

The test harness is described in [tests/README.md](tests/README.md).

## Project history

The project began as a multi-GPU CUDA engine and pivoted to CPU-first after
its second milestone — the rationale is recorded in
[ADR-004](docs/adr/ADR-004-cpu-first-pivot.md), and the retired GPU design in
[docs/design/retired/](docs/design/retired).
