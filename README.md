# LLM Inference Engine

A production-grade, **CPU-first** inference engine for decoder-only
transformer models (Llama, Qwen, and similar families), written in C++20 with
hand-vectorized SIMD kernels — NEON on Apple Silicon, AVX2 on x86-64 — behind
runtime dispatch over an always-present scalar reference.

Feature scope, built incrementally along [ROADMAP.md](ROADMAP.md): paged KV
cache, continuous batching, prefix caching, custom SIMD kernels, chunked
prefill, weight-only quantization (INT8/INT4, AWQ/GPTQ — including our own
quantizer implementations and a perplexity evaluation harness), speculative
decoding, an OpenAI-compatible HTTP API with SSE streaming, and Prometheus
metrics. Inference only — no training or fine-tuning.

Design principles: clean module boundaries (ADR-002), `Status`-based error
handling (ADR-003), a strict correctness ladder — HuggingFace golden fixtures
validate the scalar reference, the scalar reference validates every
vectorized kernel — and measured performance claims only
(`benchmarks/BASELINES.md`).

## Building

Requires CMake ≥ 3.26 and a C++20 compiler (GCC 12+/Clang 16+).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Developer workflow, style enforcement, and per-ticket validation are
documented in [CLAUDE.md](CLAUDE.md); the test harness in
[tests/README.md](tests/README.md).

## Project history

The project began as a multi-GPU CUDA engine and pivoted to CPU-first after
Milestone 2 — the rationale is recorded in
[ADR-004](docs/adr/ADR-004-cpu-first-pivot.md), the original plan in
[docs/archive/ROADMAP-v1.md](docs/archive/ROADMAP-v1.md).
