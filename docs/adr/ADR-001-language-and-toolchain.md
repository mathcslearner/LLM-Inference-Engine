# ADR-001: Language & toolchain — C++20 host code, CUDA kernels

## Status

Accepted (2026-08-03). The CUDA half is superseded by
[ADR-004](ADR-004-cpu-first-pivot.md) (2026-08-07): the engine is CPU-first —
C++20 with SIMD kernels; there is no CUDA in the build. The C++20/CMake/
toolchain decisions below remain in force.

## Context

The project is a production-grade inference engine for decoder-only transformer
models, targeting NVIDIA GPUs on Linux x86-64, with performance goals in the
class of existing engines (vLLM, TensorRT-LLM). That workload dictates the
technical constraints:

- **Custom CUDA kernels are on the critical path.** Paged attention, fused
  norms/activations, quantized GEMM epilogues, and sampling kernels must be
  written and tuned by hand; the language must interoperate with the CUDA
  toolchain, cuBLAS, and NCCL with zero friction.
- **The host side is latency-sensitive.** Continuous batching gives the CPU a
  per-step budget of a few hundred microseconds for scheduling, batch assembly,
  and launch; GC pauses or interpreter overhead on this path are unacceptable.
- **Explicit memory control is the product.** The paged KV cache, caching
  allocators, and pinned-memory staging are the engine's core mechanics, not
  incidental details.
- The project is also explicitly a clean-architecture exercise: one language
  for the whole engine, mechanically enforced style, and thorough testing.

## Decision

We will write the engine in **C++20** for all host code and **CUDA C++** for
GPU kernels. **Python appears only in `tools/`** (HuggingFace golden-fixture
generation, load testing) and is never part of the engine, its runtime, or its
build.

Toolchain:

- **Build system:** CMake ≥ 3.26, dependencies pinned via `FetchContent`
  (policy in [`docs/dependencies.md`](../dependencies.md)).
- **Compilers:** GCC 12+ and Clang 16+ on Linux x86-64 (both kept green in CI);
  CUDA toolkit 12.x from M2 on. Warnings-as-errors for project code.
- **Style tooling:** clang-format and clang-tidy pinned to upstream LLVM 20
  (`scripts/clang-tools.sh`), so formatting is deterministic across machines.
- **macOS is a CPU-only development convenience** (`-DENGINE_ENABLE_CUDA=OFF`,
  Homebrew LLVM 20), never a deployment target; Linux + CUDA is the product.

### Alternatives considered

- **Rust.** Genuine advantages: memory safety by construction, modern tooling
  (cargo, rustfmt, clippy) that C++ approximates only with effort. Rejected
  because the GPU story is the project: CUDA kernel development in Rust means
  FFI wrappers around every kernel launch, immature bindings for cuBLAS/NCCL
  and CUDA graphs, and losing first-class tooling (Nsight, `cuda-gdb`,
  `compute-sanitizer`) at the layer where we will spend most of our time. The
  safety win also lands mostly outside the hot core, which is dominated by raw
  device memory and pointer arithmetic that Rust would wrap in `unsafe` anyway.
- **Python orchestration + C++/CUDA core (the vLLM shape).** Advantages: fast
  iteration on scheduling/serving logic, effortless HuggingFace integration.
  Rejected: the two-language boundary is a permanent tax on refactoring,
  testing, and packaging; the Python-side scheduler eventually becomes the
  bottleneck (vLLM itself has been migrating hot paths to native code); and a
  from-scratch engine gets no leverage from Python's ML ecosystem at runtime —
  we need it only for offline fixture generation, which `tools/` covers.
- **Triton (or similar DSLs) for kernels.** Attractive for fused elementwise
  and norm kernels, but it embeds a Python dependency in the build, and the
  hardest kernels (paged attention, quantized GEMM) still end up hand-written.
  Deliberately writing all kernels in CUDA also serves the project's
  pedagogical goal. May be revisited per-kernel later; would need a new ADR.

## Consequences

- We own memory safety. Mitigations are structural: warnings-as-errors,
  `CHECK`/`DCHECK` invariants (ADR-003), ASan/UBSan in CI once M1 lands real
  data structures, and the CPU-reference testing ladder from `CLAUDE.md`.
- C++20 (not 23) caps the feature set: no `std::expected` (ADR-003 builds
  `StatusOr<T>` instead), no `std::print` (we pin fmt). In exchange, both GCC
  12 and Clang 16 support everything we use today.
- Two compilers in CI is a standing cost that buys standard-conformance
  discipline and catches nvcc-adjacent breakage early.
- Contributors need CMake + a modern compiler only; no Python environment is
  required to build or test the engine (CPU CI proves this on every push).
