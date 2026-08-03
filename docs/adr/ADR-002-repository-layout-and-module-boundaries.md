# ADR-002: Repository layout & module dependency rules

## Status

Accepted (2026-08-03)

## Context

The engine will grow to ~18 modules built over 16 milestones by many focused
sessions. Without explicit, mechanically enforced boundaries, an inference
engine's natural failure mode is a monolith where the scheduler reaches into
CUDA streams and the server knows about KV blocks — at which point nothing can
be tested in isolation. The roadmap's correctness methodology (CPU reference
validates GPU kernels) and CPU-only CI both *require* that hardware-independent
code never links CUDA. We need the layout and the dependency rules fixed before
any substantial code exists, so every later ticket lands inside them.

## Decision

### Top-level layout

| Path | Contents |
|---|---|
| `src/<module>/` | One directory per module (table below); each is a static library `engine_<module>` with alias `engine::<module>` |
| `tests/` | `unit/`, `integration/`, `fixtures/` (committed golden data), `common/` helpers |
| `tools/` | Python dev tooling (fixture generation, load testing) — never in the engine build |
| `benchmarks/` | Throughput/kernel benchmarks and `BASELINES.md` |
| `docs/` | `adr/` (decisions), `design/` (subsystem designs), `dependencies.md` |
| `cmake/` | Build-system modules (warning set, dependency pins) |
| `scripts/` | Developer scripts (format, tidy, toolchain resolution) |

### Module dependency diagram

Arrows mean "may depend on" and always point downward. **The graph is acyclic
by construction and must stay that way.**

```
server → runtime → scheduler ─┐
                   engine ────┼→ model / tokenizer / kvcache / sampling / quant / spec / distributed
                              └→ tensor / memory / cuda / kernels / cpu → core
```

Expanded into layers (each module may depend on any module in a *lower* layer,
plus the intra-layer edges listed explicitly):

| Layer | Modules | Additional intra-layer edges allowed |
|---|---|---|
| 5 · serving | `server` | — |
| 4 · orchestration | `runtime` | — |
| 3 · execution | `engine`, `scheduler` | — (`engine` and `scheduler` never depend on each other; `runtime` mediates) |
| 2 · domain | `model`, `tokenizer`, `kvcache`, `sampling`, `quant`, `spec`, `distributed` | none today; any future edge (e.g. `spec → model`) requires amending this ADR |
| 1 · compute substrate | `memory`, `tensor`, `cuda`, `kernels`, `cpu` | `tensor → memory`; `cuda → memory`; `kernels → cuda, tensor`; `cpu → tensor` |
| 0 · foundation | `core` | — (depends on nothing but pinned third-party libs) |

Cross-cutting exception: **`metrics`** may depend only on `core`, and any module
may depend on `metrics` — instrumentation must never create a layering reason to
skip it.

### Rules

1. **No cycles**, including transitively. CMake enforces this: static-library
   link cycles fail at configure/link time.
2. **Dependencies are declared, not smuggled.** A module may include headers
   only from modules it links via `target_link_libraries(… engine::<module>)`.
   Includes are rooted at `src/` (`#include "core/status.h"`), so every include
   names its module and boundary violations are greppable.
3. **CPU-only code never links CUDA.** Only `cuda`, `kernels`, `memory` (its
   CUDA allocators), and `distributed` (NCCL) may touch the CUDA toolkit, and
   only behind `ENGINE_ENABLE_CUDA`. Every module — including those four, with
   GPU paths compiled out — builds and passes its CPU tests with
   `-DENGINE_ENABLE_CUDA=OFF`, which is what CPU-only CI runs.
4. **`scheduler` stays pure decision logic** (its module contract): it consumes
   descriptions of requests and budgets and returns decisions — it never
   touches tensors, streams, or the model. That is why it sits beside `engine`,
   not above the compute layer edge `engine` has.
5. **`tests/`, `benchmarks/`, and `main`** may depend on any module; no module
   depends on them. `tools/` is outside the dependency graph entirely.
6. **Amendments go through this ADR.** A ticket that needs a forbidden edge is
   a design smell first and an ADR amendment second — never a silent
   `target_link_libraries` addition. (Adding a *listed* edge when a milestone
   first needs it — e.g. `tensor → memory` in M1 — is expected and requires no
   amendment.)

### Alternatives considered

- **Single library, directory conventions only.** Less CMake boilerplate, and
  refactors don't move link lines. Rejected: convention without enforcement
  decays; per-module static libraries make every illegal dependency a build
  error and give tests small link targets.
- **Header-only include discipline via tooling (include-what-you-use,
  clang-tidy checks) instead of link boundaries.** Kept as a possible future
  *addition*, but tooling-only enforcement is advisory; the linker is not.
- **Coarser modules** (e.g. one `gpu` module swallowing `cuda` + `kernels` +
  `memory`). Rejected because the correctness ladder needs `kernels` separable
  from its consumers, and the KV cache / scheduler split is precisely what
  keeps scheduling testable without a GPU.

## Consequences

- Every module is unit-testable against only its lower layers; scheduler and
  sampling logic stay testable on CPU-only CI forever.
- ~18 small CMakeLists and some link-line ceremony; adding a module means
  touching `src/CMakeLists.txt` and creating the boilerplate. Accepted cost.
- Some future features will genuinely want new edges (speculative decoding
  proposing through a draft model, quantized weights inside model loading).
  Forcing those through ADR amendments is deliberate friction: it keeps the
  diagram true instead of aspirational.
- The diagram in `CLAUDE.md` mirrors this ADR for day-to-day reference; this
  ADR is the authority if they ever diverge.
