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
| 1 · compute substrate | `memory`, `tensor`, `cuda`, `kernels`, `cpu` | `tensor → memory`; `tensor → cuda` (Amendment 3); `memory → cuda` (Amendment 2, was `cuda → memory`); `kernels → cuda, tensor`; `cpu → tensor`; `memory → tensor_base` (Amendment 1) |
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
   CUDA allocators), `tensor` (its transfer TU, Amendment 3), and
   `distributed` (NCCL) may touch the CUDA toolkit, and only behind
   `ENGINE_ENABLE_CUDA`. Every module — including those five, with GPU paths
   compiled out — builds and passes its CPU tests with
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

## Amendments

### Amendment 1 (2026-08-03, with M1-T01): `memory → tensor_base`

M1 surfaced a knot in layer 1: `Tensor` holds a `Buffer` (so `tensor` links
`memory`, the listed edge), but `Buffer` records the `Device` it lives on,
and `Device` — with the other plain value types `DataType` and `Shape` —
lives in `src/tensor/` per the roadmap. To keep the graph acyclic without
moving files, the tensor module builds two targets:

- `engine::tensor_base` — INTERFACE (header-only): `dtype.h`, `shape.h`,
  `device.h`, `half.h`. Depends only on `core`; these headers include
  nothing from `memory` or the rest of `tensor`.
- `engine::tensor` — the compiled library; links `engine::tensor_base`,
  `engine::memory`, `engine::core`.

New allowed edge: **`memory → tensor_base`** (and `tensor_base` may be
linked by any module that today may link `tensor`). `memory` still must not
include `tensor.h`/`ops.h` — only the base value-type headers. No link or
include cycle exists. Rationale and details: `docs/design/tensor.md` §2.1.

### Amendment 2 (2026-08-04, with M2-T05): `memory → cuda`, replacing `cuda → memory`

Implementing `CudaAllocator` (in `memory`, per the roadmap's file layout)
surfaced that the intra-layer edge points the wrong way. The allocator needs
three things from `cuda`: `device_count()` to validate device indices
(cuda-backend design §5.2), `ScopedSetDevice` to allocate on the right
device, and the `ToStatus` error mapping so device OOM becomes
`kResourceExhausted` from exactly one code table (§4.2). Duplicating those
inside `memory` — the only alternative that keeps the old edge — would fork
the error-mapping table the design deliberately centralizes.

The reverse edge was speculative: `cuda` links nothing from `memory` and
its planned consumers of buffers (cuBLAS workspaces, M5/M11) can receive
memory from `engine`, which links both modules. So the edge **flips**
rather than growing a cycle:

- Removed: `cuda → memory`.
- Added: **`memory → cuda`**, PRIVATE in CMake — `memory`'s public headers
  stay toolkit-free (cuda-backend design §2.2); only its CUDA-build `.cpp`
  files include `cuda` headers.

`cuda` keeps its `core` link. (*Correction 2026-08-04:* this amendment
originally also said `cuda` keeps a `tensor_base` link — reachable per
Amendment 1 — but nothing in `src/cuda/` uses tensor types and that link
never existed in CMake; it remains allowed, just unused.) No link or
include cycle exists. Rationale and details:
`docs/design/cuda-backend.md` §2.1.

### Amendment 3 (2026-08-04, with M2-T07): `tensor → cuda`

The cuda-backend design (§2.1) originally planned `tensor`'s host↔device
transfer TU to call only raw toolkit functions (`cudaMemcpyAsync`) — toolkit
access without a module edge, so nothing above `tensor` would depend on the
`cuda` module. Implementation proved that insufficient, for the same reason
Amendment 2 was needed: the transfer contract depends on cuda-module *state
and code*, not just the toolkit.

- **The per-device engine default stream.** A null `StreamHandle` means "the
  engine default stream" (design §6.3), and the stream-less `Tensor::to` is
  defined to enqueue there and synchronize (§8.3). That stream is
  process-wide state owned by `cuda` (`DefaultStream`). A tensor-local
  substitute stream would fork stream identity: work enqueued through
  `tensor` would no longer be FIFO-ordered with everything else on the
  engine default stream — a silent-correctness hazard, strictly worse than
  the forked error table Amendment 2 rejected. (The CUDA legacy default
  stream is banned outright, design §6.1.)
- **The error mapping.** `cudaMemcpyAsync`/`cudaStreamSynchronize` failures
  must map through the one §4.2 code table (`ToStatus`,
  `CUDA_RETURN_IF_ERROR`), whose definitions live in `engine_cuda`.

So the edge is added rather than worked around:

- Added: **`tensor → cuda`**, PRIVATE in CMake — `tensor`'s public headers
  stay toolkit-free (they include only the toolkit-free
  `cuda/stream_handle.h`, which any module that may link `tensor` may
  already reach); only the CUDA-build `transfer.cpp` includes `cuda`'s
  toolkit-touching internals. Rule 3's toolkit list gains `tensor` (its
  transfer TU).

Consequence: every module above `tensor` now (transitively) may link the
`cuda` module — in practice already true via `memory → cuda`, and harmless
because both edges are PRIVATE and header-hygiene keeps the toolkit out of
public surfaces. `cuda` links nothing from `tensor` (nor, in practice,
`tensor_base` — see the Amendment 2 correction), so no cycle exists. Rationale and details: `docs/design/cuda-backend.md` §2.1
(refined in M2-T07) and §8.
