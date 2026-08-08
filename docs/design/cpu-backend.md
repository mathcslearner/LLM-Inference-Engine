# CPU backend: threading, SIMD dispatch & kernel validation

**Milestone:** M3 (design doc: M3-T03; implementation: M3-T04 … M3-T06)
**Governs:** `src/parallel/`, `src/kernels/`, and the *role* of `src/cpu/`
(whose implementation lands in M5) — plus the kernel-validation methodology
every later kernel ticket (M5, M6, M11–M14) follows.
**Cites:** ADR-001 (language/toolchain), ADR-002 (module boundaries, through
Amendment 4), ADR-003 (error handling), ADR-004 (the CPU-first pivot),
`docs/design/tensor.md` (dtypes, half conversions, allocator).

This is the working contract for the CPU compute substrate. Implementation
tickets must conform to it; if implementation reveals a design flaw, this doc
is updated in the same change with a note on what changed and why
(`docs/design/README.md`).

---

## 1. Scope & non-goals

This document fixes the decisions every CPU kernel builds on: who owns
threads, how work is partitioned deterministically, how {scalar, NEON, AVX2}
variants are compiled and selected at runtime, what dtype rules kernels obey,
and how a kernel is *proven* correct — including on an ISA the CI runner
cannot execute. It exists because these are exactly the decisions that, made
ad-hoc per kernel, produce irreproducible results and untestable fast paths.

**In scope:** the `parallel` module (thread pool, `parallel_for`,
`parallel_reduce`), the `kernels` module's dispatch infrastructure and build
pattern, the dtype/accumulation policy, alignment and weight-layout
conventions, the validation methodology (oracle chain + platform matrix), and
the CI additions M3 makes.

**Non-goals (of this doc, not the project):**

- **No individual kernel designs.** GEMM blocking, attention layouts, and
  fusion plans belong to their milestones' design docs (M5-T01 model
  execution, M6+). This doc gives them the substrate and the rules.
- **No quantized-kernel design.** INT8/INT4 kernels (M13–M14) will follow the
  same dispatch and validation pattern; their numerics get their own doc.
- **No NUMA awareness, thread affinity, or work stealing** (v1; see §10).
- **No nested parallelism.** One level of parallelism, owned by `parallel`,
  CHECK-enforced (§3.4). Kernels never create threads.
- **No GPU anything** (ADR-004). The dispatch seam is ISA-shaped, not
  device-shaped; a future GPU backend enters at the `engine` backend seam,
  not here.

---

## 2. Module layout & layering

Three modules share layer 1's compute duties, with distinct roles:

| Module | Path | Role | Lands |
|---|---|---|---|
| `parallel` | `src/parallel/` | Thread pool + deterministic `parallel_for` / `parallel_reduce`. The **only** module that creates threads. | M3-T04 |
| `kernels` | `src/kernels/` | Production kernels: scalar + NEON + AVX2 variants behind runtime dispatch. The scalar variant is the always-available *fallback*, compiled on every platform. | M3-T05/T06, then M6+ |
| `cpu` | `src/cpu/` | The **reference backend**: a complete, unoptimized, clarity-first fp32 forward pass (GEMM, norms, RoPE, attention). The correctness oracle for everything in `kernels`. | M5 |

Positions in the ADR-002 layer diagram (Amendment 4 is the authority; restated
here per this ticket's acceptance criteria):

- `parallel` sits in **layer 1** (compute substrate), beside `memory`:
  **`parallel → core` only.** It knows nothing of tensors — it schedules
  index ranges, not data.
- `kernels`: **`kernels → tensor, parallel`** (plus `core` transitively).
- `cpu`: **`cpu → tensor`** today; M5's threaded reference GEMM will use the
  listed-edge pattern (`cpu → parallel` is an *addition ADR-002 must record*
  when M5-T02 needs it — flagged here so it is an amendment, not a smuggled
  link line).
- Consumers: `engine` may link all three; `tests/` and `benchmarks/` link
  anything.

Files this milestone creates:

| File | Module | Contents | Ticket |
|---|---|---|---|
| `src/parallel/thread_pool.h/.cpp` | parallel | `ThreadPool`, pool sizing | M3-T04 |
| `src/parallel/parallel_for.h/.cpp` | parallel | `parallel_for`, `parallel_reduce` | M3-T04 |
| `src/kernels/dispatch.h/.cpp` | kernels | `Isa`, feature detection, `SelectedIsa()`, `KernelTable` | M3-T05 |
| `src/kernels/elementwise.h/.cpp` | kernels | public entries + tables for add/mul/scale | M3-T06 |
| `src/kernels/reduce.h/.cpp` | kernels | public entries + tables for sum/max | M3-T06 |
| `src/kernels/convert.h/.cpp` | kernels | fp16/bf16 ↔ fp32 conversion entries | M3-T06 |
| `src/kernels/scalar/*.cpp` | kernels | scalar variants (all platforms) | M3-T06 |
| `src/kernels/neon/*.cpp` | kernels | NEON variants (arm64 builds only) | M3-T06 |
| `src/kernels/avx2/*.cpp` | kernels | AVX2 variants (x86-64 builds only) | M3-T06 |

### 2.1 Why both a `cpu` reference and scalar kernel variants

The two look redundant; they are not, and the distinction is load-bearing:

- **`kernels/scalar/` is a production fallback.** It is dispatched when no
  vector ISA is available or when `ENGINE_FORCE_ISA=scalar` is set, so it must
  be reasonably efficient (threaded, cache-aware where it matters). It is
  *subject to* validation, not the source of truth.
- **`src/cpu/` is the oracle.** Written for obviousness — textbook loop
  nests, no tricks — and validated against HuggingFace fixtures (M5). Every
  optimized kernel from M6 on is tested against it. An oracle you optimize
  is an oracle you no longer trust; `cpu` is deliberately never optimized.

Bootstrap ordering: M3-T06's primitives (elementwise, reductions,
conversions) predate the M5 reference, so for those ops **the scalar variant
doubles as the reference** — acceptable because they are simple enough to
validate directly against `std::` computations, `tensor/ops.h` semantics, and
the M1-T07 `half.h` goldens, and because §6.3 requires the vector variants to
match scalar *bit-exactly*, which is a stronger check than tolerance. From M5
onward, new kernels are validated against `cpu`, and `cpu` against fixtures.

---

## 3. Threading model (`src/parallel/`, M3-T04)

### 3.1 The pool

One persistent, fixed-size worker pool, created lazily on first use and sized
once:

```cpp
namespace engine::parallel {

class ThreadPool {
 public:
  // Spawns `num_threads` workers (CHECK: >= 1). Joins them in the destructor.
  explicit ThreadPool(int num_threads);
  ~ThreadPool();
  int num_threads() const;
  // parallel_for / parallel_reduce free functions take a pool; see below.
};

// The process-wide pool used by kernels: lazily constructed function-local
// static, sized by ENGINE_NUM_THREADS if set (numeric values clamp to >= 1;
// non-numeric is fatal at startup; empty counts as unset), else
// physical_core_count().
ThreadPool& DefaultPool();

// Physical (not logical) cores: macOS via sysctl("hw.physicalcpu"); Linux via
// sysfs topology (counting unique cores, so SMT siblings collapse); fallback
// std::thread::hardware_concurrency(). Never returns < 1.
int physical_core_count();

}  // namespace engine::parallel
```

**Physical cores, not logical.** The kernels this pool exists for (GEMM,
attention, norms) are memory-bandwidth- and FMA-port-bound; SMT siblings
contend for both and typically *hurt* on this workload class. Default =
physical cores; `ENGINE_NUM_THREADS` exists for experiments and for CI
determinism tests. (Apple Silicon P/E asymmetry is acknowledged and ignored
in v1 — the scheduler places threads; affinity is deferred, §10.)

`ThreadPool` is an ordinary class — tests construct and destroy private pools
(M3-T04's restart/shutdown-cleanliness criterion); production code uses
`DefaultPool()`. Workers park on a condition variable between regions; no
busy-spinning in v1 (revisit with benchmarks only, §10).

*M3-T04 implementation note:* this section originally left unparsable
`ENGINE_NUM_THREADS` values unspecified ("clamped to >= 1" covers only
numeric input). Implementation resolved it with the §4.3 stance: a
non-numeric value is fatal at startup with an actionable message — silently
ignoring developer/test configuration is the failure class ADR-004 exists to
prevent — while numeric values below 1 clamp to 1 and an empty value counts
as unset. The pool also serializes concurrently submitted regions internally
(one region in flight at a time), so multi-threaded callers are safe by
construction rather than by convention.

### 3.2 `parallel_for`: deterministic static partitioning

```cpp
// Runs body(begin, end) over disjoint sub-ranges covering [0, n).
// Blocks until every sub-range completed. body must not throw (§3.4) and
// must write only to locations disjoint from other sub-ranges' writes.
void parallel_for(ThreadPool& pool, std::int64_t n, std::int64_t grain,
                  FunctionRef<void(std::int64_t begin, std::int64_t end)> body);
```

(`FunctionRef` is a non-owning callable reference — implementation may start
as `const std::function&`; invocation cost is per *chunk*, not per element,
so this is not hot.)

Partitioning rules — the determinism foundation:

1. **Chunks depend only on `(n, grain)`.** `num_chunks = ceil(n / grain)`;
   chunk `c` covers `[c·grain, min((c+1)·grain, n))`. Thread count never
   enters the formula.
2. **`grain` is a per-call-site constant**, chosen by the kernel (a named
   constant, e.g. `kAddGrain = 4096`), never computed from pool size or
   runtime load. This is what makes chunk boundaries — and therefore §3.3's
   reduction partials — reproducible on any machine at any thread count.
3. **Assignment is static round-robin:** worker `w` of `T` executes chunks
   `w, w+T, w+2T, …`. No work stealing, no dynamic queues — scheduling is
   predictable and the implementation stays small enough to reason about
   under TSAN. (Load imbalance from skewed chunks is a known cost; §10.)
4. **Small ranges never touch the pool:** if `num_chunks <= 1`, `body(0, n)`
   runs inline on the caller. `grain` therefore doubles as the serial
   threshold. `n == 0` returns immediately.
5. The calling thread blocks (condition variable) until the region
   completes; it does not execute chunks in v1. One core briefly idle per
   region is accepted for implementation clarity (caller participation is a
   measured-perf follow-up, §10).

For `parallel_for` the disjoint-write rule already makes results independent
of scheduling; the partitioning rules exist so that `parallel_reduce` — and
any kernel whose *internal* blocking follows chunk boundaries — is bitwise
reproducible.

### 3.3 `parallel_reduce`: fixed combine tree

```cpp
// partial(c_begin, c_end) computes chunk c's partial; combine(a, b) folds two
// partials. Returns identity when n == 0.
template <typename T>
T parallel_reduce(ThreadPool& pool, std::int64_t n, std::int64_t grain,
                  T identity,
                  FunctionRef<T(std::int64_t, std::int64_t)> partial,
                  FunctionRef<T(T, T)> combine);
```

Mechanics, all mandatory:

1. Partials land in a `num_chunks`-sized array **indexed by chunk id** —
   which worker computed a partial is irrelevant to where it lands.
2. After the barrier, the *calling thread* folds the array in a **fixed
   pairwise (binary) tree over chunk indices**: repeated halving passes,
   `s[i] = combine(s[i], s[i + half])` for `i < half` (odd element carries),
   until one value remains. The fold order is a pure function of
   `num_chunks`, hence of `(n, grain)` — **bit-identical results at any
   thread count**, which is M3-T04's acceptance test at counts {1, 2, 8}
   over adversarial fp32 inputs (mixed magnitudes, e.g. `{1e30f, 1.0f,
   -1e30f, …}`, where fold order visibly changes the answer).
3. `combine` need not be associative in exact arithmetic (fp add is not);
   determinism comes from the fixed order, not from algebra. Pairwise
   folding also improves fp error growth over left-to-right — a free bonus,
   not a contract.
4. Partial slots are padded to a cache line (64 B) to avoid false sharing;
   padding is a performance detail with no effect on values.

### 3.4 Ownership, nesting, errors

- **Thread ownership:** `parallel` owns every thread it creates; nothing else
  in the engine creates threads for compute. (The `runtime` module's engine
  loop thread, M9+, is an orchestration thread, not a compute thread — it is
  the *caller* of parallel regions.) Kernels express parallelism exclusively
  through `parallel_for`/`parallel_reduce`.
- **No nested parallelism, CHECK-enforced:** a `thread_local bool
  inside_parallel_region` is set in workers while executing a chunk (and on
  the caller for the inline path); entering `parallel_for`/`parallel_reduce`
  with it set is a `CHECK` failure. Nesting is a programmer error under
  ADR-003 — it means a kernel called a kernel, which the layering already
  forbids.
- **No exceptions across the boundary** (ADR-003): bodies must not throw. An
  escaping exception is not caught or transported — it terminates. All
  recoverable-error checking (shape validation, allocation) happens *before*
  entering a region; inside a region there is only arithmetic and CHECK.
- **Pool lifecycle errors** (failed thread spawn) are `CHECK` — a process
  that cannot create threads at startup has no meaningful degraded mode.

*M3 audit implementation notes (2026-08-08):* three hardenings closing gaps
between this section's contract and the M3-T04 code. (1) "An escaping
exception terminates" is now *enforced*, not just documented: bodies (and
`parallel_reduce`'s `combine`) are invoked through `noexcept` frames on both
the inline and pooled paths, so the failure mode no longer depends on
`n` vs `grain` (previously the inline path propagated the exception to the
caller while the pooled path terminated in a worker). (2) The one internal
allocation inside a region entry point — `parallel_reduce`'s
cache-line-padded partial-slot vector — converts `std::bad_alloc` to `CHECK`
(same stance as the pool constructor's `std::system_error`), so ADR-003's
no-exceptions boundary holds even under allocation failure. (3)
`ThreadPool::Run` itself CHECKs the thread-local region flag (moved to
`thread_pool.h`'s `detail`): `Run` is a public seam, and calling it from
inside a region previously self-deadlocked silently on the pool's region
serialization.

### 3.5 OpenMP: considered and rejected

The honest pros: OpenMP is mature, its runtime schedulers are excellent,
`#pragma omp parallel for` would replace most of M3-T04, and both toolchains
in play support it. It is rejected on four grounds:

1. **Reduction ordering is unspecified.** Deterministic, thread-count-
   invariant reductions are a headline property of this engine (ADR-004
   consequences; CLAUDE.md determinism rule). `#pragma omp reduction` gives
   no ordering guarantee, so we would hand-build §3.3's chunk-partial +
   fixed-tree machinery *anyway* — inheriting OpenMP's runtime while using
   none of its reduction support.
2. **Runtime fragmentation.** Three build environments, three runtimes:
   Homebrew LLVM's `libomp` on macOS, GCC's `libgomp` and Clang 18's
   `libomp` in CI — each with its own env-var semantics, thread-count
   defaults, and spin-wait behavior. The pool this doc specifies is ~200
   lines and behaves identically everywhere.
3. **Ownership and enforcement.** §3.4's rules (no nesting, no exceptions,
   CHECK on misuse) are contracts we can enforce in our own code; with
   OpenMP they become conventions about pragmas.
4. **The threading layer is part of the deliverable.** Post-pivot (ADR-004),
   SIMD + threading engineering *is* the depth this project demonstrates;
   outsourcing the threading layer to a pragma undercuts that in exchange
   for saving a small, well-understood amount of code.

Decision: **rejected for kernels and engine code; not banned from
`benchmarks/` comparisons.** Revisiting requires amending this doc.

---

## 4. SIMD dispatch (`src/kernels/`, M3-T05)

### 4.1 ISA set & feature detection

```cpp
namespace engine::kernels {
enum class Isa : std::uint8_t { kScalar, kNeon, kAvx2 };

// The ISA every dispatched kernel will use, resolved exactly once:
// ENGINE_FORCE_ISA if set (fatal if unknown/unavailable, §4.3), else the
// best the host supports.
Isa SelectedIsa();
}
```

Detection is intentionally shallow:

- **arm64:** NEON (ASIMD) is architecturally mandatory in AArch64 — best ISA
  is `kNeon`, no probing.
- **x86-64:** `kAvx2` requires cpuid AVX2 **and** FMA **and** F16C (all
  Haswell-2013+; F16C is needed by the conversion kernels, §5), plus the
  OSXSAVE/`xgetbv` check that the OS preserves YMM state. Anything less →
  `kScalar`.
- Everything else → `kScalar`.

There is no per-kernel ISA choice and no mixed-ISA execution: one process,
one `SelectedIsa()`, memoized on first call. This keeps behavior explainable
and makes the forced-ISA test matrix (§8.1) mean what it says.

### 4.2 Kernel registry & dispatch

One function pointer per kernel, one table per kernel, resolved at first use:

```cpp
// dispatch.h
template <typename Fn>
struct KernelTable {
  Fn scalar;         // never null: every kernel ships a scalar variant
  Fn neon = nullptr; // null on non-arm64 builds or if not yet implemented
  Fn avx2 = nullptr; // null on non-x86-64 builds or if not yet implemented
};

// Returns the entry for SelectedIsa(), falling back to scalar when the
// selected ISA's slot is null (a kernel may gain its vector variants later
// than its scalar one; fallback is per-kernel, silent, and safe — the scalar
// variant is always correct).
template <typename Fn>
Fn Select(const KernelTable<Fn>& table);
```

Public entry points are ordinary functions with intrinsic-free signatures;
the resolved pointer is a function-local static, so dispatch cost after the
first call is one indirect call:

```cpp
// elementwise.h — public, includable everywhere; no intrinsics.
void AddF32(const float* a, const float* b, float* out, std::int64_t n);

// elementwise.cpp
void AddF32(const float* a, const float* b, float* out, std::int64_t n) {
  static const auto* fn = Select(kAddF32Table);
  fn(a, b, out, n);
}
```

Per-ISA variants live in per-ISA namespaces (`scalar::AddF32`,
`neon::AddF32`, `avx2::AddF32`), declared in an internal header whose
declarations are guarded by the same arch macros the build uses
(`ENGINE_ARCH_ARM64` / `ENGINE_ARCH_X86_64`, defined by `kernels`' CMake from
`CMAKE_SYSTEM_PROCESSOR`), so a table slot is `nullptr` exactly when the
variant's TU is absent from the build — declaration set and source list can
never disagree without a compile error.

**Threading is orthogonal to dispatch:** each public kernel decides — above a
per-kernel size threshold (its `grain`, §3.2) — to run its variant inside
`parallel_for`; below it, inline on the caller. The ISA variant is the loop
body either way.

*M3-T05 implementation notes:* three additions beyond the sketch above, all
testability-driven. (1) `Select` gained an explicit-ISA overload
`Select(table, isa)` — the public `Select(table)` delegates to it with
`SelectedIsa()` — so the §9 null-slot-fallback tests can exercise all three
selections on one host; it also CHECKs the never-null `scalar` slot. (2)
`SelectedIsa()`'s resolution is factored into `detail::BestHostIsa()` and
`detail::ResolveIsa(force_value, best_host)`, both exposed for tests: the
process-wide memoization (which `EXPECT_DEATH`'s fork inherits) makes the
§4.3 failure paths unreachable through the public entry, while the
forced-scalar ctest pass (§8.1) covers the real-environment happy path
end-to-end. (3) The §9 probe kernel landed as `kernels/probe.h` (public
`Isa DispatchProbe()`) with variants in `scalar|neon|avx2/probe.cpp` — the
first instantiation of the §4.4 layout; its vector variants compute their
answer with real intrinsics, so a per-TU flag misconfiguration fails to
compile rather than passing silently.

*M3 audit notes (2026-08-08):* two coverage holes in the silent-fallback
design were closed. (1) Because vector and scalar variants are bit-identical
by specification, no behavioral test can detect an *unwired* table slot —
an omitted `.neon`/`.avx2` designator would silently fall back to scalar and
every bit-compare-against-scalar sweep would pass vacuously (the ADR-004
"claims coverage it does not have" failure class). Each kernel module now
exposes `detail::<Kernel>Variant(Isa)` — the table entry `Select` would
return — and each suite pins the build's vector slots to the expected
per-ISA symbols by pointer identity. The recipe in §4.5 inherits this:
a new kernel's tests include the wiring assertion. (2) The §4.3 failure
modes are additionally exercised end-to-end through the real environment
variable by two ctest registrations (unknown value; host-unavailable ISA)
asserting the process dies with the actionable message at first dispatch —
previously only `detail::ResolveIsa` was death-tested in-process.

### 4.3 `ENGINE_FORCE_ISA`

Values: exactly `scalar`, `neon`, or `avx2` (case-sensitive). Read once,
inside `SelectedIsa()`'s memoized init. Two failure modes, both **fatal at
startup with an actionable message** (CHECK-style, naming the variable, the
bad value, and the host's available ISAs):

- unknown value (`ENGINE_FORCE_ISA=avx512`), and
- forcing an ISA the host cannot execute (`neon` on x86-64).

Fatal-not-Status is a deliberate ADR-003 stance, recorded here: the variable
is developer/test configuration resolved once at startup — a precondition,
not a runtime input. The alternative (fall back silently, or thread a
`Status` through every hot-path kernel signature) is strictly worse: a test
suite whose forced-scalar pass silently ran AVX2 would *claim* coverage it
does not have, which is the exact failure class ADR-004 exists to prevent.

### 4.4 Build pattern: per-ISA translation units

```
src/kernels/
  dispatch.h / dispatch.cpp        # Isa, detection, Select
  elementwise.h / elementwise.cpp  # public entries + tables
  internal/elementwise_impl.h      # per-ISA declarations, arch-guarded
  scalar/elementwise.cpp           # compiled everywhere, no arch flags
  neon/elementwise.cpp             # arm64 builds only; no extra flags (baseline)
  avx2/elementwise.cpp             # x86-64 builds only; per-TU flags
```

```cmake
# kernels/CMakeLists.txt (pattern)
if(ENGINE_ARCH_X86_64)
  target_sources(engine_kernels PRIVATE avx2/elementwise.cpp)
  set_source_files_properties(avx2/elementwise.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
```

Rules, all load-bearing:

1. **No target-wide or toolchain-wide arch flags, ever.** `-mavx2` appears
   only as a per-source property on `avx2/*.cpp`. Consequence: the compiler
   cannot emit an AVX2 instruction outside those TUs, so
   illegal-instruction-on-older-hardware bugs are structurally impossible —
   dispatch decides at runtime what actually executes, and everything
   outside `avx2/` runs on any x86-64.
2. **Intrinsic headers (`<immintrin.h>`, `<arm_neon.h>`) appear only inside
   per-ISA TUs.** Public headers stay intrinsic-free (grep-enforceable, the
   same hygiene rule the retired CUDA design applied to toolkit headers).
3. **Scalar TUs compile on every platform, always** — they are the fallback,
   the forced-scalar test target, and the portability floor.
4. Per-ISA TUs of ISAs the target cannot use are *absent from the build*
   (not compiled-and-unused): NEON TUs only on arm64, AVX2 TUs only on
   x86-64.
5. No function multiversioning (`target_clones`), no `#pragma` arch
   switching inside a TU — the TU boundary *is* the arch boundary.

### 4.5 Recipe: adding a kernel

The pattern every kernel ticket from M3-T06 onward follows (M3-T05's
acceptance criteria require it recorded here):

1. Declare the public entry in the family header (`kernels/foo.h`) —
   intrinsic-free signature, doc comment stating shape rules and the
   numerics class (§6.3) it belongs to.
2. Implement `scalar::Foo` in `scalar/foo.cpp`. Choose the `grain` constant
   and wire threading via `parallel_for`/`parallel_reduce` in the public
   entry (not inside variants — variants are single-threaded chunk bodies).
3. Add the `KernelTable`, the `Select`-based public entry, and the
   arch-guarded declarations in `internal/foo_impl.h`.
4. Land tests **in the same change**: correctness vs the oracle (`cpu`
   reference once M5 exists; direct reference computation before then),
   the §9 size/alignment sweep, explicit tolerances per §6.3, and
   registration in the forced-scalar pass (§8.1).
5. Add `neon::Foo` and `avx2::Foo` in their per-ISA TUs (per-TU flags per
   §4.4), with the bit-exactness or tolerance obligation §6.3 assigns to the
   kernel's class. Vector variants may land in a later ticket than scalar —
   the null-slot fallback covers the gap.
6. If the kernel is performance-motivated, add a microbenchmark
   (`benchmarks/kernels/`) and record before/after in
   `benchmarks/BASELINES.md` — no perf claim without a delta (CLAUDE.md).
7. Never touch global compile flags. If the kernel needs a new ISA feature
   (e.g. AVX-512), that is a §10 revisit, not a flag.

---

## 5. Dtype policy

- **fp32 accumulation, everywhere, always.** All arithmetic — GEMM
  accumulators, norm sums, softmax, reductions — is fp32 regardless of
  storage dtype. No fp64 accumulation (costs ~2× bandwidth/lanes for
  precision inference does not need; the fixed pairwise fold in §3.3 already
  bounds error growth better than naive left-to-right); no fp16 accumulation
  (unacceptable error for norms/softmax).
- **fp16/bf16 are storage formats.** Loads convert to fp32 at the kernel
  boundary; stores round once at the end. The *definition* of conversion is
  `tensor/half.h` (M1-T07): round-to-nearest-even narrowing, NaN preserved,
  bit-exact and constexpr (tensor design §3.1). Vectorized conversions
  (M3-T06) must match `half.h` **bit-exactly** on every input class,
  including subnormals and NaN payload truncation — this is testable against
  the committed half goldens, and it is why hardware paths were checked
  before being admitted: NEON `vcvt`/`fcvt` and x86 F16C `vcvtps2ph` both
  implement RNE fp32↔fp16 (F16C availability is folded into the `kAvx2`
  gate, §4.1); bf16 conversion is shift+RNE-rounding, implementable exactly
  on both ISAs with integer ops.

  *M3-T06 implementation note — hardware NaN quieting.* The "checked before
  being admitted" claim above holds for every numeric input class, but not
  for signaling NaNs: both conversion units force the quiet bit on every
  SNaN (verified empirically on the M2 dev machine: `FCVTL` widens fp16
  `0x7D01` to `0x7FE02000` where half.h's exact answer is `0x7EA02000`),
  while half.h preserves a surviving payload verbatim and forces the quiet
  bit only when the truncated payload would read as inf. The vector fp16
  variants therefore take the hardware result for non-NaN lanes and blend
  in NaN lanes rebuilt with the same integer ops half.h uses (detect via
  `abs > inf-pattern` compare, reconstruct sign|exp|payload, select) — a
  few extra ops per vector, bit-exactness preserved on all 2^16 patterns
  (tested exhaustively via the round-trip identity). The bf16 variants are
  pure integer arithmetic end-to-end and need no such carve-out.
- **Integer dtypes** (`kInt32`/`kInt64`) pass through kernels only as ids
  and indices in v1. `kInt4`/`kFP8E4M3` remain reserved-but-unimplemented
  until the quantization milestones (M13/M14), which extend this policy in
  their own design doc.

---

## 6. Kernel validation methodology

### 6.1 The oracle chain, re-rooted on CPU

ADR-004's central promise: nothing is validated by inspection. The v1 chain
(fixtures → CPU reference → GPU kernels) survives with the top rung replaced:

```
HuggingFace fixtures (tools/, committed under tests/fixtures/)
        │  tolerance-based comparison (HF's op order differs; §6.3 class T)
        ▼
src/cpu/ reference backend (M5: naive, clarity-first, fixture-validated)
        │  per-class comparison: bit-exact or stated tolerance (§6.3)
        ▼
src/kernels/ dispatched variants — scalar, NEON, AVX2 (M3-T06, M6+)
```

Pre-M5 bootstrap for the M3-T06 primitives: the scalar variant is the
reference (§2.1), itself checked against direct `std::`/`ops.h` computations
and the `half.h` goldens; NEON/AVX2 are then required to match scalar
bit-exactly, which is stronger than any tolerance.

### 6.2 The platform matrix — proving an ISA the CI runner lacks

The question this section must answer (ticket acceptance): **how is a kernel
proven correct on an ISA the CI runner cannot execute?** Answer: no single
machine covers all three backends, but the two machines we have jointly
execute every one of them, against the same test sources, with scalar as the
shared anchor:

| Host | Best-ISA pass | Forced-scalar pass | When it runs |
|---|---|---|---|
| macOS arm64 (dev machine) | NEON | scalar | every ticket — the mandatory per-ticket validation workflow (CLAUDE.md) |
| Linux x86-64 (CI runners) | AVX2 | scalar | every push / PR |

Three properties make the composition sound:

1. **Every variant executes on real hardware** before a ticket is done:
   NEON on the dev machine, AVX2 in CI, scalar on both.
2. **Scalar is the cross-platform anchor.** It runs on both hosts, and the
   inputs feeding it are bit-identical across platforms (seeded fills are
   bit-exact by construction, `tensor/ops.h`; `half.h` conversions are
   constexpr software implementations). So NEON-agrees-with-scalar (arm64)
   and AVX2-agrees-with-scalar (x86-64) compose into all-backends-agree,
   without any machine running both vector ISAs.
3. **Process closes the gap CI cannot.** CI never executes NEON; the merge
   gate is therefore *CI green + the dev-machine suite green*, and the
   per-ticket workflow (build + full ctest + format on every iteration) is
   what makes the second half real. This is recorded as a known asymmetry,
   revisited in §8.3.

### 6.3 Numerics classes & tolerance taxonomy

Every kernel is assigned a class in its doc comment; every numerical test
states its tolerance explicitly (CLAUDE.md rule — "within tolerance" with no
number is not a test):

- **Class E — elementwise & conversions** (add, mul, scale, fp16/bf16↔fp32):
  **bit-identical across ISAs and thread counts.** Per-element pure fp ops
  with one rounding each; IEEE 754 makes same-input-same-rounding exact, and
  chunking cannot reorder independent elements. Tests compare bit patterns,
  not tolerances.
- **Class R — reductions** (`sum`, `max`): **bit-identical across thread
  counts** (fixed chunk tree, §3.3) **and across ISAs**, via the fixed-width
  virtual-lane convention below.
- **Class T — everything else** (GEMM, norms, softmax, attention, M5+):
  scalar/reference-vs-optimized within a **stated per-test tolerance**
  (absolute + relative, or ULPs), because FMA contraction, blocking, and
  vector horizontal ops legitimately change rounding. Fixture comparisons
  (HF → `cpu`) are always Class T — HF computes in its own order.

**The fixed 16-lane accumulation convention (Class R, fp32 `sum`):** naive
vectorization breaks cross-ISA bit-identity because lane counts differ (NEON
4, AVX2 8) and accumulation order follows lane structure. Instead, the
*specification* of `SumF32` over a chunk is: 16 interleaved fp32
accumulators, lane `j` accumulating elements `i` with `i mod 16 == j`
(tail elements included, same rule), then a fixed halving fold
(`lane[j] += lane[j+8]`, then `+4`, `+2`, `+1`). Every ISA implements this
spec exactly: scalar with a 16-float array, NEON with 4×`float32x4_t`, AVX2
with 2×`__m256` (16 floats is the natural unroll on both — this costs
nothing). Plain adds only — no FMA in reductions. Per-chunk partials are
then bit-identical across ISAs, and §3.3's tree makes the total invariant to
thread count. `max` is order-insensitive for the totally-ordered values;
inputs are required NaN-free (documented precondition — inference produces
finite activations; debug builds may assert).

Class-R bit-identity is a *specified convention*, not an emergent property —
which is exactly why it is written here, where every implementation ticket
can be held to it.

*M3-T06 implementation notes (Class R, sharpening the convention where the
implementation found it underspecified — `src/kernels/reduce.h` carries the
authoritative statement):*

1. **Accumulator initialization is +0.0f, and that is part of the spec.**
   The convention above left the lanes' starting value implicit; it is
   observable (`SumF32` of `{-0.0f}` is `+0.0f`, since `+0 + -0 = +0`), so
   it is now specified. A useful consequence: with +0-initialized lanes an
   accumulator can never become `-0.0f`, so adding padded `+0.0f` elements
   is bit-safe — though the shipped variants use per-element tail adds into
   the spilled lane array rather than padding.
2. **The grain is part of the spec.** `kReduceGrain` (32768) fixes the
   chunk decomposition and hence the partials and fold-tree shape; changing
   it changes `SumF32` results at the last-ulp level. It is a named
   constant in `reduce.h` with this warning attached, not a tuning knob.
3. **`max` ±0 ties are specified as -0.0f < +0.0f.** "Order-insensitive
   for the totally-ordered values" was not quite enough: ±0 compare equal
   yet differ in bits, and the ISAs disagree on ties (ARM `FMAX` returns
   +0 for (±0), x86 `maxps` returns its second operand, `std::max` its
   first). The spec is now total-order max with `-0 < +0` — NEON native,
   scalar via an explicit signbit tie-break, AVX2 via a
   compare-equal/bitwise-AND blend — making the result one bit pattern on
   every ISA with no lane convention needed (total-order max is
   associative).

---

## 7. Aligned allocation & weight layout

- **Allocation guarantees (already shipped):** `CpuAllocator` returns 64-byte
  (cache-line) aligned memory by default; the `CachingAllocator` pool honors
  requests up to its fixed 256-byte upstream alignment (M1-T05/M2-T06,
  tensor design §6). Every tensor the engine allocates is therefore at least
  64-byte aligned at offset 0.
- **Alignment is a performance property, never a correctness precondition.**
  Kernels must be correct for *any* pointer alignment and any length —
  views, slices, and safetensors mmap offsets all produce unaligned bases.
  Concretely: use unaligned vector loads (`loadu` / NEON `ld1`, which is
  alignment-agnostic; the penalty on modern cores is negligible) and handle
  tails with scalar epilogues following the same element-order convention as
  the body (§6.3). The §9 sweep (sizes crossing vector-width boundaries ×
  offset bases) enforces this mechanically. No kernel may require padded
  buffers.
- **Weight layout (v1):** weights are used in checkpoint order — row-major,
  contiguous, `[out_features, in_features]` for projection matrices as
  safetensors stores them (M4), in the checkpoint's dtype (fp32/fp16/bf16;
  tensor design dtype-preservation rule). No repacking, blocking, or padding
  at load time in M3–M5. Packed/tiled GEMM layouts are **M6-T02's decision**
  (its ticket + doc update), with one constraint imposed now: the
  checkpoint-order tensor remains the source of truth, and any packed form
  is derived from it — so the `cpu` oracle and the optimized path always
  read provably identical weights.

---

## 8. CI

The existing matrix (format; GCC/Clang × Debug/Release build+test;
aggregate `ci` gate) is unchanged. M3 adds:

> *Amendment (2026-08-08, post-M4-T03):* the matrix above originally
> included a full-sweep clang-tidy job; it outgrew the 15-minute job
> timeout and was removed. clang-tidy is a local-only check now (scoped
> per-ticket rules in CLAUDE.md §Build & test), with the accepted gap that
> TUs outside the arm64 dev machine's compile database — the x86-64 cpuid
> path and the `avx2/` per-ISA TUs — are tidy-checked nowhere; CI's x86-64
> build jobs still compile them warnings-as-errors.

### 8.1 The forced-scalar test pass

Kernel-dependent test binaries are registered with ctest **twice**: the
normal run (host's best ISA) and a second registration with
`ENVIRONMENT "ENGINE_FORCE_ISA=scalar"` and a distinguishing prefix/label
(e.g. `unit-scalar/`, label `scalar`). Mechanism: an opt-in flag on
`engine_add_tests` (M3-T05 implements; exact spelling is the ticket's
latitude). Properties that matter:

- It is the *same binaries and same test sources* — no `#ifdef`-forked
  expectations. A scalar-vs-vector behavioral difference fails one pass and
  not the other, which localizes it immediately.
- It runs everywhere ctest runs: both CI runners and the dev machine, so the
  §6.2 matrix's "scalar on both hosts" row is automatic, not manual.
- Only kernel-dependent suites opt in (dispatch resolution is per-process,
  and doubling `parallel`/`tensor`/`core` suites buys nothing).

*M3-T05 implementation note:* the flag is spelled `SCALAR_PASS`; the second
registration uses prefix `<tree>-scalar/` and appends the `scalar` label
(so `ctest -L scalar` selects the pass). `tests/unit/dispatch_test.cpp` is
the first opt-in.

### 8.2 TSAN job (with M3-T04)

A new `tsan` job in `ci.yml`: Clang, Debug, `-fsanitize=thread` (a CMake
option, e.g. `ENGINE_SANITIZE=thread`, so the dev machine can run the same
configuration), executing the `parallel` unit tests including the stress
test (many small regions in a tight loop — the pattern kernels will actually
produce). Scoped to `parallel`'s tests via a ctest label rather than the
whole suite: TSAN's 5–15× slowdown is spent where the threads are. Feeds the
aggregate `ci` gate like every other job. ASan/UBSan across the whole suite
remain a separate, pre-existing concern and are unaffected.

### 8.3 macOS arm64 CI job — **decided: not now**

The ticket requires this decided here. Facts: the repository is private on
the free GitHub plan (2 000 Actions minutes/month); macOS runners bill at a
**10× minute multiplier**, so one ~10-minute macOS job costs ~100 billed
minutes per push — a handful of active days would exhaust the entire monthly
allowance on this job alone.

**Decision: no macOS arm64 CI job in M3.** The coverage it would add is NEON
execution on push — but NEON already executes on every ticket via the
mandatory dev-machine workflow, the merge gate is explicitly *CI green +
local suite green* (§6.2), and forced-scalar runs on both platforms. The
residual risk (a NEON-only regression pushed without running the local
suite) is a process failure the job would catch late anyway.

**Revisit triggers (recorded so this is a decision, not drift):** the repo
going public (arm64 macOS runners become free for public repos) or a paid
minutes budget. If revived: a single `macos-14`-or-later job, Homebrew LLVM
per CLAUDE.md, Release only, running the full suite (both ISA passes) —
mirroring the dev machine, not the Linux matrix.

---

## 9. Testing strategy

Consolidated obligations for M3's implementation tickets (each lands with its
ticket, same change as the feature):

**`parallel` (M3-T04):**
- Range/chunk edge cases: `n == 0`, `n == 1`, `n < grain` (inline path),
  `n < num_threads·grain`, `n` exactly at chunk boundaries, `grain == 1`.
- Determinism: `parallel_reduce` over adversarial mixed-magnitude fp32 input
  is bit-identical at thread counts {1, 2, 8}, and matches the same fixed
  tree computed serially.
- Lifecycle: private pools construct/destroy cleanly and repeatedly; no
  leaks under the sanitizer jobs.
- Misuse: nested `parallel_for` from a worker → CHECK death test.
- Stress (TSAN): thousands of small regions back-to-back, race-clean.

**`kernels` dispatch (M3-T05):**
- `SelectedIsa()` returns the expected ISA per platform (compile-time
  expectations per arch macro).
- `ENGINE_FORCE_ISA=scalar` verifiably changes the selected variant
  (observable via a test-only probe kernel whose variants are
  distinguishable).
- Unknown value and unavailable ISA → death tests asserting the actionable
  message.
- Null-slot fallback: a table with only `scalar` populated dispatches scalar
  under any selection.

**`kernels` ops (M3-T06):**
- The sweep: sizes {1, 15, 16, 17, 4096, 1M+} (crossing every vector width
  and the threading threshold) × unaligned bases/offsets — Class E/R
  bit-comparison against scalar per §6.3.
- Conversions match the M1-T07 `half.h` goldens bit-exactly (normals,
  subnormals, ±0, ±inf, NaN).
- The full op suite re-runs under the forced-scalar registration (§8.1).
- Microbenchmark scaffold (`benchmarks/kernels/`) with the first
  `BASELINES.md` entry: vectorized conversion ≥ 2× scalar at 1M elements on
  the dev machine (advisory).

Standing rules restated once: numerical tests state tolerances explicitly;
kernel tests land in the same change as the kernel; a variant with no test
against its reference does not merge.

---

## 10. Deferred (known, intentionally not designed here)

- **Caller participation & wait strategy:** the calling thread executing
  chunks; spin-then-park waiting. Perf work — requires the M3-T06 bench
  scaffold to exist first; any change must preserve §3.2's partitioning
  rules.
- **Work stealing / dynamic scheduling:** static round-robin is v1; revisit
  only with a measured imbalance on real kernels (chunk-id-indexed partials
  already make schedule changes determinism-safe).
- **Thread affinity, P/E-core policy (Apple Silicon), NUMA:** none in v1.
- **Wider/newer ISAs:** AVX-512, AMX, SVE — the `Isa` enum, per-TU flag
  pattern, and 16-lane Class-R convention were chosen so an addition slots
  in without breaking bit-identity commitments.
- **Native fp16/bf16 arithmetic** (NEON fp16 FMA, AVX-512 BF16): would
  change the fp32-accumulation policy — a dtype-policy amendment with its
  own error analysis, likely alongside quantization (M13/M14).
- **Grain autotuning:** grains are named per-kernel constants in v1; any
  tuning must remain a pure function of `(n, grain)` inputs to preserve
  reproducibility.
- **Exception transport out of regions:** deliberately absent (ADR-003);
  would only be revisited if a third-party dependency inside kernels forced
  it, which none does.
