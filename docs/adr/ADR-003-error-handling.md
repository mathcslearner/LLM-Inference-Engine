# ADR-003: Error handling — Status/StatusOr, CHECK, no exceptions

## Status

Accepted (2026-08-03)

## Context

An inference server has two very different kinds of failure. Some are expected
at runtime and must be handled gracefully per-request without disturbing the
rest of the batch: malformed request parameters, a missing weight file, an
exhausted KV-block pool. Others are programmer errors — violated invariants
that cannot occur in a correct program — where continuing would corrupt state.
A single mechanism serves both badly. C++ offers exceptions, error codes,
`std::expected` (C++23, unavailable at our C++20 baseline — ADR-001), or a
hand-rolled result type; whatever we choose must work in hot per-token paths,
across the CUDA boundary, and be enforceable at module boundaries (ADR-002).

The primitives implementing this policy landed in M0-T07: `src/core/status.h`
and `src/core/check.h`. This ADR records the policy those headers document.

## Decision

Errors are handled by one of exactly two mechanisms, chosen by failure class:

1. **Recoverable errors → `Status` / `StatusOr<T>`.** Anything that can
   legitimately fail at runtime returns `engine::core::Status`, or
   `StatusOr<T>` when the operation also produces a value. Callers propagate
   with `RETURN_IF_ERROR` / `ASSIGN_OR_RETURN` or handle explicitly.
   - Codes come from the fixed `StatusCode` taxonomy (kInvalidArgument,
     kNotFound, kResourceExhausted, …). Values are stable — they will cross
     API and process boundaries (server responses, metrics) — so codes are
     append-only and never renumbered.
   - Messages are composed with fmt via per-code factories
     (`NotFoundError("no weight named {}", name)`), compile-time checked.
   - `Is<Code>` predicates support dispatch on cause.
2. **Programmer errors → `CHECK` / `DCHECK`.** Invariants that cannot fail in
   a correct program (internal index in range, precondition between our own
   modules) are asserted with `CHECK`, which prints the condition and source
   location and aborts. `DCHECK` is `CHECK` in Debug and compiles to nothing
   under NDEBUG — for invariants too hot to verify in Release (per-token
   loops, kernels). **Never `CHECK` on input**: anything a user, file, or
   network peer can influence is a recoverable error by definition.
3. **No exceptions across module boundaries.** Third-party code may throw
   internally, but every engine module's public API is exception-free: it
   reports failure exclusively through `Status`. We do not build with
   `-fno-exceptions` (dependencies need them); we simply never let one escape
   a module.
4. **Bridging:** `CHECK_OK(expr)` asserts a `Status`-returning expression
   succeeded — for contexts where failure is a programmer error (tests, init
   paths whose inputs were already validated).

CUDA and NCCL calls (from M2 on) follow the same split: launch/config errors
that reflect engine bugs are `CHECK`-class; runtime conditions a caller could
handle (out of device memory during allocation) map to `Status` codes.

### Alternatives considered

- **Exceptions.** Idiomatic C++, zero-cost on the happy path with modern
  ABIs. Rejected: invisible control flow is a poor fit for a latency-bound
  engine loop where every failure edge must be explicit; nvcc/device code
  cannot use them, guaranteeing a second mechanism anyway; and per-request
  fault isolation in continuous batching wants errors as values that travel
  through queues, not stack unwinding through the scheduler.
- **`std::expected<T, E>`.** The standard shape of the same idea, but C++23 —
  outside our toolchain baseline. `StatusOr<T>` is deliberately close in
  spirit, easing any future migration.
- **`absl::Status` / abseil.** Battle-tested and the direct inspiration for
  our API, but pulls in a large dependency for two classes and disagrees with
  our fmt-based formatting. Writing our own (~one file plus tests) keeps the
  dependency policy (`docs/dependencies.md`) tight; the familiar API is kept
  intentionally so the trade-off stays reversible.
- **Bare error codes / `errno` style.** No payloads, trivially ignorable,
  and compose poorly. Rejected without much ceremony.

## Consequences

- Every fallible public API is visibly fallible in its signature; grepping
  `StatusOr<` enumerates them. Error-path test coverage becomes natural.
- Propagation macros are mildly unidiomatic C++ and add a line per call; we
  accept the ceremony for explicitness. (`ASSIGN_OR_RETURN` hides a temporary
  — the one macro-magic concession.)
- A stable code taxonomy must be curated: new codes append, never renumber,
  and the server layer will own a `StatusCode → HTTP` mapping in M9.
- Aborting on `CHECK` means a single violated invariant kills the process by
  design — correct for state-corruption bugs, but it obligates us to keep
  `CHECK` out of anything input-reachable. Code review enforces that line.
- `status.h` / `check.h` carry condensed copies of this policy in their file
  comments; changes to the policy must update both this ADR and those headers.
