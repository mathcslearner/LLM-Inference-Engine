# Architecture Decision Records

An ADR records one architecturally significant decision: the context that forced
it, the decision itself, the alternatives that were rejected, and the
consequences we accept. ADRs exist so that later contributors (and later
sessions) can distinguish "deliberate choice" from "historical accident" without
archaeology through commit history.

## When an ADR is required

- Any decision that constrains multiple modules or is expensive to reverse:
  language/toolchain choices, module boundaries, error handling, threading
  model, serialization formats, wire protocols.
- Adding a dependency with architectural weight (e.g. an HTTP server library) —
  see the policy in [`docs/dependencies.md`](../dependencies.md).
- Any deliberate deviation from a previously accepted ADR (written as a new ADR
  that supersedes the old one).

Subsystem-internal design (data structures, algorithms, API sketches) belongs in
[`docs/design/`](../design/README.md) instead; a design doc may cite ADRs but
never contradict one.

## Conventions

- **Naming:** `ADR-NNN-short-slug.md`. Numbers are three-digit, monotonically
  increasing, and never reused — including for rejected or superseded ADRs.
- **Template:** start from [`template.md`](template.md). The Status, Context,
  Decision, and Consequences sections are mandatory.
- **Status lifecycle:** `Proposed` → `Accepted`; an accepted ADR may later
  become `Superseded by ADR-NNN`. Once accepted, an ADR is immutable except for
  status updates and links to superseding ADRs — if the decision changes, write
  a new ADR rather than editing history.

## Index

| ADR | Title | Status |
|---|---|---|
| [ADR-001](ADR-001-language-and-toolchain.md) | Language & toolchain: C++20 + CUDA | Accepted |
| [ADR-002](ADR-002-repository-layout-and-module-boundaries.md) | Repository layout & module dependency rules | Accepted |
| [ADR-003](ADR-003-error-handling.md) | Error handling: Status/StatusOr, CHECK, no exceptions | Accepted |
