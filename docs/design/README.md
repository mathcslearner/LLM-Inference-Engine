# Subsystem design docs

Each substantial subsystem gets a design doc here, written **before** its
implementation milestone — the roadmap schedules a design-doc ticket as the
first ticket of such milestones (e.g. `docs/design/tensor.md` in M1-T01).

A design doc is the working contract for one subsystem: scope and explicit
non-goals, core data structures, API sketches, ownership/threading rules, and
how the subsystem will be tested. It differs from an ADR (`docs/adr/`) in
scope and mutability: ADRs record project-wide decisions and are immutable
once accepted; design docs cover one subsystem and are **living documents** —
implementation must conform to the doc, and if implementation reveals a design
flaw, the doc is updated in the same change with a note on what changed and
why. Never silently diverge from a design doc.

Design docs may cite ADRs but never contradict them; a conflict means the ADR
needs superseding first.

Docs land here as their milestones begin. Naming: `<subsystem>.md`
(`tensor.md`, `kvcache.md`, `scheduler.md`, …).
