# Architecture Decision Records

Lightweight records of architectural decisions, in roughly chronological
order. Each ADR captures one decision: the problem that forced the
choice, the option taken, the alternatives considered, and the
consequences (good and bad) we accept by taking it.

ADRs are append-only. When a decision is superseded by a later one, the
original record stays in place and gains a `Status: Superseded by ADR
NNNN` line at the top — the reasoning at the time is part of the value.

## Format

[Michael Nygard style](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions.html),
trimmed to four sections per ADR: Status, Context, Decision,
Consequences. No template-bloat; if a section has nothing useful to say,
it is omitted.

## Index

| #    | Title                                                     | Status   |
|------|-----------------------------------------------------------|----------|
| 0001 | [Run ChromaDB as a sidecar service](0001-chroma-sidecar.md) | Accepted |
| 0002 | [Run ingestion as a per-cycle subprocess](0002-ingestion-subprocess.md) | Accepted |
| 0003 | [Soft startup probe for chat-model validity](0003-chat-model-startup-probe.md) | Accepted |
