# Chat agent: date injection and model upgrade

## Problem

The chat agent's system prompt never included the current date, so the LLM
defaulted to its training-era assumption (~late 2024). This caused incorrect
temporal context when users asked date-sensitive questions or the model
constructed date-range queries via `query_docs`.

## Changes

### Date injection in system prompt

Added `datetime.now(UTC).strftime("%Y-%m-%d")` to the first system block in
`_prepare_chat_request`. The date is computed at request time (not server
startup) so it stays correct across midnight boundaries and for every message
in ongoing conversations.

### Model upgrade: Sonnet -> Opus

Changed the default `CHAT_MODEL` from `claude-sonnet-4-20250514` to
`claude-opus-4-latest`. Using the `latest` alias means the server picks up
new Opus 4 releases automatically without code changes. The model remains
overridable via the `DOCSERVER_CHAT_MODEL` environment variable.

### Documentation

- Updated `docs/architecture.md` to document the model and date injection.
- Added `DOCSERVER_CHAT_MODEL` to the environment variable table in
  `docs/operations.md`.

### Tests

- Added `TestSystemPromptDate` in `test_chat_endpoint.py` that verifies the
  Anthropic API call receives a system prompt starting with the UTC date.

## Files changed

- `src/docserver/server.py` — import `UTC`/`datetime`, change `CHAT_MODEL`,
  inject date into system blocks
- `tests/test_chat_endpoint.py` — new test class for date injection
- `docs/architecture.md` — document model and date in `/api/chat` description
- `docs/operations.md` — add `DOCSERVER_CHAT_MODEL` env var
