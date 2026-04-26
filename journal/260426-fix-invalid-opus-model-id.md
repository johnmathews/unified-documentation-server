# 260426 — Fix invalid Opus model ID (production hotfix)

## Incident

Chat endpoint returned `502 Bad Gateway` in production. Underlying error from
the Anthropic API:

```
404 not_found_error: model: claude-opus-4-latest
```

All chat requests failed. The bug was introduced yesterday in commit `c061f60`
("Inject current UTC date into chat system prompt and upgrade to Opus"), which
set `CHAT_MODEL = "claude-opus-4-latest"`.

## Root cause

Anthropic does **not** publish a `-latest` alias for the Opus 4 model family.
The valid Opus 4.x aliases are version-pinned: `claude-opus-4-7`,
`claude-opus-4-6`, `claude-opus-4-5`, `claude-opus-4-1`, `claude-opus-4-0`.
The yesterday commit message claimed the new value would be "auto-updating",
but no such mechanism exists for this model family — the request was rejected
with a 404 the first time it hit the API in production.

The bug slipped past CI because `tests/test_chat_endpoint.py` mocks the
Anthropic client, so the model string is never validated against the real API
during tests.

## Fix

- `src/docserver/server.py` — `CHAT_MODEL` set to `"claude-opus-4-7"`, the
  current latest Opus per <https://platform.claude.com/docs/en/docs/about-claude/models/overview>.
- `docs/operations.md` — corrected the `DOCSERVER_CHAT_MODEL` table row to
  document that Anthropic has no `-latest` alias for Opus 4, and listed the
  valid version-pinned aliases.
- `docs/architecture.md` — updated the chat endpoint description with the new
  default model ID.

The `DOCSERVER_CHAT_MODEL` env-var override remains in place, so production
can be unblocked immediately by setting the env var without a redeploy if
needed.

## Follow-ups (not in this hotfix)

- Consider adding a startup-time sanity check that issues a 1-token API call
  against the configured model so an invalid model ID fails the container's
  healthcheck instead of every user request.
- Consider an integration test that hits the real Anthropic API (gated on a
  secret) to catch model-name regressions.
- When upgrading models in future, always verify the alias against the
  official models overview page before committing.
