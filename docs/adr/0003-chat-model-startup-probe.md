# ADR 0003 — Soft startup probe for chat-model validity

**Status:** Accepted (2026-04-29)

## Context

On 2026-04-25, commit `c061f60` set the docserver's chat model to
`claude-opus-4-latest`, claiming in the commit message that this would
"auto-update". Anthropic does not publish a `-latest` alias for the
Opus 4 family — the only valid aliases are version-pinned
(`claude-opus-4-7`, `claude-opus-4-6`, etc.). Every chat request
returned `404 not_found_error: model: claude-opus-4-latest` from the
Anthropic API, which the docserver wrapped as HTTP 502.

Production was broken from the moment the deploy landed until the
hotfix in `7b34119`. The bug class is "config string flows untouched
into a third-party API"; CI couldn't catch it because
`tests/test_chat_endpoint.py` mocks `anthropic.Anthropic`, so the
model argument never reached a real validator.

The hotfix corrected the value but did not address the underlying
fragility: any future model-name change has the same blind spot.

## Decision

At server startup, after the chat-model env var resolves, the
docserver makes a cheap `client.models.retrieve(chat_model)` call
against the Anthropic API.

1. On `NotFoundError`, log at ERROR with the upstream message, set the
   module-level `_chat_model_valid = False`, and **continue starting
   normally**. Search and document-browsing endpoints keep working;
   only the chat endpoints are degraded.
2. On `APIConnectionError` (network glitch during boot), log a warning
   and assume valid — do not fail closed for a transient blip.
3. If `ANTHROPIC_API_KEY` is unset, skip the probe entirely and assume
   valid (no key means no chat anyway).
4. `/health` exposes `chat_model_valid` and `chat_model_error` so the
   webapp can disable its chat UI proactively.
5. `/api/chat` and `/api/chat/stream` short-circuit with HTTP 503
   (`Service Unavailable`) when `_chat_model_valid is False`, with a
   body containing the upstream error message — better than 502
   because 503 specifically signals "the dependency this endpoint
   needs is unavailable; retrying without a fix won't help".
6. The chat error-mapping table is narrowed at the same time:
   `NotFoundError`/`BadRequestError` → 500 (server config),
   `RateLimitError` → 429 (preserve upstream signal),
   `APIConnectionError`/`InternalServerError` → 502 (true gateway
   failure). A 4xx from Anthropic is no longer a 502.

The test suite gains a one-line guard: each chat test asserts
`mock_client.messages.create.call_args.kwargs["model"] == CHAT_MODEL`,
so the value of `CHAT_MODEL` cannot drift from what the tests expect
without a test failure.

## Alternatives considered

1. **Hard-exit on probe failure** (`sys.exit(1)`). The first version of
   this plan. Rejected because a misconfigured chat model would also
   stop the search and document-browsing surfaces, which have nothing
   to do with the chat dependency. Silent degradation was the harm in
   the original incident; over-correcting to "kill everything when
   chat is wrong" replaces silent degradation with louder
   over-coupling.
2. **Static allow-list of known model prefixes.** Cheap, no API call,
   but doesn't catch deprecated/retired model IDs that match the
   prefix. The probe tests against the live API, which is where the
   failure mode actually lived.
3. **Real-API integration test in CI** (gated on a CI secret). Would
   also have caught the original bug, but spends API credits on every
   PR and doesn't help on first-boot in production. The startup probe
   plus the mock-call assertion together cover the same defect class
   without recurring cost.
4. **Do nothing; rely on the hotfix.** Rejected: the hotfix repaired
   the symptom but left the bug class intact. The next model change
   would have the same blind spot.

## Consequences

### Positive

1. **Failure mode is observable.** A misconfigured model produces an
   ERROR log line within ~1 s of container startup, a `false` on
   `/health.chat_model_valid`, and a clean 503 (with the upstream
   message in the body) on every chat request. Compare to silent 502s
   on the original incident.
2. **Search and browsing keep working.** The probe failure does not
   take down the rest of the server. The webapp can disable just the
   chat UI based on `/health.chat_model_valid` and the rest of the
   product remains usable.
3. **Status codes regain their meaning.** A 4xx from Anthropic is now
   a 5xx-with-clear-class (500 for server config, 429 for rate limit)
   instead of a generic 502 that misleads the operator into thinking
   the gateway/network is the problem.

### Negative

1. **Network dependency at boot.** Cold starts now make one Anthropic
   API call. Adds ~200 ms to startup; if Anthropic is unreachable
   right at boot, we get a warning and proceed. If they are flaky
   later, the probe state from boot is what `/health` reports — it is
   not re-run periodically.
2. **No re-probe on model change at runtime.** The probe only runs at
   startup. If the model is deprecated mid-run by Anthropic, requests
   start failing with `NotFoundError` and the new error mapping
   produces 500s. The supervisor does not restart the docserver on
   that — the operator notices via logs or `/api/chat` returning 500.
3. **Two layers of protection might look redundant.** The probe sets
   `_chat_model_valid` and the chat handlers also catch
   `NotFoundError` per-request. Both are kept on purpose: the probe
   gives loud-and-early; the per-request handler covers the runtime
   deprecation case.

## References

1. Hotfix that surfaced the problem: commit `7b34119`,
   `journal/260426-fix-invalid-opus-model-id.md`.
2. Implementation: commit `e6977c2`.
3. Anthropic models reference (used to pick the new default
   `claude-opus-4-7`):
   <https://platform.claude.com/docs/en/docs/about-claude/models/overview>
