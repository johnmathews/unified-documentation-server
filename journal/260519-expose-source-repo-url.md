# 260519 — Expose per-source repo URL/branch on /health

Backend half of webapp fix-batch-7 **W4** ("View on GitHub" link). The
webapp needs to know, per source, whether it is github-backed and where,
so it can build a blob URL for the current document. That metadata
already lived in `RepoSource` but was never sent to the frontend.

## Changes

- `src/docserver/config.py`:
  - `_github_base_url(path)` — normalises a source's configured path
    (https / ssh / `git@github.com:` / `.git` suffix) to a browseable
    `https://github.com/owner/repo` base. Returns `None` for local
    paths and non-github hosts (gitlab, bitbucket) so the webapp shows
    no link rather than a broken one.
  - `RepoSource.github_url` property delegating to it.
- `src/docserver/server.py`: the `/health` per-source breakdown now
  includes `repo_url` (from `github_url`) and `branch`, looked up from
  `config.sources` by name. Additive, optional fields — no change to
  ingestion, storage, or the doc-ID scheme.
- Docs: `docs/architecture.md` `/health` entry now lists
  `repo_url`/`branch`.

## Tests

- `tests/test_config.py::TestGithubBaseUrl` — https/ssh/git@/.git forms
  normalise correctly; local paths and non-github hosts → `None`
  (asserted both via `_github_base_url` and the `RepoSource.github_url`
  property).
- `tests/test_server.py::TestHealthEndpoint::test_health_returns_ok` —
  asserts `repo_url`/`branch` present in each source entry and
  `repo_url` is either `None` or a `https://github.com/` URL.
- Full suite: `uv run pytest tests/ -q` → **505 passed**. `ruff` clean.

## Cross-reference

Frontend consumer + the other six items of this batch:
`unified-documentation-webapp` journal `260519-fix-batch-7-ux-and-chat-page.md`.

## Note (out of scope, flagged to user)

`config/sources.local.yaml` has a duplicate source name
(`unified-documentation-server` listed twice) which makes
`load_config` raise on local startup. Not fixed here — it's the user's
gitignored local config, not committed state.
