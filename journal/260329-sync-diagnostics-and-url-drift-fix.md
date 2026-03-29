# Sync Diagnostics and Origin URL Drift Fix

**Date:** 2026-03-29

## Problem

The documentation server was reported as not picking up changes pushed to remote
source repos. The scheduler runs every 5 minutes and completes successfully, but
all sources consistently show `changed=False` and all files are skipped as unchanged.

## Investigation

Traced the full ingestion pipeline: scheduler -> git fetch+reset -> file enumeration
-> SHA-256 content hash comparison -> upsert. The code was logically correct --
confirmed by writing integration tests using real git bare repos (not mocked).

Key finding: the existing sync tests all mocked GitPython, so the actual git
clone -> fetch -> detect changes -> reindex pipeline had never been tested end-to-end.

## Root Cause

Identified a likely cause: **origin URL drift**. When Docker volumes persist clone
directories across container rebuilds, the clone's git origin URL (stored in
`.git/config`) can become stale if the user rotates a token or changes the source
URL in `sources.yaml`. The `origin.fetch()` call uses the URL from the clone's
config, not the current config. This means:

1. Initial clone uses `https://OLD_TOKEN@github.com/user/repo.git`
2. User rotates token, updates `sources.yaml` to use `https://NEW_TOKEN@github.com/...`
3. Container restarts, but the clone on the volume still has the old URL
4. `origin.fetch()` uses the old URL -- if the token expired, it may silently
   fail (for public repos, anonymous fetch succeeds but may hit rate limits or
   miss private content)

## Changes

### Bug fix: origin URL sync (`ingestion.py`)

Before each fetch, the code now compares the clone's `origin.url` against the
current source config path. If they differ, it updates the origin URL via
`origin.set_url()` before fetching. This ensures token rotations and URL changes
in `sources.yaml` take effect without needing to delete the clone.

### Diagnostic logging (`ingestion.py`)

Added INFO-level logging for:
- **Fetch results**: per-ref flags (HEAD_UPTODATE, FAST_FORWARD, NEW_HEAD, etc.)
  so you can see what `git fetch` actually did
- **HEAD sha on unchanged sync**: logs the current HEAD commit hash even when
  nothing changed, so you can verify the clone is at the expected commit
- **Origin URL updates**: logs when the clone's URL is updated to match config

Previously the "no changes" path only logged at DEBUG level, making it invisible
in production.

### Integration tests (`tests/test_ingestion.py`)

Added `TestRemoteSyncIntegration` class with 6 tests using real git bare repos:
- Initial clone indexes files correctly
- New commits on the remote are detected and indexed
- Modified files are detected via content hash
- Deleted files are removed from the knowledge base
- Unchanged repos correctly skip all files
- Origin URL changes are picked up when config changes

### Documentation (`docs/operations.md`)

Added troubleshooting section "Changes not being picked up from remote repos"
with step-by-step diagnostic guide referencing the new log events.

## Test Results

All 205 tests pass (199 existing + 6 new integration tests).
