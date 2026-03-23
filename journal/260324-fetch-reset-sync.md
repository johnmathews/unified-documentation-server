# Replace git pull with fetch + hard reset

## Problem

The sync logic used `git pull` to update cloned repos. Since these clones are
read-only (the server never writes to them), a merge-based pull is unnecessary
and can fail if the local state ever diverges from the remote — for example due
to a force-push on the remote, manual interference in the clone directory, or
corrupted git state. When pull fails, the server logs a warning and continues
with stale data indefinitely.

## Decision

Replace `git pull` with `git fetch` + `git reset --hard origin/<branch>` in both
`_sync_remote()` and `_sync_local()`. This guarantees the local clone always
matches the remote exactly. There are no merge conflicts possible, no stale data
from a stuck pull, and force-pushes on the remote are handled transparently.

Change detection now compares `HEAD` commit SHA before and after the reset
instead of checking `FetchInfo.NEW_HEAD` flags from pull.

## Impact

- Remote and local repo syncs now always succeed as long as the fetch succeeds
- Force-pushes on the remote no longer cause permanent sync failures
- Any local modifications (shouldn't happen, but if they do) are silently
  overwritten, which is correct for a read-only indexing use case
