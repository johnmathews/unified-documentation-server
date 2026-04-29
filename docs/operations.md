# Operations

## Monitoring

### Health Endpoint

`GET http://localhost:8085/health` from the host (`/health` on port `8080` inside the docserver container)
returns JSON with the current indexing status and per-source breakdown:

```json
{
  "status": "ok",
  "total_sources": 2,
  "total_chunks": 150,
  "sources": [
    {
      "source": "home-server-docs",
      "file_count": 12,
      "chunk_count": 85,
      "last_indexed": "2026-03-23T10:00:00",
      "last_checked": "2026-03-30T12:05:00+00:00"
    },
    {
      "source": "nanoclaw",
      "file_count": 5,
      "chunk_count": 65,
      "last_indexed": "2026-03-23T10:00:00",
      "last_checked": "2026-03-30T12:05:00+00:00"
    }
  ]
}
```

Each source entry includes:

- `source` -- source name from `sources.yaml`
- `file_count` -- number of parent documents (markdown files)
- `chunk_count` -- number of indexed chunks
- `last_indexed` -- timestamp of the most recent content change that triggered re-indexing
- `last_checked` -- timestamp of the most recent successful sync check (updates every poll cycle, even when no content changed)

The top-level response also includes:

- `last_ingestion` -- diagnostic metrics from the most recent ingestion cycle, populated only after the first cycle completes. Fields: `completed_at` (ISO timestamp), `duration_s` (total cycle wall time), `rss_at_start_mb` / `rss_at_end_mb` (process peak RSS observed at cycle boundaries; monotonically non-decreasing because `ru_maxrss` is a lifetime peak), `rss_growth_mb` (delta), `flush_count`, `flush_total_s`, `flush_max_s` (per-batch upsert+embed timing).
- `chat_model_valid` -- `false` if the configured `DOCSERVER_CHAT_MODEL` was rejected by the Anthropic API at startup (e.g. invalid alias). When `false`, both `/api/chat` and `/api/chat/stream` short-circuit with HTTP 503 and the webapp can disable its chat UI proactively. The probe is skipped if `ANTHROPIC_API_KEY` is unset, leaving this field `true` (validity is assumed).
- `chat_model_error` -- the Anthropic error message that caused the probe to fail, or `null`.

Returns **503** with `{"status": "error"}` if the knowledge base is unreachable.

### Rescan Endpoint

`POST http://localhost:8085/rescan` (port `8080` inside the docserver container) triggers an immediate ingestion
cycle without waiting for the next poll interval. Optionally pass `?source=name` to rescan a single source.

```bash
# Rescan all sources
curl -X POST http://localhost:8085/rescan

# Rescan a single source
curl -X POST "http://localhost:8085/rescan?source=SRE%20documentation"
```

Returns JSON with ingestion stats and duration:

```json
{
  "status": "ok",
  "duration_ms": 1200,
  "stats": {
    "SRE documentation": {"files": 24, "new": 0, "modified": 1, "skipped": 23, "deleted": 0, "errors": 0, "upserted": 4}
  }
}
```

### Docker Health Check

The Dockerfile configures a health check against `/health`:

- Interval: 30s
- Timeout: 10s
- Start period: 60s (the embedding model is now baked into the image at build time so cold start no longer downloads it)
- Retries: 3

### Logging

Logs are structured JSON to stdout by default, suitable for Docker log collection. Each line includes `timestamp`, `level`, `logger`, `message`, and optional structured fields.

Configure with environment variables:

- `DOCSERVER_LOG_FORMAT` -- `json` (default) or `text`
- `DOCSERVER_LOG_LEVEL` -- `INFO` (default), `DEBUG`, `WARNING`, etc.

Key log events (filter with `grep '"event":'`):

| Event | Extra Fields | Description |
|-------|-------------|-------------|
| `config` | `source` | Config file loaded, each source configured |
| `kb_init` | -- | SQLite and ChromaDB initialization |
| `startup` | -- | Server started with poll interval and data dir |
| `model_cached` | -- | Embedding model loaded from persistent cache |
| `model_download_start/done` | -- | Embedding model being downloaded |
| `ingestion_start` | `sources` | Ingestion cycle beginning |
| `sync_start` / `sync_done` | `source`, `changed` | Git sync per source |
| `sync_unchanged` | `source`, `head` | Sync found no new commits (logs HEAD sha for diagnosis) |
| `fetch_info` | `source` | Per-ref fetch results with flags (HEAD_UPTODATE, FAST_FORWARD, etc.) |
| `origin_url_update` | `source` | Clone's origin URL was updated to match current config (e.g., after token rotation) |
| `clone_start` / `clone_done` | `source` | First-time clone of remote repo |
| `clone_error` | `source`, `branch`, `path` | Clone failed (URL, auth, branch, network, or disk issue) |
| `repo_path_missing` | `source`, `path` | Repo directory does not exist (mount missing or clone failed) |
| `repo_path_not_dir` | `source`, `path` | Repo path exists but is not a directory |
| `local_path_missing` | `source`, `path`, `parent_exists` | Local source path not found, with parent dir check |
| `local_path_not_dir` | `source`, `path` | Local source path is not a directory |
| `invalid_clone` | `source`, `path` | Clone directory exists but is not a valid git repo |
| `fetch_error` | `source`, `branch`, `path` | Fetch or reset failed (network, auth, branch, or corruption) |
| `corrupt_head` | `source`, `path` | HEAD unreadable due to corrupt refs — clone will be deleted and re-cloned |
| `no_files_matched` | `source`, `path`, `patterns`, `top_level_contents`, `found_doc_dirs` | Detailed diagnostics when glob patterns match nothing |
| `files_excluded` | `source`, `excluded_count` | Files removed by `exclude_patterns` |
| `files_found` | `source`, `file_count` | Files matched by glob patterns |
| `indexing_file` | `source`, `doc_id`, `change_type`, `chunks`, `progress` | Per-file progress: `[3/24] Indexing new file 'docs/setup.md' (5 chunks)` |
| `skip_summary` | `source`, `processed`, `skipped` | Summary after file loop: how many processed vs skipped |
| `ingestion_source_done` | `source`, `stats` | Per-source completion with new/modified/skipped/deleted/error counts |
| `rename_detected` | `old_name`, `new_name` | Source rename detected via URL matching |
| `rename_clone_dir` | `old_path`, `new_path` | Clone directory renamed for migrated source |
| `orphan_cleanup` | `source`, `deleted` | Orphaned source removed from KB (source removed from config) |
| `orphan_cleanup_dir` | `path` | Orphaned clone directory removed |
| `ingestion_done` | `stats` | Full cycle completion |
| `search` | `duration_ms` | Each search query with timing |
| `reindex` | `duration_ms`, `stats` | Manual reindex via MCP tool |
| `chat_request` | `conversation_id`, `model`, `history_len` | Chat endpoint called (includes user message preview) |
| `chat_complete` | `conversation_id`, `iterations`, `tool_call_count`, `reply_len`, `duration_ms` | Chat request finished |
| `chat_stream_request` | `conversation_id`, `model`, `history_len` | SSE streaming chat started |
| `chat_stream_complete` | `conversation_id`, `iterations`, `tool_call_count`, `reply_len`, `duration_ms` | SSE streaming chat finished |
| `chat_tool_call` | `tool`, `duration_ms`, `result_len` | Individual tool execution during chat (includes timing) |
| `chat_token_usage` | `iteration` | Token counts per Anthropic API call (input, output, cache) |
| `chat_rate_limit` | -- | Anthropic rate limit hit during chat |
| `conversation_create` | `conversation_id` | New conversation persisted |

Credentials in source URLs are redacted in all log output (`https://<redacted>@...`).

Docker log rotation is configured in `docker-compose.yml`: 3 files, 10MB max each.

## Source Sync Model

The server is a **read-only observer** — it never modifies source files.

Sources are classified as **remote** or **local** automatically based on the path. Git URLs (`https://`, `git@`, `ssh://`, `git://`, or paths ending in `.git`) are remote; everything else is local.

### Remote sources

Remote repos are cloned into `<data_dir>/clones/<source_name>` — a disposable copy owned by the server. On each poll cycle the server runs `git fetch` + `git reset --hard origin/<branch>` on this clone to guarantee it matches the remote exactly. This is safe because the clone is never used as a working directory.

### Local sources

Local sources are read directly from the configured path. **No git commands are ever run on local sources** — no fetch, no reset, no checkout. The server treats the directory as a plain filesystem tree regardless of whether it contains a `.git` directory.

Change detection for local sources relies entirely on content-hash comparison during ingestion: each file's SHA-256 hash is compared against the previously indexed hash, and only files with changed content are re-indexed.

This design ensures the server can never destroy uncommitted work in a local repository.

## Troubleshooting

### Failed ingestion

Check logs for these messages:

- `"Failed to parse"` -- A markdown file could not be parsed. The file is skipped but other files continue.
- `"Failed to fetch"` -- A git fetch or reset failed for a remote source. The log includes the redacted URL, branch, and clone directory, plus a list of possible causes (network, auth, branch deleted, etc.). Remote clones use `fetch` + `reset --hard` (not `pull`) so the clone always matches the remote exactly. **Local sources are never modified** — the server reads local files as-is without running any git commands.
- `"Failed to clone"` -- Initial clone of a remote repo failed. The empty clone directory is automatically removed so the next ingestion cycle retries. The log lists possible causes: bad URL, expired credentials, nonexistent branch, network issues, disk space.
- `"Unexpected error syncing"` -- Catch-all for other sync failures. Includes the source path, remote flag, and branch.

Each source is ingested independently. One source failing does not block others.

### Remote repo not cloning

If a remote source shows `"exists but is not a valid git repository"`, a previous failed clone left a corrupted directory. The log message includes the exact path and a suggested fix. Delete the directory and restart:

```bash
docker exec unified-documentation-server rm -rf "/data/clones/<source name>"
docker compose restart docserver
```

Note: since the clone error handler now auto-cleans empty directories on failure, this manual step is mainly needed for clones that partially succeeded (e.g., interrupted mid-download).

### No files found for a source

If a source syncs successfully but finds no matching files, the logs provide detailed diagnostics:

- The exact directory that was searched
- Which glob patterns were tried and how many files each matched
- A listing of the top-level directory contents (so you can see what's actually there)
- Whether common documentation directories (`docs/`, `doc/`, `wiki/`, etc.) exist
- Suggested fixes (e.g., updating patterns to `docs/**/*.md` if a `docs/` directory was found)

Common causes:

- **Wrong glob patterns:** The default pattern `**/*.md` searches the repo root recursively. If docs are in a subdirectory, use `docs/**/*.md` instead.
- **Mount not working:** For local sources, the directory may not be mounted into the container. The log checks whether the parent directory exists to help distinguish "wrong path" from "mount missing entirely".
- **Empty repo:** The repo was cloned but contains no markdown files.

### Changes not being picked up from remote repos

If you push changes to a source repo but the docserver logs keep showing `"All N file(s) unchanged"`, check these things in order:

1. **Check the `fetch_info` log.** After each sync, the server now logs per-ref fetch results. Look for:
   - `HEAD_UPTODATE` -- the remote truly has no new commits (your push may not have landed yet, or you pushed to a different branch than the one configured in `sources.yaml`)
   - `FAST_FORWARD` -- new commits were fetched and the clone was updated. If you see this but files are still "unchanged", the content hash comparison is working correctly and the file content hasn't changed.
   - No `fetch_info` log at all -- the fetch returned no info, which can indicate a connectivity or authentication issue that didn't raise an exception.

2. **Check the HEAD sha.** The `sync_unchanged` event now logs `HEAD=<sha>`. Compare this against the latest commit on your remote branch. If they don't match, the fetch isn't picking up your changes.

3. **Check the `origin_url_update` event.** If you rotated a token or changed a source URL in `sources.yaml`, the server now automatically updates the clone's origin URL. Look for this log to confirm the URL was updated. Without this fix, clones on the Docker volume would keep using the stale URL from the original clone.

4. **Check the branch name.** Ensure the `branch` field in `sources.yaml` matches the branch you're pushing to. The default is `main`. If your repo uses `master` or another default branch, you need to specify it explicitly.

5. **Force a fresh clone.** If the clone on the Docker volume is in a bad state, delete it and let the next cycle re-clone:
   ```bash
   docker exec unified-documentation-server rm -rf "/data/clones/<source-name>"
   ```
   The next ingestion cycle (within 5 minutes) will re-clone and re-index.

### Search returns no results

1. Hit the health endpoint and check `total_chunks`. If 0, no documents have been indexed.
2. Verify `config/sources.yaml` has sources configured and paths are correct.
3. First ingestion runs immediately on startup -- if the container just started, wait for the start period (up to 60s) and check again.
4. Check logs for ingestion errors on the source you expect results from.
5. Look for `no_files_matched` events which include directory contents and pattern diagnostics.

### Container shows as unhealthy

The Docker health check calls `GET /health`. A 503 response or connection failure triggers unhealthy status.

Common causes:

- The `/data` volume is not mounted or not writable. Verify the `docserver-data` volume exists and the container user (UID 1000) has write access.
- The server failed to start. Check container logs with `docker logs unified-documentation-server`.
- The server is still booting. The start period is 60s to allow Python init, KB open, and the first MCP server bind.

### Large files skipped

Files over 5MB are skipped during ingestion. A warning is logged with the file path and size. This is intentional to avoid excessive memory usage during parsing and embedding.

## Backup and Restore

Persistent data is split across **two** Docker volumes since the chroma sidecar split (WU6):

| Volume           | Mounted at                | Contents                                                          |
|------------------|---------------------------|-------------------------------------------------------------------|
| `docserver-data` | `/data` in docserver       | `/data/documents.db` (SQLite metadata, in WAL mode), `/data/clones/` (git working trees), `/data/models/` (cached ONNX embedding model, ~110 MB) |
| `chroma-data`    | `/chroma-data` in chroma sidecar | ChromaDB vector store (chunk text + embeddings)             |

A complete backup needs both volumes — backing up only `docserver-data` will leave you without the embeddings on
restore, forcing a full re-index of every document.

### Backup

```bash
docker cp unified-documentation-server:/data /backup/docserver-data
docker cp unified-documentation-chroma:/chroma-data /backup/chroma-data
```

For a more atomic snapshot, stop the stack first (`docker compose stop`) so neither side writes while the copy runs.

### Restore

Stop the stack, restore both volumes, then start:

```bash
docker compose down
docker run --rm -v unified-documentation-server_docserver-data:/data \
  -v /backup/docserver-data:/backup busybox sh -c 'cp -a /backup/. /data/'
docker run --rm -v unified-documentation-server_chroma-data:/chroma-data \
  -v /backup/chroma-data:/backup busybox sh -c 'cp -a /backup/. /chroma-data/'
docker compose up -d
```

### Full rebuild

All data is rebuildable from source repositories. Deleting both volumes and restarting will re-clone every source and
re-index every document from scratch. Takes longer than restoring a backup but needs no backup files.

```bash
docker compose down -v   # removes both volumes
docker compose up -d     # rebuilds from source repos on the next ingestion cycle
```

If only the chroma store is corrupted (e.g. after a Chroma major upgrade with an incompatible on-disk format), you can
nuke just that volume and the SQLite metadata will tell the worker to re-embed every chunk:

```bash
docker compose down
docker volume rm unified-documentation-server_chroma-data
docker compose up -d
```

## Resource Usage

Split across the three compose services since WU6:

| Service     | Typical RSS                                   | `mem_limit` | Notes                                                 |
|-------------|-----------------------------------------------|-------------|-------------------------------------------------------|
| `docserver` | ~50–80 MB steady state (no embedding model)   | 512 MB      | Spawns the ingestion worker per cycle; itself stays small. |
| ingestion worker (transient subprocess of docserver) | up to ~350 MB during a cycle, 0 between | inherits the container's 512 MB; capped to 400 MB by `RLIMIT_AS` (`DOCSERVER_INGEST_MEM_LIMIT_MB`) | Loads the ONNX model on every cycle, exits when done — RSS returns to the OS. |
| `chroma`    | ~50–100 MB depending on index size            | 256 MB      | Stateless above its `/chroma-data` volume.            |

Disk: ~100 MB per 10K documents (mostly embeddings in `chroma-data`).

CPU: low at idle; spikes during ingestion (embedding generation in the worker subprocess). The worker runs at `nice
+10` by default (`DOCSERVER_INGEST_NICE`) so it yields to the docserver's request handlers on a contended core.

Log rotation: 3 files of 10 MB max for the docserver, 3 × 5 MB for chroma + webapp (configured per service in
`docker-compose.yml`).

### Memory reclaim

Before WU6 the docserver process ran ingestion on a thread, so it accumulated glibc-arena pages over hours.
`Ingester._run_once_safe` ran `gc.collect()` and `libc.malloc_trim(0)` after each cycle to flush the arenas back to
the kernel, and that path still exists for tests that exercise the in-process `Ingester` directly.

In production it is mostly redundant: the ingestion worker is a fresh subprocess per cycle, so its entire heap is
returned to the OS on exit. The docserver itself does not allocate at ingestion-cycle scale, so its RSS stays close to
its tens-of-megabytes steady state without help.

If you ever see the **docserver** RSS climb unbounded over hours, that is now a regression rather than expected — open
a bug.

## Configuration Changes

### Changing sources

Edit `config/sources.yaml` and restart the container:

```bash
docker compose restart docserver
```

The `reindex` MCP tool only re-indexes sources that were loaded at startup. Adding a new source requires a restart so the config is reloaded.

Unchanged files are skipped during ingestion (compared by SHA-256 content hash), so restarts and re-indexes are fast when nothing has changed — even after a fresh clone where filesystem mtimes are reset.

When a source is removed from the config, the next full ingestion cycle automatically cleans up the old source's KB entries and clone directory. When a source is **renamed** (same repo URL, different name), the ingester detects this via URL matching and migrates the data in-place — preserving existing embeddings, clone directories, and content hashes — so no re-cloning or re-embedding is needed.

### Private repositories

Source paths in `sources.yaml` support `${VAR}` environment variable expansion. Use this to authenticate with private repos without hardcoding tokens:

```yaml
sources:
  - name: "private-repo"
    path: "https://${GITHUB_TOKEN}@github.com/user/repo.git"
    branch: "main"
```

Pass the token to the container via `docker-compose.yml`:

```yaml
environment:
  - GITHUB_TOKEN=${GITHUB_TOKEN}
```

And set the actual value in your `.env` file. The token needs **Contents: Read-only** permission on the target repository (use a fine-grained GitHub PAT).

If a referenced environment variable is not set, the server will raise an error at startup.

### Environment variables

Set in `docker-compose.yml` under `environment`, or in a `.env` file alongside `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCSERVER_POLL_INTERVAL` | `300` | How often (in seconds) the server polls sources for changes. Each cycle syncs remote repos (git fetch), checks local directories for modified files, and re-indexes anything that changed. Default is 300 (5 minutes). |
| `DOCSERVER_DATA_DIR` | `/data` | Root directory for all persistent data: SQLite database, ChromaDB vector store, git clones of remote repos, and cached embedding model. Mount a Docker volume here. |
| `DOCSERVER_CONFIG` | `/config/sources.yaml` | Path to the YAML config file that defines which repositories to index. Mount your config file to this path. |
| `DOCSERVER_HOST` | `0.0.0.0` | Server bind address. Default binds to all interfaces inside the container. |
| `DOCSERVER_PORT` | `8080` | Server listen port (inside the container). Map to a host port in `docker-compose.yml`. |
| `DOCSERVER_LOG_FORMAT` | `json` | Log output format: `json` for structured Docker log collection, `text` for human-readable local development. |
| `DOCSERVER_LOG_LEVEL` | `INFO` | Python log level. Set to `DEBUG` for verbose ingestion diagnostics, `WARNING` to reduce noise. |
| `DOCSERVER_CHAT_MODEL` | `claude-opus-4-7` | Anthropic model ID for the chat agent. Defaults to the current latest Opus alias. Anthropic does not publish a `-latest` alias for the Opus 4 family — set this to a specific version-aliased ID (e.g. `claude-opus-4-7`, `claude-opus-4-6`) or a pinned snapshot (e.g. `claude-opus-4-1-20250805`). To switch to a cheaper model, use `claude-sonnet-4-6` or `claude-haiku-4-5`. |
| `DOCSERVER_CHROMA_HOST` | `chroma` (in compose) | Hostname of the Chroma sidecar service. When set, the docserver and ingestion worker connect via `chromadb.HttpClient` instead of opening a `PersistentClient` directly. **Required in production** — two `PersistentClient` instances on the same on-disk path corrupt the store in Chroma 1.5.x. Leave unset for tests, which use `PersistentClient` against a tmp dir. |
| `DOCSERVER_CHROMA_PORT` | `8000` | Port the Chroma sidecar listens on. Matches the `--port` argument to `chroma run`. |
| `DOCSERVER_INGEST_NICE` | `10` (in supervisor) | Nice offset applied at the start of each ingestion worker subprocess. Lower CPU priority than the docserver process, so the request handlers stay responsive on a contended core. Set to `0` to disable. |
| `DOCSERVER_INGEST_MEM_LIMIT_MB` | unset (compose: `400`) | Soft + hard ceiling on the worker's address space (`RLIMIT_AS`), in MiB. When set lower than the container's `mem_limit`, the worker is killed first under memory pressure, leaving the docserver process untouched. |

Changes to environment variables require a container restart to take effect.

## CI/CD

A GitHub Actions workflow (`.github/workflows/docker-publish.yml`) builds and pushes the Docker image to `ghcr.io/johnmathews/unified-documentation-server` on every push to `main`.

The image is tagged with:

- `latest` -- always points to the most recent build from `main`
- `sha-<short>` -- the git commit SHA for traceability

The workflow authenticates to `ghcr.io` using the built-in `GITHUB_TOKEN` secret (no manual secret configuration needed).
