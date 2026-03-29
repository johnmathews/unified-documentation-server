# Operations

## Monitoring

### Health Endpoint

`GET :8080/health` returns JSON with the current indexing status and per-source breakdown:

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
      "last_indexed": "2026-03-23T10:00:00"
    },
    {
      "source": "nanoclaw",
      "file_count": 5,
      "chunk_count": 65,
      "last_indexed": "2026-03-23T10:00:00"
    }
  ]
}
```

Each source entry includes:

- `source` -- source name from `sources.yaml`
- `file_count` -- number of parent documents (markdown files)
- `chunk_count` -- number of indexed chunks
- `last_indexed` -- timestamp of the most recent indexing for that source

Returns **503** with `{"status": "error"}` if the knowledge base is unreachable.

### Rescan Endpoint

`POST :8080/rescan` triggers an immediate ingestion cycle without waiting for the next poll interval. Optionally pass `?source=name` to rescan a single source.

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
- Start period: 120s (allows time for embedding model load and first ingestion)
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

Credentials in source URLs are redacted in all log output (`https://<redacted>@...`).

Docker log rotation is configured in `docker-compose.yml`: 3 files, 10MB max each.

## Troubleshooting

### Failed ingestion

Check logs for these messages:

- `"Failed to parse"` -- A markdown file could not be parsed. The file is skipped but other files continue.
- `"Failed to fetch"` -- A git fetch or reset failed for a source. The log includes the redacted URL, branch, and clone directory, plus a list of possible causes (network, auth, branch deleted, etc.). The server uses `fetch` + `reset --hard` (not `pull`) so that the local clone always matches the remote exactly, with no merge conflicts possible.
- `"Failed to clone"` -- Initial clone of a remote repo failed. The empty clone directory is automatically removed so the next ingestion cycle retries. The log lists possible causes: bad URL, expired credentials, nonexistent branch, network issues, disk space.
- `"Unexpected error syncing"` -- Catch-all for other sync failures. Includes the source path, remote flag, and branch.

Each source is ingested independently. One source failing does not block others.

### Remote repo not cloning

If a remote source shows `"exists but is not a valid git repository"`, a previous failed clone left a corrupted directory. The log message includes the exact path and a suggested fix. Delete the directory and restart:

```bash
docker exec documentation-mcp-server rm -rf "/data/clones/<source name>"
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
   docker exec documentation-mcp-server rm -rf "/data/clones/<source-name>"
   ```
   The next ingestion cycle (within 5 minutes) will re-clone and re-index.

### Search returns no results

1. Hit the health endpoint and check `total_chunks`. If 0, no documents have been indexed.
2. Verify `config/sources.yaml` has sources configured and paths are correct.
3. First ingestion runs immediately on startup -- if the container just started, wait for the start period (up to 120s) and check again.
4. Check logs for ingestion errors on the source you expect results from.
5. Look for `no_files_matched` events which include directory contents and pattern diagnostics.

### Container shows as unhealthy

The Docker health check calls `GET /health`. A 503 response or connection failure triggers unhealthy status.

Common causes:

- The `/data` volume is not mounted or not writable. Verify the `docserver-data` volume exists and the container user (UID 1000) has write access.
- The server failed to start. Check container logs with `docker logs documentation-mcp-server`.
- The embedding model is still loading. The start period is 120s to account for this.

### Large files skipped

Files over 5MB are skipped during ingestion. A warning is logged with the file path and size. This is intentional to avoid excessive memory usage during parsing and embedding.

## Backup and Restore

All persistent data lives in the `docserver-data` Docker volume, mounted at `/data` inside the container.

Contents:

| Path | Description |
|------|-------------|
| `/data/documents.db` | SQLite database (document metadata) |
| `/data/chroma/` | ChromaDB vector store (embeddings) |
| `/data/clones/` | Git clones of remote repos |
| `/data/models/` | Cached ONNX embedding model (~110MB) |

### Backup

```bash
docker cp documentation-mcp-server:/data /backup/docserver-data
```

### Restore

Copy the backed-up data back into the volume and restart the container.

### Full rebuild

The data is fully rebuildable from source repositories. Deleting the volume and restarting the container will re-clone all repos and re-index everything from scratch. This takes longer than restoring a backup but requires no backup files.

```bash
docker compose down -v   # removes the volume
docker compose up -d     # rebuilds from source repos
```

## Resource Usage

| Resource | Typical | Limit |
|----------|---------|-------|
| Memory | ~1--1.5 GB (embedding model + ChromaDB) | 2 GB (set in docker-compose.yml) |
| Disk | ~100 MB per 10K documents | -- |
| CPU | Low at idle; spikes during ingestion (embedding generation) | -- |

Log rotation: 3 files of 10 MB max (configured in `docker-compose.yml` logging options).

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
    is_remote: true
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
| `DOCSERVER_POLL_INTERVAL` | `300` | Seconds between ingestion cycles |
| `DOCSERVER_DATA_DIR` | `/data` | Persistent storage directory |
| `DOCSERVER_CONFIG` | `/config/sources.yaml` | Path to config file |
| `DOCSERVER_HOST` | `0.0.0.0` | Server bind address |
| `DOCSERVER_PORT` | `8080` | Server port |
| `DOCSERVER_LOG_FORMAT` | `json` | `json` or `text` |
| `DOCSERVER_LOG_LEVEL` | `INFO` | Python log level |

Changes to environment variables require a container restart to take effect.

## CI/CD

A GitHub Actions workflow (`.github/workflows/docker-publish.yml`) builds and pushes the Docker image to `ghcr.io/johnmathews/documentation-mcp-server` on every push to `main`.

The image is tagged with:

- `latest` -- always points to the most recent build from `main`
- `sha-<short>` -- the git commit SHA for traceability

The workflow authenticates to `ghcr.io` using the built-in `GITHUB_TOKEN` secret (no manual secret configuration needed).
