# Documentation MCP Server

An MCP server that indexes documentation from git repositories and makes it searchable by AI agents. Designed to run as a
containerized service on a home server, providing documentation context to agents via the Model Context Protocol.

## Architecture

```
Git Repos (local/remote)
        |
        v
  [Ingestion Worker]    Subprocess spawned per cycle (~5 min), parses markdown,
        |               chunks text, embeds; exits and releases RSS to the OS
        |
        v
  [Knowledge Base]      SQLite (WAL mode) for metadata
        |               ChromaDB sidecar (HTTP) for vector embeddings
        |
        v
  [MCP Server]          FastMCP with streamable HTTP transport. Long-running,
        |               isolated from ingestion's memory + GIL pressure.
        |
        v
  AI Agent (nanoclaw)   Queries docs via MCP tools
```

### Docker Compose services

`docker compose up -d` brings up three containers:

| Service                | Image                                         | Purpose                                                                                     |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `chroma`               | `chromadb/chroma:1.5.8`                       | Owns `/chroma-data` exclusively. Serves vector queries on port 8000 over HTTP.              |
| `docserver`            | `ghcr.io/johnmathews/unified-documentation-server:latest` | The MCP server. Connects to `chroma` via `HttpClient`. Spawns the ingestion worker per tick. |
| `documentation-webapp` | `ghcr.io/johnmathews/unified-documentation-webapp:latest` | Optional web UI. Waits for `docserver` to become `healthy` before starting.                 |

The chroma sidecar is required, not optional: `chromadb >= 1.5.x` corrupts its store when two `PersistentClient`
instances open the same on-disk path, so the long-running server and the per-cycle ingestion worker need a single
process owning the database. The HTTP server fills that role.

## MCP Tools

### `search_docs` -- Semantic search

Find documentation relevant to a natural language question.

| Parameter     | Type  | Default | Description                                      |
| ------------- | ----- | ------- | ------------------------------------------------ |
| `query`       | `str` | --      | Natural language search query (required).         |
| `num_results` | `int` | `10`    | Maximum number of results to return (1--100).     |
| `source`      | `str` | `""`    | Optional source name to restrict results to one repo. |

### `query_docs` -- Structured metadata query

Query document metadata by source, path, title, or date range. Useful for questions like "list all docs in source Y" or "what was added after date Z".

| Parameter            | Type  | Default | Description                                 |
| -------------------- | ----- | ------- | ------------------------------------------- |
| `source`             | `str` | `""`    | Filter by source name.                      |
| `file_path_contains` | `str` | `""`    | Filter by substring in file path.           |
| `title_contains`     | `str` | `""`    | Filter by substring in title.               |
| `created_after`      | `str` | `""`    | ISO date string, e.g. `"2024-01-01"`.       |
| `created_before`     | `str` | `""`    | ISO date string.                            |
| `limit`              | `int` | `20`    | Maximum number of results to return (1--100). |

### `get_document` -- Retrieve by ID

Retrieve a specific document or chunk by its ID. Document IDs follow the format `source_name:relative/path` for parent documents, or `source_name:relative/path#chunkN` for chunks.

| Parameter | Type  | Default | Description                         |
| --------- | ----- | ------- | ----------------------------------- |
| `doc_id`  | `str` | --      | The document ID to retrieve (required). |

### `list_sources` -- List sources and status

List all configured documentation sources and their indexing status. Returns source names, file counts, chunk counts, and last indexed time. Takes no parameters.

### `reindex` -- Trigger re-indexing

Trigger an immediate re-indexing of documentation sources.

| Parameter | Type  | Default | Description                                            |
| --------- | ----- | ------- | ------------------------------------------------------ |
| `source`  | `str` | `""`    | Optional source name. If empty, re-indexes all sources. |

## Health Endpoint

`GET /health` returns the current status of the knowledge base, the most recent ingestion cycle, and the chat model
configuration.

**200 OK** — server is reachable. The body is a structured snapshot:

```json
{
  "status": "healthy",
  "total_sources": 3,
  "total_chunks": 542,
  "poll_interval_seconds": 300,
  "sources": [ /* per-source health */ ],
  "last_ingestion": {
    "completed_at": "2026-04-29T17:25:00+00:00",
    "duration_s": 4.2,
    "rss_at_end_mb": 240.0,
    "flush_count": 3
  },
  "last_ingestion_failure": null,
  "chat_model_valid": true,
  "chat_model_error": null
}
```

Notable fields:

- `last_ingestion` — duration and peak-RSS metrics from the most recent worker cycle. Populated only after the first
  cycle has run; null on a freshly started container.
- `last_ingestion_failure` — set when the most recent worker subprocess exited non-zero, timed out, or did not emit a
  metrics line. Useful for spotting silent ingestion stalls without scraping logs.
- `chat_model_valid` / `chat_model_error` — set by a startup probe that calls `models.retrieve(DOCSERVER_CHAT_MODEL)`
  on the Anthropic API. When false, `/api/chat` and `/api/chat/stream` short-circuit with HTTP 503 instead of letting
  every request fail at the API call.

**503 Service Unavailable** — knowledge base is unreachable or errored:

```json
{"status": "error"}
```

This endpoint is used by the Docker health check configured in `docker-compose.yml` and by the webapp's
`depends_on: condition: service_healthy` gate.

## Quick Start

### 1. Configure sources

```bash
cp config/sources.example.yaml config/sources.yaml
# Edit config/sources.yaml to add your documentation repos
```

### 2. Set required secrets

The chat agent calls the Anthropic API. Either export `ANTHROPIC_API_KEY` in your shell before running compose, or
write it into a local `.env` file (Docker Compose auto-loads `.env` from the project root):

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

If you do not need the chat endpoints, set `ANTHROPIC_API_KEY=unset` (or any non-empty value) and skip them — the
search and metadata MCP tools work without an Anthropic key.

### 3. Run with Docker Compose

`docker-compose.yml` is the canonical deploy file and it works out of the box: it brings up the three services with
named volumes only, no host-specific bind mounts. If you want to index a directory that lives on the host filesystem,
uncomment the example bind-mount stanza in the `docserver` service's `volumes:` block and add a matching `sources:`
entry in `config/sources.yaml` whose `path:` points at the container-side mount.

```bash
docker compose up -d
```

This brings up three containers — `chroma`, `docserver`, and `documentation-webapp` — and two named volumes
(`chroma-data`, `docserver-data`). The webapp waits for the docserver's `/health` to be green before starting; the
docserver waits for the chroma sidecar to be reachable.

Host ports (per `docker-compose.yml`):

| Service                | Host | Container | Notes                            |
| ---------------------- | ---- | --------- | -------------------------------- |
| `docserver`            | 8085 | 8080      | MCP and REST endpoints           |
| `documentation-webapp` | 3002 | 3000      | Browser UI                       |
| `chroma`               | —    | 8000      | Internal only; not exposed       |

### 4. Connect from an MCP client

Add to your MCP client configuration (e.g., `.mcp.json`):

```json
{
 "mcpServers": {
  "documentation": {
   "url": "http://localhost:8085/mcp"
  }
 }
}
```

### Updating to a new release

The `latest` tag on each image is overwritten on every push to `main`. To pull a fresh build:

```bash
docker compose pull              # pulls all 3 images
docker compose up -d             # recreates containers using the new images
```

The persistent volumes (`docserver-data`, `chroma-data`) are preserved across this — no re-ingestion is needed unless
the sidecar's storage format has changed in a major Chroma upgrade. Roll back with `docker compose pull --policy never`
plus an explicit older tag if a release is broken.

### Persistent volumes

| Volume           | Mounted at         | What it holds                                             |
| ---------------- | ------------------ | --------------------------------------------------------- |
| `docserver-data` | `/data` in docserver | SQLite (`documents.db`), git clones (`/data/clones/`), cached ONNX embedding model (`/data/models/`) |
| `chroma-data`    | `/chroma-data` in chroma | ChromaDB vector store (chunks + embeddings)         |

`config/sources.yaml` is bind-mounted read-only into the docserver container — edit it on the host and run
`docker compose restart docserver` to pick up changes (see `docs/operations.md` § Configuration Changes).

## Configuration

### sources.yaml

```yaml
sources:
 - name: "my-docs"
   path: "/repos/my-docs" # Local path (mount in docker-compose)
   branch: "main"
   patterns:
    - "**/*.md"

 - name: "remote-docs"
   path: "https://github.com/user/repo.git"
   branch: "main"

poll_interval: 300 # Seconds between index cycles
data_dir: "/data" # Persistent storage path
```

### Environment Variables

| Variable                          | Default                | Description                                                                                        |
| --------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| `DOCSERVER_CONFIG`                | `/config/sources.yaml` | Path to config file                                                                                |
| `DOCSERVER_DATA_DIR`              | `/data`                | Persistent storage directory                                                                       |
| `DOCSERVER_POLL_INTERVAL`         | `300`                  | Polling interval in seconds                                                                        |
| `DOCSERVER_HOST`                  | `0.0.0.0`              | Server bind address                                                                                |
| `DOCSERVER_PORT`                  | `8080`                 | Server port                                                                                        |
| `DOCSERVER_LOG_FORMAT`            | `json`                 | Log format (`json` or `text`)                                                                      |
| `DOCSERVER_LOG_LEVEL`             | `INFO`                 | Log level                                                                                          |
| `DOCSERVER_CHAT_MODEL`            | `claude-opus-4-7`      | Anthropic model ID for the chat agent. Use a version-aliased ID; Anthropic does not publish a `-latest` alias for Opus 4. |
| `DOCSERVER_CHROMA_HOST`           | unset (compose: `chroma`) | Hostname of the Chroma sidecar. **Required in production**; tests fall back to `PersistentClient` when unset. |
| `DOCSERVER_CHROMA_PORT`           | `8000`                 | Port the Chroma sidecar listens on.                                                                |
| `DOCSERVER_INGEST_NICE`           | `10` (set by supervisor) | Nice offset applied to each ingestion worker subprocess. Lower priority than the docserver process. |
| `DOCSERVER_INGEST_MEM_LIMIT_MB`   | unset (compose: `400`) | Soft + hard `RLIMIT_AS` ceiling on the worker, in MiB. Lower than the container `mem_limit` so the worker is killed first under memory pressure. |

See `docs/operations.md` for the full table including all options.

## Development

```bash
uv sync --group dev
uv run pytest tests/ -v
```

## How It Works

1. **Ingestion runs as a separate process.** An `IngesterSupervisor` in the docserver process owns an APScheduler timer.
   On each tick (and on every `POST /rescan`), it spawns `python -m docserver.ingestion_worker` as a subprocess. The
   worker loads the embedding model, runs one cycle, and exits — its peak RSS is fully released to the OS, so the
   long-running docserver process stays at its small steady-state working set even when a cycle peaks high. If the
   worker OOMs or crashes, the docserver keeps serving requests; only the cycle is lost.

2. **Sync.** The worker polls configured git repos. For remote repos, it clones on first run then pulls updates. For
   local repos (mounted as volumes), it pulls if they have a remote, or just reads the files directly.

3. **Parsing.** Markdown files are parsed to extract titles (first `#` heading), creation dates (from git history), and
   modification times. Documents are split into ~400-character chunks at section and paragraph boundaries, with each
   chunk prefixed by its heading hierarchy (e.g. `[Setup > Ports]`) and ~100 chars of overlap between chunks. Lists and
   code fences are kept intact.

4. **Storage.** Parent document metadata goes into SQLite (in WAL mode so the docserver can read while the worker
   writes). Document chunks are embedded client-side via ONNX Runtime and stored in the ChromaDB sidecar service for
   semantic search; only pre-computed vectors cross the wire, so the sidecar stays small (~256 MB).

5. **Serving.** The FastMCP server exposes tools over streamable HTTP. Agents can search semantically, query by
   metadata, or retrieve specific documents. A `/health` endpoint returns indexing status (including the most recent
   worker cycle's RSS / duration) for container orchestration and operator visibility.
