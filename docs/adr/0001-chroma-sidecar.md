# ADR 0001 — Run ChromaDB as a sidecar service

**Status:** Accepted (2026-04-29)

## Context

ChromaDB had been embedded in the docserver process via
`chromadb.PersistentClient(path="/data/chroma")` since the project
started. That worked while there was exactly one process touching the
on-disk store.

Two pressures broke that assumption:

1. The `chromadb >= 1.5.x` storage layer (`chromadb_rust_bindings`)
   does not protect its SQLite-backed store against multiple
   `PersistentClient` instances opening the same path. Concrete
   confirmation: `chroma-core/chroma#5868` (maintainer-acknowledged
   data corruption + file locking issue), with the fix landing
   post-1.5.5 in `chroma-core/chroma#6373`. Our pinned version is 1.5.5
   and does not have the fix. So the moment we want a second process
   reading or writing the store, the embedded `PersistentClient`
   pattern is unsafe.
2. The webapp had been seeing 502s during ingestion because the
   docserver's RSS spiked near the 512 MB container `mem_limit` and
   either OOM-killed or starved the request loop. Moving ingestion out
   of the request-serving process is the architectural fix (see ADR
   0002), but that immediately needs (1) — two processes against the
   same Chroma store.

## Decision

ChromaDB runs as a separate Docker service (the `chroma` container in
`docker-compose.yml`) using the official `chromadb/chroma:1.5.8` image
with `chroma run --path /chroma-data --host 0.0.0.0 --port 8000`. It
owns its own named volume `chroma-data` exclusively.

The docserver and the per-cycle ingestion worker (ADR 0002) both
connect via `chromadb.HttpClient(host=DOCSERVER_CHROMA_HOST,
port=DOCSERVER_CHROMA_PORT)`. They are pure clients; the sidecar is the
sole writer to `/chroma-data`. Multi-process safety problem dissolves —
there is still only one process touching the files.

`KnowledgeBase` keeps `PersistentClient` as a fallback for the test
suite, used when `chroma_host` is unset on the `Config`. Tests are
single-process, so the fallback path is safe and avoids needing to
spin up a Chroma server in CI.

## Alternatives considered

1. **Stay on `PersistentClient`, fix the 502s some other way.**
   Rejected because every robust fix to the 502 problem (subprocess
   ingestion, multiprocessing pool, even just running an out-of-band
   `python -m docserver.reindex` for backfills) requires a second
   process touching the store. Single-process is a dead end as soon as
   we do anything beyond the current setup.
2. **Wait for Chroma to backport `#6373` to a 1.5.x stable.** Possible
   but indefinite. The PR landed against post-1.5.5 mainline; backports
   are not promised. Blocking the 502 fix on an upstream we do not
   control is too much external dependency.
3. **Move off Chroma entirely** (e.g. to a Postgres + `pgvector`
   sidecar). Larger blast radius — would touch the embedding pipeline,
   the schema, all KB tests. Sidecar mode for Chroma is the smaller
   change.

## Consequences

### Positive

1. **Multi-process access is safe.** The HTTP server serialises
   reads/writes through a single owner of the on-disk store. Unlocks
   ADR 0002.
2. **Clearer failure boundaries.** Chroma can crash, restart, or be
   upgraded independently. `depends_on: condition: service_healthy`
   keeps the docserver offline until Chroma is reachable, so cold
   boots are deterministic.
3. **Embedding stays client-side.** The `OnnxEmbeddingFunction` is
   registered on the collection in the docserver / worker, so
   embeddings are computed in the calling process before being sent
   over HTTP. The Chroma sidecar never loads the ~500 MB ONNX model;
   it just stores and queries vectors. 256 MB `mem_limit` on the
   sidecar is generous.

### Negative

1. **Operational complexity.** One more container to start, supervise,
   back up, and upgrade. `docker compose up -d` now brings up three
   services; backups now need both `docserver-data` and `chroma-data`.
2. **Per-query HTTP hop.** Every `collection.query()` and
   `collection.upsert()` becomes a localhost HTTP round-trip. Typical
   overhead is sub-millisecond to a few ms, negligible at our scale,
   but not zero.
3. **First-deploy migration is silent.** The old `/data/chroma/`
   directory inside `docserver-data` is no longer read by anyone; the
   new `chroma-data` volume starts empty. The first ingestion cycle
   re-embeds every chunk (a few minutes of CPU on the worker, no
   operator action). The old `/data/chroma/` is dead weight until
   manually deleted.
4. **Image-tag mismatch.** Python client is pinned at
   `chromadb==1.5.5`; Docker Hub jumps `1.5.6 → 1.5.7 → 1.5.8` (no
   1.5.5 image). We pin to `chromadb/chroma:1.5.8`. Chroma's wire
   protocol is meant to be stable across patch versions but this
   specific pairing has not been formally tested.
5. **Healthcheck portability.** The compose healthcheck currently uses
   `</dev/tcp/127.0.0.1/8000` which requires bash inside the chroma
   image. If the upstream image switches to a distroless or alpine
   base, the healthcheck breaks and we replace it with a Python
   `urllib.request` one-liner.

## References

1. Issue documenting the multi-process corruption:
   <https://github.com/chroma-core/chroma/issues/5868>
2. Fix PR (post-1.5.5):
   <https://github.com/chroma-core/chroma/pull/6373>
3. Chroma client/server documentation:
   <https://docs.trychroma.com/docs/run-chroma/client-server>
4. Implementation: commit `92ed605` (`docker-compose.yml`,
   `KnowledgeBase`, `Config`, env-var docs).
