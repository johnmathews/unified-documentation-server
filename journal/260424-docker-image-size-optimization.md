# Docker Image Size Optimization

**Date:** 2026-04-24

## Problem

The Docker image had grown to ~1.2 GB uncompressed (~512 MB compressed on GHCR).
Investigation revealed three main causes:

1. **`chown -R /app` layer duplication (455 MB):** The single-stage Dockerfile ran
   `chown -R docserver:docserver /data /config /app` after installing dependencies and
   the ONNX model. Docker creates a full copy of every changed file in a new layer, so
   the entire `.venv` (343 MB) and `models-cache` (111 MB) were duplicated.

2. **Unused transitive dependencies (~35 MB):** `onnxruntime` pulls in `sympy` and
   `mpmath` for symbolic graph optimization, which is unused during inference.

3. **`uv` binary in final image (~49 MB):** The CMD used `uv run python -m docserver`,
   which also caused `uv run` to re-sync missing packages at container startup (including
   re-downloading stripped sympy and dev-only ruff), adding both size and cold-start latency.

## Changes

- Converted to a **multi-stage build**: builder stage does `uv sync` and model download,
  final stage uses `COPY --from=builder --chown=docserver:docserver` to set ownership
  during copy rather than in a separate layer.
- **Stripped sympy/mpmath** in the builder stage before copying to final.
- Changed CMD to `.venv/bin/python -m docserver`, removing the need for `uv` in the
  final image entirely.
- Changed model download step from `uv run python` to `.venv/bin/python` to avoid
  `uv run` pulling dev dependencies (ruff) during the build.

## Result

| Metric            | Before  | After   | Saved    |
|-------------------|---------|---------|----------|
| Uncompressed size | 1.2 GB  | 647 MB  | 553 MB (46%) |

## Verification

- Docker build succeeds
- Container starts and serves `/health` endpoint
- ONNX embeddings produce correct 768-dim vectors without sympy
- No package re-downloading at container startup
- All 364 tests pass (87% coverage)
