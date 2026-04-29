FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev --no-editable

# Pre-download the ONNX embedding model into the image layer.
# At runtime, it is copied to /data/models (persistent volume) if not already there.
ENV DOCSERVER_MODEL_DIR=/app/models-cache
RUN .venv/bin/python -c "from docserver.embedding import OnnxEmbeddingFunction; ef = OnnxEmbeddingFunction(); ef._ensure_model()"
ENV DOCSERVER_MODEL_DIR=

# Remove sympy (~30MB) - onnxruntime lists it as a dep but only uses it for
# symbolic graph optimization, not inference. Safe to strip.
RUN rm -rf .venv/lib/python*/site-packages/sympy \
           .venv/lib/python*/site-packages/sympy-*.dist-info \
           .venv/lib/python*/site-packages/mpmath \
           .venv/lib/python*/site-packages/mpmath-*.dist-info

# ── final stage ──
FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -r -u 1000 -m docserver && \
    mkdir -p /data /config && \
    chown docserver:docserver /data /config

WORKDIR /app
COPY --from=builder --chown=docserver:docserver /app /app

USER docserver

EXPOSE 8080

# start-period was 120s when the embedding model was downloaded on first
# boot. Since commit e178a52 the model is baked into the image at build
# time, so cold start no longer waits on a network download. 60s is enough
# for Python interpreter init + KB open + first MCP server bind.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD [".venv/bin/python", "-m", "docserver"]
