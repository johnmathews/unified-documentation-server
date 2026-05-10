"""ONNX Runtime cross-encoder reranker for L2 of the hybrid search pipeline.

Mirrors the patterns in :mod:`docserver.embedding`: lazy download from HF Hub
with a pinned revision, ``functools.cached_property`` for the ONNX session and
tokenizer, and a singleton accessed via :func:`get_reranker`. The Dockerfile
pre-bakes the model into ``/app/reranker-cache`` so production never has to
hit the network on cold start.

Failure mode: if the model fails to load or score (network down on cold
cache, corrupt file, unexpected ONNX output shape), :func:`rerank` logs at
ERROR and returns the input candidates unchanged. Search must never 500
because the reranker is sick.
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from docserver.knowledge_base import SearchResult

logger = logging.getLogger(__name__)

_MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Pinned to the commit that exposes onnx/model_quint8_avx2.onnx with the
# tokenizer.json schema we expect. Bump this only after verifying the new
# revision still ships the int8 ONNX variant and the tokenizer is compatible.
_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"

_MAX_SEQ_LENGTH = 512
_RERANK_BATCH_SIZE = 50

_HF_FILES = {
    "model.onnx": "onnx/model_quint8_avx2.onnx",
    "tokenizer.json": "tokenizer.json",
}

_IMAGE_CACHE = Path("/app/reranker-cache")


def _default_reranker_dir() -> Path:
    """Resolve the reranker model cache directory.

    Precedence (mirrors :func:`docserver.embedding._default_model_dir`):

    1. ``DOCSERVER_RERANKER_MODEL_DIR`` env var.
    2. ``/data/models/ms-marco-MiniLM-L6-v2`` — the persistent volume in
       Docker, seeded from ``/app/reranker-cache`` when empty.
    3. ``~/.cache/docserver/onnx_models/ms-marco-MiniLM-L6-v2`` for local dev.
    """
    env_dir = os.environ.get("DOCSERVER_RERANKER_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    container_path = Path("/data/models/ms-marco-MiniLM-L6-v2")
    if container_path.parent.parent.exists():
        if not container_path.exists() and _IMAGE_CACHE.exists():
            import shutil

            logger.info(
                "Seeding reranker cache from image layer %s -> %s",
                _IMAGE_CACHE,
                container_path,
                extra={"event": "reranker_seed"},
            )
            shutil.copytree(_IMAGE_CACHE, container_path)
        return container_path

    return Path.home() / ".cache" / "docserver" / "onnx_models" / "ms-marco-MiniLM-L6-v2"


def _download_model_files(target_dir: Path) -> None:
    """Download ONNX model and tokenizer from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)

    for local_name, repo_path in _HF_FILES.items():
        dest = target_dir / local_name
        if dest.exists():
            continue
        logger.info(
            "Downloading reranker file %s from %s",
            repo_path,
            _MODEL_REPO,
            extra={"event": "reranker_download_file"},
        )
        downloaded = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=repo_path,
            revision=_MODEL_REVISION,
            local_dir=str(target_dir),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != dest:
            downloaded_path.rename(dest)


class OnnxCrossEncoder:
    """ONNX-backed cross-encoder reranker for ms-marco-MiniLM-L6-v2."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else _default_reranker_dir()
        self._model_ready = False

        try:
            import onnxruntime

            self._ort = onnxruntime
        except ImportError as err:
            raise ImportError("onnxruntime is required: pip install onnxruntime") from err
        try:
            import tokenizers

            self._Tokenizer = tokenizers.Tokenizer
        except ImportError as err:
            raise ImportError("tokenizers is required: pip install tokenizers") from err

    def _ensure_model(self) -> None:
        if self._model_ready:
            return

        model_path = self._model_dir / "model.onnx"
        tokenizer_path = self._model_dir / "tokenizer.json"
        if model_path.exists() and tokenizer_path.exists():
            logger.info(
                "Reranker model loaded from cache at %s",
                self._model_dir,
                extra={"event": "reranker_cached"},
            )
            self._model_ready = True
            return

        logger.info(
            "Reranker model not found at %s, downloading...",
            self._model_dir,
            extra={"event": "reranker_download_start"},
        )
        _download_model_files(self._model_dir)
        logger.info(
            "Reranker model download complete",
            extra={"event": "reranker_download_done"},
        )
        self._model_ready = True

    def unload(self) -> bool:
        unloaded = False
        for attr in ("_session", "_tokenizer"):
            if attr in self.__dict__:
                del self.__dict__[attr]
                unloaded = True
        if unloaded:
            logger.info(
                "Reranker model unloaded from memory",
                extra={"event": "reranker_unloaded"},
            )
        return unloaded

    @cached_property
    def _tokenizer(self) -> Any:
        # Cross-encoders see ``[CLS] query [SEP] passage [SEP]``. Truncate
        # only the second sequence so the query is never lost. Padding is
        # batch-dynamic (no fixed length) — we pad to the longest pair in
        # each call rather than always to 512.
        tokenizer = self._Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=_MAX_SEQ_LENGTH, strategy="only_second")
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        return tokenizer

    @cached_property
    def _session(self) -> Any:
        so = self._ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._ort.get_available_providers()
        # CoreML is slower than CPU for this workload (mirrors embedding.py).
        providers = [p for p in providers if p != "CoreMLExecutionProvider"]

        return self._ort.InferenceSession(
            str(self._model_dir / "model.onnx"),
            providers=providers,
            sess_options=so,
        )

    def _score_pairs(self, query: str, passages: list[str]) -> npt.NDArray[np.float32]:
        """Run the cross-encoder over (query, passage) pairs in batches.

        Returns a 1-D array of logits (one float per passage); higher = more
        relevant. Empty input returns an empty array without invoking the
        session.
        """
        if not passages:
            return np.zeros(0, dtype=np.float32)

        all_scores: list[npt.NDArray[np.float32]] = []
        expected_inputs = {i.name for i in self._session.get_inputs()}

        for start in range(0, len(passages), _RERANK_BATCH_SIZE):
            batch = passages[start : start + _RERANK_BATCH_SIZE]
            encoded = self._tokenizer.encode_batch([(query, p) for p in batch])

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

            onnx_input: dict[str, npt.NDArray[np.int64]] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in expected_inputs:
                onnx_input["token_type_ids"] = type_ids

            logits = self._session.run(None, onnx_input)[0]
            # ms-marco-MiniLM-L6-v2 outputs shape (batch, 1); squeeze the
            # singleton dim. Defensive against (batch,) too.
            scores = np.asarray(logits, dtype=np.float32).reshape(-1)
            all_scores.append(scores)

        return np.concatenate(all_scores)

    def _rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        self._ensure_model()
        passages = [c["content"] for c in candidates]
        scores = self._score_pairs(query, passages)

        # Build a parallel list of reranked candidates with the rerank score
        # added; preserve the original L1 ``score`` field for debugging.
        scored: list[tuple[float, SearchResult]] = []
        for cand, rerank_score in zip(candidates, scores, strict=True):
            new_cand = cast("SearchResult", dict(cand))
            new_cand["score"] = float(rerank_score)
            scored.append((float(rerank_score), new_cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


_reranker: OnnxCrossEncoder | None = None


def get_reranker() -> OnnxCrossEncoder:
    """Return the process-global reranker singleton (lazily constructed)."""
    global _reranker
    if _reranker is None:
        _reranker = OnnxCrossEncoder()
    return _reranker


def rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int = 10,
) -> list[SearchResult]:
    """Rerank L1 candidates with the cross-encoder; return top ``top_k``.

    Replaces the candidates' ``score`` field with the cross-encoder logit
    (higher = more relevant). On any failure (model load error, ONNX
    inference error, tokenizer error), logs at ERROR and returns
    ``candidates[:top_k]`` unchanged so the caller still gets a usable list.
    """
    try:
        return get_reranker()._rerank(query, candidates, top_k)
    except Exception:
        logger.exception(
            "Reranker failed; returning L1 results unchanged.",
            extra={"event": "reranker_failed"},
        )
        return candidates[:top_k]
