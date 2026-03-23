"""ONNX Runtime embedding function for all-mpnet-base-v2, compatible with ChromaDB."""

from __future__ import annotations

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space

logger = logging.getLogger(__name__)

_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
_MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
_MAX_SEQ_LENGTH = 384
_EMBEDDING_DIM = 768

# Files needed from HuggingFace Hub.
# Using int8-quantized AVX2 variant (~110MB vs ~436MB unoptimized) for x86-64 AMD/Intel.
_HF_FILES = {
    "model.onnx": "onnx/model_quint8_avx2.onnx",
    "tokenizer.json": "tokenizer.json",
}


def _default_model_dir() -> Path:
    """Resolve the default model cache directory.

    Precedence:
      1. DOCSERVER_MODEL_DIR env var
      2. /data/models (persistent volume in Docker)
      3. ~/.cache/docserver/onnx_models/all-mpnet-base-v2 (local dev fallback)
    """
    env_dir = os.environ.get("DOCSERVER_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    container_path = Path("/data/models/all-mpnet-base-v2")
    if container_path.parent.parent.exists():
        return container_path
    return Path.home() / ".cache" / "docserver" / "onnx_models" / "all-mpnet-base-v2"


def _download_model_files(target_dir: Path) -> None:
    """Download ONNX model and tokenizer from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)

    for local_name, repo_path in _HF_FILES.items():
        dest = target_dir / local_name
        if dest.exists():
            continue
        logger.info("Downloading %s from %s", repo_path, _MODEL_REPO)
        downloaded = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=repo_path,
            revision=_MODEL_REVISION,
            local_dir=str(target_dir),
        )
        # hf_hub_download may place the file in a subdirectory matching repo_path.
        # Move it to the expected flat location if needed.
        downloaded_path = Path(downloaded)
        if downloaded_path != dest:
            downloaded_path.rename(dest)


class OnnxEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function using ONNX Runtime for all-mpnet-base-v2."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else _default_model_dir()

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
        """Download model files if they don't exist locally."""
        model_path = self._model_dir / "model.onnx"
        tokenizer_path = self._model_dir / "tokenizer.json"
        if model_path.exists() and tokenizer_path.exists():
            return
        _download_model_files(self._model_dir)

    @cached_property
    def _tokenizer(self) -> Any:
        tokenizer = self._Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=_MAX_SEQ_LENGTH)
        tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=_MAX_SEQ_LENGTH)
        return tokenizer

    @cached_property
    def _session(self) -> Any:
        so = self._ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._ort.get_available_providers()
        # CoreML is slower than CPU for this workload.
        providers = [p for p in providers if p != "CoreMLExecutionProvider"]

        return self._ort.InferenceSession(
            str(self._model_dir / "model.onnx"),
            providers=providers,
            sess_options=so,
        )

    @staticmethod
    def _normalize(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return cast("npt.NDArray[np.float32]", v / norm[:, np.newaxis])

    def _forward(self, documents: list[str], batch_size: int = 32) -> npt.NDArray[np.float32]:
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            encoded = [self._tokenizer.encode(d) for d in batch]

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            onnx_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Some ONNX exports include token_type_ids, others don't.
            expected_inputs = {i.name for i in self._session.get_inputs()}
            if "token_type_ids" in expected_inputs:
                onnx_input["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

            model_output = self._session.run(None, onnx_input)
            last_hidden_state = model_output[0]

            # Mean pooling with attention mask weighting.
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden_state.shape
            )
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )

            embeddings = self._normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings)

    def __call__(self, input: Documents) -> Embeddings:
        self._ensure_model()
        embeddings = self._forward(input)
        return cast(
            "Embeddings",
            [np.array(e, dtype=np.float32) for e in embeddings],
        )

    @staticmethod
    def name() -> str:
        return "onnx_mpnet_base_v2"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> EmbeddingFunction[Documents]:
        return OnnxEmbeddingFunction(model_dir=config.get("model_dir"))

    def get_config(self) -> dict[str, Any]:
        return {"model_dir": str(self._model_dir)}

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        pass

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        pass
