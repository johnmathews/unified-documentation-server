"""Tests for the ONNX embedding function."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docserver.embedding import OnnxEmbeddingFunction, _default_model_dir


@pytest.fixture(scope="module")
def ef() -> OnnxEmbeddingFunction:
    """Module-scoped to avoid re-downloading/loading the model for each test."""
    return OnnxEmbeddingFunction()


class TestOnnxEmbeddingFunction:
    def test_returns_768_dimensions(self, ef: OnnxEmbeddingFunction) -> None:
        result = ef(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 768

    def test_batch_input(self, ef: OnnxEmbeddingFunction) -> None:
        docs = ["first document", "second document", "third document"]
        result = ef(docs)
        assert len(result) == 3
        for emb in result:
            assert len(emb) == 768

    def test_embeddings_are_normalized(self, ef: OnnxEmbeddingFunction) -> None:
        result = ef(["test normalization"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_closer_than_dissimilar(self, ef: OnnxEmbeddingFunction) -> None:
        embeddings = ef(
            [
                "the cat sat on the mat",
                "a kitten rested on the rug",
                "quantum mechanics explains particle behavior",
            ]
        )
        cat1 = np.array(embeddings[0])
        cat2 = np.array(embeddings[1])
        quantum = np.array(embeddings[2])

        sim_similar = np.dot(cat1, cat2)
        sim_dissimilar = np.dot(cat1, quantum)
        assert sim_similar > sim_dissimilar

    def test_empty_string(self, ef: OnnxEmbeddingFunction) -> None:
        result = ef([""])
        assert len(result) == 1
        assert len(result[0]) == 768


class TestChromaDBInterface:
    """Test ChromaDB EmbeddingFunction interface methods."""

    def test_name(self) -> None:
        assert OnnxEmbeddingFunction.name() == "onnx_mpnet_base_v2"

    def test_default_space(self, ef: OnnxEmbeddingFunction) -> None:
        assert ef.default_space() == "cosine"

    def test_supported_spaces(self, ef: OnnxEmbeddingFunction) -> None:
        spaces = ef.supported_spaces()
        assert "cosine" in spaces
        assert "l2" in spaces
        assert "ip" in spaces

    def test_get_config(self, ef: OnnxEmbeddingFunction) -> None:
        config = ef.get_config()
        assert "model_dir" in config
        assert isinstance(config["model_dir"], str)

    def test_build_from_config(self, ef: OnnxEmbeddingFunction) -> None:
        config = ef.get_config()
        rebuilt = OnnxEmbeddingFunction.build_from_config(config)
        assert isinstance(rebuilt, OnnxEmbeddingFunction)

    def test_validate_config(self) -> None:
        OnnxEmbeddingFunction.validate_config({"model_dir": "/tmp/test"})

    def test_validate_config_update(self, ef: OnnxEmbeddingFunction) -> None:
        ef.validate_config_update({}, {"model_dir": "/tmp/new"})


class TestUnload:
    """Test lazy unloading and transparent reload of the embedding model."""

    def test_unload_returns_true_when_loaded(self, ef: OnnxEmbeddingFunction) -> None:
        """unload() should return True when session/tokenizer were cached."""
        # Ensure model is loaded by running an embedding
        ef(["trigger load"])
        assert "_session" in ef.__dict__
        assert "_tokenizer" in ef.__dict__
        assert ef.unload() is True

    def test_unload_returns_false_when_not_loaded(self) -> None:
        """unload() should return False on a fresh instance (nothing cached)."""
        func = OnnxEmbeddingFunction()
        assert func.unload() is False

    def test_unload_evicts_cached_properties(self, ef: OnnxEmbeddingFunction) -> None:
        """After unload, _session and _tokenizer should not be in __dict__."""
        ef(["trigger load"])
        ef.unload()
        assert "_session" not in ef.__dict__
        assert "_tokenizer" not in ef.__dict__

    def test_reload_after_unload_produces_correct_embeddings(
        self, ef: OnnxEmbeddingFunction
    ) -> None:
        """After unload, the next __call__ should transparently reload and work."""
        before = ef(["hello world"])
        ef.unload()
        after = ef(["hello world"])
        # Embeddings should be identical (deterministic model)
        np.testing.assert_allclose(before[0], after[0], atol=1e-6)

    def test_unload_is_idempotent(self, ef: OnnxEmbeddingFunction) -> None:
        """Calling unload() twice should be safe — second call returns False."""
        ef(["trigger load"])
        assert ef.unload() is True
        assert ef.unload() is False

    def test_unload_logs_event(self, ef: OnnxEmbeddingFunction, caplog: pytest.LogCaptureFixture) -> None:
        """unload() should log when it actually unloads."""
        ef(["trigger load"])
        with caplog.at_level("INFO", logger="docserver.embedding"):
            ef.unload()
        assert any("model_unloaded" in r.message or "unloaded" in r.message for r in caplog.records)


class TestModelDownload:
    """Test model download and initialization logic."""

    def test_ensure_model_skips_when_files_exist(self, ef: OnnxEmbeddingFunction) -> None:
        """_ensure_model should not download when files already exist."""
        with patch("docserver.embedding._download_model_files") as mock_dl:
            ef._ensure_model()
            mock_dl.assert_not_called()

    def test_ensure_model_downloads_when_missing(self, tmp_path: Path) -> None:
        """_ensure_model should trigger download when model files are missing."""
        func = OnnxEmbeddingFunction(model_dir=tmp_path / "nonexistent")
        with patch("docserver.embedding._download_model_files") as mock_dl:
            func._ensure_model()
            mock_dl.assert_called_once()

    def test_custom_model_dir(self, tmp_path: Path) -> None:
        """Should accept a custom model directory."""
        func = OnnxEmbeddingFunction(model_dir=tmp_path)
        assert func._model_dir == tmp_path


class TestDefaultModelDir:
    """Test _default_model_dir resolution."""

    def test_uses_env_var_when_set(self) -> None:
        with patch.dict("os.environ", {"DOCSERVER_MODEL_DIR": "/custom/models"}):
            result = _default_model_dir()
            assert result == Path("/custom/models")

    def test_uses_data_volume_when_available(self) -> None:
        import os

        env = os.environ.copy()
        env.pop("DOCSERVER_MODEL_DIR", None)
        with patch.dict("os.environ", env, clear=True), patch(
            "docserver.embedding.Path"
        ) as MockPath:
            # Simulate /data existing (container environment)
            mock_parent = MagicMock()
            mock_parent.parent.exists.return_value = True
            MockPath.return_value = mock_parent
            MockPath.home.return_value = Path.home()
            result = _default_model_dir()
            assert result == mock_parent

    def test_uses_home_cache_when_no_data_volume(self) -> None:
        import os

        env = os.environ.copy()
        env.pop("DOCSERVER_MODEL_DIR", None)
        with patch.dict("os.environ", env, clear=True), patch(
            "docserver.embedding.Path"
        ) as MockPath:
            # Simulate /data not existing (local dev)
            mock_container = MagicMock()
            mock_container.parent.parent.exists.return_value = False
            MockPath.return_value = mock_container
            MockPath.home.return_value = Path.home()
            result = _default_model_dir()
            assert ".cache" in str(result)
            assert "all-mpnet-base-v2" in str(result)


class TestImportErrors:
    """Test helpful error messages when dependencies are missing."""

    def test_missing_onnxruntime(self) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="onnxruntime is required"),
        ):
            OnnxEmbeddingFunction.__init__(MagicMock(), model_dir="/tmp")

    def test_missing_tokenizers(self) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "tokenizers":
                raise ImportError("No module named 'tokenizers'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="tokenizers is required"),
        ):
            OnnxEmbeddingFunction.__init__(MagicMock(), model_dir="/tmp")
