"""Tests for the docserver.ingestion_worker entry point.

These tests exercise the worker by invoking ``main()`` directly with a
patched config and KB, rather than spawning a real subprocess. The
subprocess-spawning behaviour belongs to the supervisor's tests, not
here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import docserver.ingestion_worker as worker_module
from docserver.config import Config, RepoSource

if TYPE_CHECKING:
    from pathlib import Path


def _make_source_dir(root: Path, name: str, files: dict[str, str]) -> Path:
    src = root / name
    src.mkdir()
    for fname, content in files.items():
        (src / fname).write_text(content)
    return src


@pytest.fixture
def small_config(tmp_path: Path):
    """A minimal Config pointing to one local source with one markdown file."""
    src = _make_source_dir(tmp_path, "repo-w", {"a.md": "# Hello\n\nWorld."})
    return Config(
        sources=[RepoSource(name="repo-w", path=str(src))],
        data_dir=str(tmp_path / "data"),
    )


def test_main_returns_zero_on_success(small_config, capsys):
    """A normal cycle returns exit code 0 and emits one JSON metrics line."""
    with patch.object(worker_module, "load_config", return_value=small_config):
        rc = worker_module.main(argv=[])

    assert rc == 0
    captured = capsys.readouterr()
    metrics_lines = [
        ln for ln in captured.out.splitlines() if '"ingestion_cycle_complete"' in ln
    ]
    assert len(metrics_lines) == 1, f"expected exactly one metrics line, got {captured.out!r}"
    payload = json.loads(metrics_lines[0])
    assert payload["event"] == "ingestion_cycle_complete"
    assert "stats" in payload
    assert "metrics" in payload
    assert payload["stats"]["repo-w"]["upserted"] >= 1
    assert payload["metrics"]["flush_count"] >= 1


def test_main_returns_one_when_config_load_fails():
    with patch.object(worker_module, "load_config", side_effect=RuntimeError("boom")):
        rc = worker_module.main(argv=[])
    assert rc == 1


def test_main_returns_one_when_kb_open_fails(small_config):
    with patch.object(worker_module, "load_config", return_value=small_config), patch.object(
        worker_module, "KnowledgeBase", side_effect=RuntimeError("disk full")
    ):
        rc = worker_module.main(argv=[])
    assert rc == 1


def test_main_returns_one_when_run_once_raises(small_config):
    fake_kb = MagicMock()
    fake_ingester = MagicMock()
    fake_ingester.run_once.side_effect = RuntimeError("ingest crash")
    with patch.object(worker_module, "load_config", return_value=small_config), patch.object(
        worker_module, "KnowledgeBase", return_value=fake_kb
    ), patch.object(worker_module, "Ingester", return_value=fake_ingester):
        rc = worker_module.main(argv=[])

    assert rc == 1
    fake_kb.close.assert_called_once()  # always close even on crash


def test_main_passes_source_and_force_flags(small_config):
    """--source X --force should reach Ingester.run_once."""
    fake_kb = MagicMock()
    fake_ingester = MagicMock()
    fake_ingester.run_once.return_value = {"x": {"upserted": 0}}
    fake_ingester._last_ingestion = {"duration_s": 0.1}
    with patch.object(worker_module, "load_config", return_value=small_config), patch.object(
        worker_module, "KnowledgeBase", return_value=fake_kb
    ), patch.object(worker_module, "Ingester", return_value=fake_ingester):
        rc = worker_module.main(argv=["--source", "x", "--force"])

    assert rc == 0
    fake_ingester.run_once.assert_called_once_with(sources=["x"], force=True)


def test_nice_env_var_applied(small_config, monkeypatch):
    """DOCSERVER_INGEST_NICE should be passed to os.nice."""
    monkeypatch.setenv("DOCSERVER_INGEST_NICE", "10")
    with patch.object(worker_module, "load_config", return_value=small_config), patch.object(
        worker_module.os, "nice"
    ) as mock_nice:
        _ = worker_module.main(argv=[])
    mock_nice.assert_called_once_with(10)
