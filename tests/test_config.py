"""Tests for the configuration module."""

import os
import tempfile

import pytest
import yaml

from docserver.config import Config, RepoSource, load_config


def test_load_config_defaults_when_no_file():
    config = load_config("/nonexistent/path.yaml")
    assert isinstance(config, Config)
    assert config.sources == []
    assert config.data_dir == "/data"
    assert config.poll_interval_seconds == 300
    assert config.server_host == "0.0.0.0"
    assert config.server_port == 8080


def test_load_config_from_yaml():
    data = {
        "sources": [
            {
                "name": "test-repo",
                "path": "/repos/test",
                "branch": "develop",
                "patterns": ["docs/**/*.md"],
            }
        ],
        "poll_interval": 60,
        "data_dir": "/tmp/test-data",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()

        config = load_config(f.name)

    os.unlink(f.name)

    assert len(config.sources) == 1
    assert config.sources[0].name == "test-repo"
    assert config.sources[0].branch == "develop"
    assert config.sources[0].glob_patterns == ["docs/**/*.md"]
    assert config.poll_interval_seconds == 60
    assert config.data_dir == "/tmp/test-data"


def test_env_vars_override_yaml(monkeypatch):
    data = {
        "sources": [],
        "poll_interval": 60,
        "data_dir": "/yaml-data",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()

        monkeypatch.setenv("DOCSERVER_DATA_DIR", "/env-data")
        monkeypatch.setenv("DOCSERVER_POLL_INTERVAL", "120")
        monkeypatch.setenv("DOCSERVER_PORT", "9090")

        config = load_config(f.name)

    os.unlink(f.name)

    assert config.data_dir == "/env-data"
    assert config.poll_interval_seconds == 120
    assert config.server_port == 9090


def test_repo_source_defaults():
    src = RepoSource(name="test", path="/test")
    assert src.branch == "main"
    assert src.glob_patterns == ["**/*.md"]
    assert src.is_remote is False
