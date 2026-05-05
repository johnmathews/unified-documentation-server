"""Tests for the configuration module."""

import os
import tempfile

import pytest
import yaml

from docserver.config import Config, RepoSource, _expand_env_vars, _looks_like_git_url, load_config


def test_load_config_defaults_when_no_file():
    config = load_config("/nonexistent/path.yaml")
    assert isinstance(config, Config)
    assert config.sources == []
    assert config.data_dir == "/data"
    assert config.poll_interval_seconds == 1800
    assert config.server_host == "0.0.0.0"
    assert config.server_port == 8080
    # chroma_host defaults to None — KnowledgeBase falls back to PersistentClient.
    assert config.chroma_host is None
    assert config.chroma_port == 8000


def test_chroma_env_vars(monkeypatch):
    """DOCSERVER_CHROMA_HOST/PORT should be picked up by load_config."""
    monkeypatch.setenv("DOCSERVER_CHROMA_HOST", "chroma-sidecar")
    monkeypatch.setenv("DOCSERVER_CHROMA_PORT", "9000")
    config = load_config("/nonexistent/path.yaml")
    assert config.chroma_host == "chroma-sidecar"
    assert config.chroma_port == 9000


def test_chroma_host_empty_string_treated_as_none(monkeypatch):
    """An explicitly-empty DOCSERVER_CHROMA_HOST should fall back to None
    so the KnowledgeBase still uses PersistentClient. This matters in tests
    or local dev where the env var might be set to "" by mistake."""
    monkeypatch.setenv("DOCSERVER_CHROMA_HOST", "")
    config = load_config("/nonexistent/path.yaml")
    assert config.chroma_host is None


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
    assert src.exclude_patterns == []
    assert src.is_remote is False


def test_load_config_with_exclude_patterns():
    data = {
        "sources": [
            {
                "name": "blog",
                "path": "/repos/blog",
                "exclude_patterns": ["data/**", "drafts/**"],
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()

        config = load_config(f.name)

    os.unlink(f.name)

    assert config.sources[0].exclude_patterns == ["data/**", "drafts/**"]


def test_load_config_exclude_patterns_defaults_to_empty():
    data = {
        "sources": [
            {"name": "repo", "path": "/repos/repo"},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()

        config = load_config(f.name)

    os.unlink(f.name)

    assert config.sources[0].exclude_patterns == []


def test_duplicate_source_names_raises():
    data = {
        "sources": [
            {"name": "my-repo", "path": "/repos/first"},
            {"name": "my-repo", "path": "/repos/second"},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()

        with pytest.raises(ValueError, match="Duplicate source name 'my-repo'"):
            load_config(f.name)

    os.unlink(f.name)


class TestGitUrlDetection:
    """Auto-detection of remote sources from git URLs."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/user/repo.git",
            "https://token@github.com/user/repo.git",
            "http://example.com/repo.git",
            "ssh://git@github.com/user/repo.git",
            "git://github.com/user/repo.git",
            "git@github.com:user/repo.git",
        ],
    )
    def test_git_urls_detected(self, url: str) -> None:
        assert _looks_like_git_url(url) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/repos/home-server-docs",
            "/data/clones/my-repo",
            "./relative/path",
            "../sibling/repo",
        ],
    )
    def test_local_paths_not_detected(self, path: str) -> None:
        assert _looks_like_git_url(path) is False

    def test_git_url_detected_as_remote_in_config(self) -> None:
        """A git URL in sources.yaml should automatically become a remote source."""
        data = {
            "sources": [
                {
                    "name": "auto-remote",
                    "path": "https://github.com/user/repo.git",
                    "branch": "main",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.sources[0].is_remote is True

    def test_is_remote_yaml_field_ignored(self) -> None:
        """The is_remote YAML field should be ignored — detection is automatic."""
        data = {
            "sources": [
                {
                    "name": "local-with-flag",
                    "path": "/repos/my-docs",
                    "is_remote": True,  # should be ignored — path is local
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.sources[0].is_remote is False

    def test_local_path_stays_local(self) -> None:
        """A plain local path must not be promoted to remote."""
        data = {
            "sources": [
                {
                    "name": "local-docs",
                    "path": "/repos/my-docs",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.sources[0].is_remote is False


class TestExpandEnvVars:
    def test_expands_single_var(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        assert _expand_env_vars("https://${MY_TOKEN}@github.com/repo.git") == (
            "https://secret123@github.com/repo.git"
        )

    def test_no_placeholders_unchanged(self):
        assert _expand_env_vars("/plain/path") == "/plain/path"

    def test_missing_var_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ValueError, match="MISSING_VAR"):
            _expand_env_vars("https://${MISSING_VAR}@example.com")

    def test_multiple_vars_in_one_string(self, monkeypatch):
        monkeypatch.setenv("USER", "john")
        monkeypatch.setenv("TOKEN", "secret")
        result = _expand_env_vars("https://${USER}:${TOKEN}@github.com")
        assert result == "https://john:secret@github.com"

    def test_partial_expansion_fails_on_missing(self, monkeypatch):
        """If one var exists but another doesn't, should still raise."""
        monkeypatch.setenv("GOOD_VAR", "ok")
        monkeypatch.delenv("BAD_VAR", raising=False)
        with pytest.raises(ValueError, match="BAD_VAR"):
            _expand_env_vars("${GOOD_VAR}:${BAD_VAR}")

    def test_dollar_without_braces_not_expanded(self):
        """$VAR (without braces) should not be expanded."""
        assert _expand_env_vars("$NOT_A_VAR") == "$NOT_A_VAR"

    def test_empty_string_unchanged(self):
        assert _expand_env_vars("") == ""

    def test_source_path_expanded_in_load_config(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "tok_abc")
        data = {
            "sources": [
                {
                    "name": "private-repo",
                    "path": "https://${GH_TOKEN}@github.com/user/repo.git",
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.sources[0].path == "https://tok_abc@github.com/user/repo.git"
