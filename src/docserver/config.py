"""Configuration module for the documentation MCP server."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")

# Patterns that indicate a git URL rather than a local filesystem path.
_GIT_URL_RE = re.compile(
    r"^(?:https?://|ssh://|git://|git@)",
    re.IGNORECASE,
)


def _looks_like_git_url(path: str) -> bool:
    """Return True if *path* looks like a remote git URL rather than a local path."""
    if _GIT_URL_RE.match(path):
        return True
    # Bare ".git" suffix on something that isn't an existing local directory
    return path.endswith(".git") and not os.path.isdir(path)


def _expand_env_vars(value: str) -> str:
    """Replace ``${VAR}`` placeholders in *value* with environment variables."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return env_val

    return _ENV_VAR_RE.sub(_replace, value)


@dataclass
class RepoSource:
    name: str
    path: str
    branch: str = "main"
    glob_patterns: list[str] = field(default_factory=lambda: ["**/*.md"])
    is_remote: bool = False


@dataclass
class Config:
    sources: list[RepoSource]
    data_dir: str = "/data"
    poll_interval_seconds: int = 30
    server_host: str = "0.0.0.0"
    server_port: int = 8080


def _parse_sources(raw: list[dict[str, Any]]) -> list[RepoSource]:
    sources = []
    for item in raw:
        name = item["name"]
        raw_path = item["path"]
        expanded_path = _expand_env_vars(raw_path)
        is_remote = _looks_like_git_url(expanded_path)

        # Log path with credentials redacted
        if "@" in expanded_path:
            display_path = re.sub(r"://[^@]+@", "://<redacted>@", expanded_path)
        else:
            display_path = expanded_path

        logger.info(
            "Configured source '%s': path=%s, remote=%s, branch=%s",
            name,
            display_path,
            is_remote,
            item.get("branch", "main"),
            extra={"event": "config", "source": name},
        )

        sources.append(
            RepoSource(
                name=name,
                path=expanded_path,
                branch=item.get("branch", "main"),
                glob_patterns=item.get("patterns", ["**/*.md"]),
                is_remote=is_remote,
            )
        )
    seen_names: dict[str, str] = {}
    for src in sources:
        if src.name in seen_names:
            raise ValueError(
                f"Duplicate source name '{src.name}': "
                f"paths '{seen_names[src.name]}' and '{src.path}' "
                f"both use the same name. Each source must have a unique name."
            )
        seen_names[src.name] = src.path

    return sources


def load_config(path: str | None = None) -> Config:
    """Load configuration from a YAML file, with env var overrides.

    Config file location precedence:
      1. ``path`` argument
      2. ``DOCSERVER_CONFIG`` environment variable
      3. ``/config/sources.yaml`` (default)

    Environment variables override YAML values:
      - DOCSERVER_DATA_DIR
      - DOCSERVER_POLL_INTERVAL
      - DOCSERVER_HOST
      - DOCSERVER_PORT
    """
    if path is None:
        path = os.environ.get("DOCSERVER_CONFIG", "/config/sources.yaml")

    logger.info("Loading config from %s", path, extra={"event": "config"})

    raw: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Config file parsed successfully", extra={"event": "config"})
    else:
        logger.warning(
            "Config file not found at %s, using defaults", path, extra={"event": "config"}
        )

    sources = _parse_sources(raw.get("sources", []))

    data_dir = os.environ.get(
        "DOCSERVER_DATA_DIR",
        str(raw.get("data_dir", "/data")),
    )

    poll_interval_seconds = int(
        os.environ.get(
            "DOCSERVER_POLL_INTERVAL",
            raw.get("poll_interval", 300),
        )
    )

    server_host = os.environ.get(
        "DOCSERVER_HOST",
        str(raw.get("server_host", "0.0.0.0")),
    )

    server_port = int(
        os.environ.get(
            "DOCSERVER_PORT",
            raw.get("server_port", 8080),
        )
    )

    return Config(
        sources=sources,
        data_dir=data_dir,
        poll_interval_seconds=poll_interval_seconds,
        server_host=server_host,
        server_port=server_port,
    )
