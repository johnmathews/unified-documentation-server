"""Configuration module for the documentation MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


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
    poll_interval_seconds: int = 300
    server_host: str = "0.0.0.0"
    server_port: int = 8080


def _parse_sources(raw: list[dict[str, Any]]) -> list[RepoSource]:
    sources = []
    for item in raw:
        sources.append(RepoSource(
            name=item["name"],
            path=item["path"],
            branch=item.get("branch", "main"),
            glob_patterns=item.get("patterns", ["**/*.md"]),
            is_remote=item.get("is_remote", False),
        ))
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

    raw: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

    sources = _parse_sources(raw.get("sources", []))

    data_dir = os.environ.get(
        "DOCSERVER_DATA_DIR",
        str(raw.get("data_dir", "/data")),
    )

    poll_interval_seconds = int(os.environ.get(
        "DOCSERVER_POLL_INTERVAL",
        raw.get("poll_interval", 300),
    ))

    server_host = os.environ.get(
        "DOCSERVER_HOST",
        str(raw.get("server_host", "0.0.0.0")),
    )

    server_port = int(os.environ.get(
        "DOCSERVER_PORT",
        raw.get("server_port", 8080),
    ))

    return Config(
        sources=sources,
        data_dir=data_dir,
        poll_interval_seconds=poll_interval_seconds,
        server_host=server_host,
        server_port=server_port,
    )
