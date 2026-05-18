"""Configuration module for the documentation MCP server."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from typing import cast

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
    exclude_patterns: list[str] = field(default_factory=list)
    is_remote: bool = False


@dataclass
class Config:
    sources: list[RepoSource]
    data_dir: str = "/data"
    poll_interval_seconds: int = 1800
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    # When chroma_host is set, KnowledgeBase uses chromadb.HttpClient against
    # the sidecar service. When None, it falls back to chromadb.PersistentClient
    # against ``{data_dir}/chroma``. Tests use the latter; production sets the
    # former via DOCSERVER_CHROMA_HOST so the docserver and the ingestion
    # worker can share the database safely.
    chroma_host: str | None = None
    chroma_port: int = 8000


def _parse_sources(raw: list[dict[str, object]]) -> list[RepoSource]:
    sources: list[RepoSource] = []
    for item in raw:
        name = str(item["name"])
        raw_path = str(item["path"])
        expanded_path = _expand_env_vars(raw_path)
        is_remote = _looks_like_git_url(expanded_path)

        # Log path with credentials redacted
        if "@" in expanded_path:
            display_path = re.sub(r"://[^@]+@", "://<redacted>@", expanded_path)
        else:
            display_path = expanded_path

        branch = str(item.get("branch", "main"))
        patterns_raw = item.get("patterns")
        if isinstance(patterns_raw, list):
            glob_patterns = [str(p) for p in cast("list[object]", patterns_raw)]
        else:
            glob_patterns = ["**/*.md"]

        exclude_raw = item.get("exclude_patterns")
        if isinstance(exclude_raw, list):
            exclude_patterns = [str(p) for p in cast("list[object]", exclude_raw)]
        else:
            exclude_patterns = []

        logger.info(
            "Configured source '%s': path=%s, remote=%s, branch=%s",
            name,
            display_path,
            is_remote,
            branch,
            extra={"event": "config", "source": name},
        )

        sources.append(
            RepoSource(
                name=name,
                path=expanded_path,
                branch=branch,
                glob_patterns=glob_patterns,
                exclude_patterns=exclude_patterns,
                is_remote=is_remote,
            )
        )
    seen_names: dict[str, str] = {}
    for src in sources:
        if src.name in seen_names:
            raise ValueError(
                f"Duplicate source name '{src.name}': "
                + f"paths '{seen_names[src.name]}' and '{src.path}' "
                + "both use the same name. Each source must have a unique name."
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

    raw: dict[str, object] = {}
    if os.path.exists(path):
        with open(path) as fh:
            loaded = cast("dict[str, object] | None", yaml.safe_load(fh))
        if isinstance(loaded, dict):
            raw = loaded
        logger.info("Config file parsed successfully", extra={"event": "config"})
    else:
        logger.warning(
            "Config file not found at %s, using defaults", path, extra={"event": "config"}
        )

    raw_sources = raw.get("sources")
    if isinstance(raw_sources, list):
        dict_items: list[dict[str, object]] = [
            cast("dict[str, object]", item)
            for item in cast("list[object]", raw_sources)
            if isinstance(item, dict)
        ]
        sources = _parse_sources(dict_items)
    else:
        sources: list[RepoSource] = []

    data_dir = os.environ.get(
        "DOCSERVER_DATA_DIR",
        str(raw.get("data_dir", "/data")),
    )

    poll_interval_seconds = int(
        os.environ.get(
            "DOCSERVER_POLL_INTERVAL",
            str(raw.get("poll_interval", 1800)),
        )
    )

    server_host = os.environ.get(
        "DOCSERVER_HOST",
        str(raw.get("server_host", "0.0.0.0")),
    )

    server_port = int(
        os.environ.get(
            "DOCSERVER_PORT",
            str(raw.get("server_port", 8080)),
        )
    )

    chroma_host_env = os.environ.get("DOCSERVER_CHROMA_HOST")
    chroma_host: str | None
    if chroma_host_env is not None:
        chroma_host = chroma_host_env or None
    else:
        chroma_host_yaml = raw.get("chroma_host")
        chroma_host = str(chroma_host_yaml) if chroma_host_yaml else None

    chroma_port = int(
        os.environ.get(
            "DOCSERVER_CHROMA_PORT",
            str(raw.get("chroma_port", 8000)),
        )
    )

    return Config(
        sources=sources,
        data_dir=data_dir,
        poll_interval_seconds=poll_interval_seconds,
        server_host=server_host,
        server_port=server_port,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )


# ---------------------------------------------------------------------------
# Document-type classifier (Stage 2 W2.2).
# ---------------------------------------------------------------------------

# Vocabulary returned when no document-types.yml exists. Stage 2 ships four types;
# the YAML schema's ``types`` block lets the operator extend without code
# changes. Keep ``DEFAULT_FALLBACK_TYPE`` in this list.
_DEFAULT_DOC_TYPES: tuple[str, ...] = ("documentation", "journal", "prompt", "not-docs")
DEFAULT_FALLBACK_TYPE = "documentation"


@dataclass(frozen=True)
class DocTypeRule:
    """A single classifier rule. Order within a list defines precedence."""

    pattern: str
    type: str


# Baked-in defaults that mirror config/document-types.example.yml. These apply
# when no document-types.yml is present so the common cases (journal/, prompts/,
# *.lock, .DS_Store) classify correctly out of the box. An explicit YAML that
# defines its own ``global_rules`` REPLACES these — see ``load_doc_types_config``.
# When changing this tuple, bump ``_DOC_TYPES_DEFAULTS_VERSION`` in
# ``knowledge_base.py`` so existing deployments reclassify on next startup.
_DEFAULT_GLOBAL_RULES: tuple[DocTypeRule, ...] = (
    DocTypeRule(pattern="**/journal/**", type="journal"),
    DocTypeRule(pattern="journal/**", type="journal"),
    DocTypeRule(pattern="**/prompts/**", type="prompt"),
    DocTypeRule(pattern="prompts/**", type="prompt"),
    DocTypeRule(pattern="**/.DS_Store", type="not-docs"),
    DocTypeRule(pattern="**/*.lock", type="not-docs"),
)


@dataclass(frozen=True)
class DocTypesConfig:
    """Parsed ``document-types.yml``. Frozen so it can be shared across threads."""

    types: tuple[str, ...] = _DEFAULT_DOC_TYPES
    fallback_type: str = DEFAULT_FALLBACK_TYPE
    global_rules: tuple[DocTypeRule, ...] = _DEFAULT_GLOBAL_RULES
    source_rules: dict[str, tuple[DocTypeRule, ...]] = field(default_factory=dict)


def _parse_rule_list(
    raw: object,
    valid_types: set[str],
    context: str,
) -> tuple[DocTypeRule, ...]:
    """Parse a list of ``{pattern, type}`` dicts. Raises on malformed entries."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{context}: expected a list of rules, got {type(raw).__name__}")

    rules: list[DocTypeRule] = []
    for idx, item in enumerate(cast("list[object]", raw)):
        if not isinstance(item, dict):
            raise ValueError(f"{context}[{idx}]: expected a mapping with 'pattern' and 'type'")
        entry = cast("dict[str, object]", item)
        pattern = entry.get("pattern")
        type_name = entry.get("type")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{context}[{idx}]: 'pattern' must be a non-empty string")
        if not isinstance(type_name, str) or not type_name:
            raise ValueError(f"{context}[{idx}]: 'type' must be a non-empty string")
        if type_name not in valid_types:
            raise ValueError(
                f"{context}[{idx}]: type '{type_name}' is not in the configured "
                f"vocabulary {sorted(valid_types)}"
            )
        rules.append(DocTypeRule(pattern=pattern, type=type_name))
    return tuple(rules)


def load_doc_types_config(
    path: str | None = None,
    *,
    known_source_names: set[str] | None = None,
) -> DocTypesConfig:
    """Load classifier config from a YAML file.

    Location precedence:
      1. ``path`` argument
      2. ``DOCSERVER_DOCUMENT_TYPES_CONFIG`` environment variable
      3. ``/config/document-types.yml`` (default)

    A missing file returns ``DocTypesConfig()`` which carries the baked-in
    default ``global_rules`` (journal/, prompts/, *.lock, .DS_Store). Validation
    errors raise ``ValueError``; references to unknown source names emit a
    warning.
    """
    if path is None:
        path = os.environ.get(
            "DOCSERVER_DOCUMENT_TYPES_CONFIG", "/config/document-types.yml"
        )

    if not os.path.exists(path):
        logger.warning(
            "document-types config not found at %s; using built-in defaults "
            "(journal/, prompts/, *.lock, .DS_Store) with fallback '%s'",
            path,
            DEFAULT_FALLBACK_TYPE,
            extra={"event": "doc_types_config_missing", "path": path},
        )
        return DocTypesConfig()

    with open(path) as fh:
        loaded = cast("dict[str, object] | None", yaml.safe_load(fh))
    raw: dict[str, object] = loaded if isinstance(loaded, dict) else {}

    raw_types = raw.get("types")
    if isinstance(raw_types, list):
        types_list = [str(t) for t in cast("list[object]", raw_types)]
        if not types_list:
            raise ValueError("document-types config: 'types' must contain at least one entry")
        types = tuple(types_list)
    else:
        types = _DEFAULT_DOC_TYPES
    valid_types = set(types)

    fallback_raw = raw.get("fallback_type", DEFAULT_FALLBACK_TYPE)
    if not isinstance(fallback_raw, str) or not fallback_raw:
        raise ValueError("document-types config: 'fallback_type' must be a non-empty string")
    if fallback_raw not in valid_types:
        raise ValueError(
            f"document-types config: fallback_type '{fallback_raw}' is not in the "
            f"vocabulary {sorted(valid_types)}"
        )

    global_rules = _parse_rule_list(raw.get("global_rules"), valid_types, "global_rules")

    raw_source_rules = raw.get("source_rules")
    source_rules: dict[str, tuple[DocTypeRule, ...]] = {}
    if isinstance(raw_source_rules, dict):
        for source_name, raw_rules in cast(
            "dict[str, object]", raw_source_rules
        ).items():
            source_name_str = str(source_name)
            parsed = _parse_rule_list(
                raw_rules, valid_types, f"source_rules['{source_name_str}']"
            )
            source_rules[source_name_str] = parsed
            if known_source_names is not None and source_name_str not in known_source_names:
                logger.warning(
                    "document-types config: source '%s' is not in sources.yaml; rules ignored",
                    source_name_str,
                    extra={"event": "doc_types_unknown_source", "source": source_name_str},
                )
    elif raw_source_rules is not None:
        raise ValueError(
            "document-types config: 'source_rules' must be a mapping of source name → rule list"
        )

    return DocTypesConfig(
        types=types,
        fallback_type=fallback_raw,
        global_rules=global_rules,
        source_rules=source_rules,
    )


def _pattern_matches(file_path: str, pattern: str) -> bool:
    """Match *file_path* against *pattern* with gitignore-style ``**/`` semantics.

    Standard :func:`fnmatch.fnmatch` already lets ``*`` cross slashes, so most
    glob patterns work as written. The one place fnmatch diverges from
    gitignore is the leading ``**/`` prefix: in gitignore, ``**/foo`` matches
    ``foo`` at any depth *including* the repo root. fnmatch requires the
    leading slash so it would miss top-level matches. We try the pattern
    verbatim first; on a miss we also try the version with the ``**/`` prefix
    stripped, so operators can write a single rule for "foo anywhere".
    """
    if fnmatch.fnmatch(file_path, pattern):
        return True
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(file_path, pattern[3:])
    return False


def classify_doc_type(
    file_path: str, source_name: str, config: DocTypesConfig
) -> str:
    """Return the type for *file_path* under *source_name*.

    Per-source rules are evaluated first, then ``global_rules``; the first
    pattern that matches wins. If nothing matches, ``config.fallback_type`` is
    returned. See :func:`_pattern_matches` for the matching semantics.
    """
    for rule in config.source_rules.get(source_name, ()):
        if _pattern_matches(file_path, rule.pattern):
            return rule.type
    for rule in config.global_rules:
        if _pattern_matches(file_path, rule.pattern):
            return rule.type
    return config.fallback_type
