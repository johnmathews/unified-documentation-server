"""Tests for the doc_types classifier config (Stage 2 W2.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from docserver.config import (
    _DEFAULT_GLOBAL_RULES,
    DEFAULT_FALLBACK_TYPE,
    DocTypeRule,
    DocTypesConfig,
    classify_doc_type,
    load_doc_types_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, body: str) -> str:
    path.write_text(body)
    return str(path)


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    """A missing config file uses the baked-in default rules.

    The defaults classify journal/prompt paths correctly so the Journal page
    works out-of-the-box without an operator copying the example YAML.
    """
    missing = tmp_path / "document-types.yml"

    config = load_doc_types_config(str(missing))

    assert config.fallback_type == DEFAULT_FALLBACK_TYPE
    assert config.global_rules == _DEFAULT_GLOBAL_RULES
    assert config.source_rules == {}
    assert "documentation" in config.types
    # Files outside the default patterns still fall through to the fallback.
    assert classify_doc_type("README.md", "any-source", config) == "documentation"
    # Default rules cover journal and prompt paths.
    assert classify_doc_type("journal/x.md", "any-source", config) == "journal"
    assert classify_doc_type("prompts/x.md", "any-source", config) == "prompt"


def test_load_valid_config(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types:
  - documentation
  - journal
  - prompt
  - not-docs
fallback_type: documentation
global_rules:
  - pattern: "**/journal/**"
    type: journal
  - pattern: "**/*.lock"
    type: not-docs
source_rules:
  example:
    - pattern: "notebooks/**"
      type: not-docs
""",
    )

    config = load_doc_types_config(path)

    assert config.fallback_type == "documentation"
    assert config.global_rules == (
        DocTypeRule(pattern="**/journal/**", type="journal"),
        DocTypeRule(pattern="**/*.lock", type="not-docs"),
    )
    assert "example" in config.source_rules
    assert config.source_rules["example"][0].type == "not-docs"


def test_classify_per_source_overrides_global(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, journal, not-docs]
fallback_type: documentation
global_rules:
  - pattern: "**/journal/**"
    type: journal
source_rules:
  noisy:
    - pattern: "**/journal/**"
      type: not-docs
""",
    )
    config = load_doc_types_config(path)

    # Source rules win when both match.
    assert classify_doc_type("journal/2026-05-13.md", "noisy", config) == "not-docs"
    # Other sources fall through to global rules.
    assert classify_doc_type("journal/2026-05-13.md", "quiet", config) == "journal"


def test_classify_first_match_wins(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, journal, not-docs]
fallback_type: documentation
global_rules:
  - pattern: "**/exception.md"
    type: documentation
  - pattern: "**/*.md"
    type: not-docs
""",
    )
    config = load_doc_types_config(path)

    # The exception must come before the catch-all in the YAML, mirroring
    # .gitignore semantics. The classifier returns the first match it sees.
    assert classify_doc_type("a/exception.md", "src", config) == "documentation"
    assert classify_doc_type("a/anything.md", "src", config) == "not-docs"


def test_classify_falls_through_to_fallback_type(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, journal]
fallback_type: journal
""",
    )
    config = load_doc_types_config(path)

    assert classify_doc_type("anything.md", "src", config) == "journal"


def test_unknown_type_in_rule_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, journal]
fallback_type: documentation
global_rules:
  - pattern: "**/*.md"
    type: prompt
""",
    )

    with pytest.raises(ValueError, match="prompt"):
        load_doc_types_config(path)


def test_fallback_type_not_in_vocabulary_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, journal]
fallback_type: prompt
""",
    )

    with pytest.raises(ValueError, match="fallback_type"):
        load_doc_types_config(path)


def test_malformed_rule_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation]
fallback_type: documentation
global_rules:
  - pattern: ""
    type: documentation
""",
    )

    with pytest.raises(ValueError, match="pattern"):
        load_doc_types_config(path)


def test_unknown_source_name_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, not-docs]
fallback_type: documentation
source_rules:
  ghost:
    - pattern: "**/*.md"
      type: not-docs
""",
    )

    with caplog.at_level("WARNING"):
        load_doc_types_config(path, known_source_names={"real-source"})

    assert any("ghost" in rec.getMessage() for rec in caplog.records)


def test_env_var_resolves_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, journal]
fallback_type: journal
""",
    )
    monkeypatch.setenv("DOCSERVER_DOCUMENT_TYPES_CONFIG", path)

    config = load_doc_types_config()

    assert config.fallback_type == "journal"


def test_default_config_classifies_journal_and_prompt_paths() -> None:
    """A bare DocTypesConfig() now ships baked-in defaults for common paths.

    Bug fix: the previous behaviour ("every file is documentation") made the
    webapp's Journal tab show 0 entries whenever no document-types.yml was
    present. The defaults now mirror config/document-types.example.yml so the
    common cases work out of the box.
    """
    config = DocTypesConfig()

    # Journal patterns: top-level and nested.
    assert classify_doc_type("journal/260518-x.md", "src", config) == "journal"
    assert classify_doc_type("nested/journal/y.md", "src", config) == "journal"
    # Prompt patterns: top-level and nested.
    assert classify_doc_type("prompts/foo.md", "src", config) == "prompt"
    assert classify_doc_type("nested/prompts/foo.md", "src", config) == "prompt"
    # Lock files and .DS_Store are not docs.
    assert classify_doc_type("uv.lock", "src", config) == "not-docs"
    assert classify_doc_type("nested/.DS_Store", "src", config) == "not-docs"
    # Everything else still falls through to the fallback.
    assert classify_doc_type("README.md", "src", config) == DEFAULT_FALLBACK_TYPE
    assert classify_doc_type("docs/architecture.md", "src", config) == DEFAULT_FALLBACK_TYPE


def test_yaml_global_rules_replace_defaults(tmp_path: Path) -> None:
    """An explicit YAML's global_rules REPLACES the defaults (no merge).

    Preserving original semantics: an operator who wants different rules must
    list everything they care about — the loader does not silently add the
    baked-in defaults underneath.
    """
    path = _write(
        tmp_path / "document-types.yml",
        """
types: [documentation, not-docs]
fallback_type: documentation
global_rules:
  - pattern: "**/*.log"
    type: not-docs
""",
    )

    config = load_doc_types_config(path)

    # Only the rule from YAML is present; the baked-in journal/prompt rules
    # are not merged in.
    assert config.global_rules == (DocTypeRule(pattern="**/*.log", type="not-docs"),)
    assert classify_doc_type("journal/x.md", "src", config) == "documentation"
