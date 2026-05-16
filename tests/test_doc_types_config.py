"""Tests for the doc_types classifier config (Stage 2 W2.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from docserver.config import (
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
    """A missing config file means every doc falls through to 'documentation'."""
    missing = tmp_path / "document-types.yml"

    config = load_doc_types_config(str(missing))

    assert config.fallback_type == DEFAULT_FALLBACK_TYPE
    assert config.global_rules == ()
    assert config.source_rules == {}
    assert "documentation" in config.types
    assert classify_doc_type("anything.md", "any-source", config) == "documentation"


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


def test_classify_with_default_config_returns_fallback() -> None:
    """A bare DocTypesConfig() (no rules) classifies every file as fallback."""
    config = DocTypesConfig()

    assert classify_doc_type("a/b/c.md", "any", config) == DEFAULT_FALLBACK_TYPE
    assert classify_doc_type("journal/x.md", "any", config) == DEFAULT_FALLBACK_TYPE
