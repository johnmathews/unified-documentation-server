# README Ingestion and Tree API Categories

**Date:** 2026-03-24

## Problem

The documentation MCP server's chat agent could not find README.md files when asked. Two causes:

1. **Ingestion gap**: Sources with custom glob patterns (e.g. `docs/**/*.md`, `journal/**/*.md`) did not pick up root-level README.md files.
2. **Tree API gap**: The `/api/tree` endpoint categorized all non-journal documents as "docs", with no concept of root-level files. The UI had no way to show READMEs at the source root.

## Changes

### Ingestion (`ingestion.py`)

After pattern-matching, `get_files()` now auto-includes root-level README.md even when custom patterns are specified. Handles case-insensitive filesystems (macOS) via `Path.samefile()` to avoid duplicates. Only one README variant is included (checks README.md, readme.md, Readme.md in order).

### Tree API (`knowledge_base.py`)

`get_document_tree()` now categorizes documents into three buckets:
- `root_docs` — files at the repo root (no directory separator in path)
- `docs` — files in subdirectories (excluding journal/)
- `journal` — files under journal/

### Documentation

Added REST API endpoints to the architecture doc (they were previously undocumented). Updated ingestion docs to mention README auto-inclusion.

## Testing

- Added `test_get_files_includes_readme_with_custom_patterns` — verifies README.md is included even with custom patterns
- Added `test_get_document_tree_root_docs` — verifies root-level files land in `root_docs` category
- Existing `test_get_document_tree` updated to assert `root_docs` key exists
- All 149 tests pass, 83% coverage
