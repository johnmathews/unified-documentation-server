"""
Ingestion layer for the documentation MCP server.

Watches git repositories, parses markdown files, chunks them, and upserts
them into the knowledge base. A background APScheduler job runs the cycle
on the configured poll interval.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from apscheduler.schedulers.background import BackgroundScheduler
from git import InvalidGitRepositoryError, Repo

if TYPE_CHECKING:
    from docserver.config import Config, RepoSource
    from docserver.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class DocumentMetadata(TypedDict, total=False):
    source: str
    file_path: str
    title: str
    created_at: str | None
    modified_at: str
    size_bytes: int
    chunk_index: int
    total_chunks: int
    is_chunk: bool
    section_path: str


class ParsedDocument(TypedDict):
    doc_id: str
    content: str
    metadata: DocumentMetadata


class Section(TypedDict):
    heading_path: str
    blocks: list[str]


# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------
CHUNK_TARGET_SIZE = (
    400  # characters — documentation is terse, smaller chunks give more precise search
)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB — skip files larger than this to avoid OOM


# ---------------------------------------------------------------------------
# RepoManager
# ---------------------------------------------------------------------------


class RepoManager:
    """Manages a single repository source: cloning, pulling, and listing files."""

    def __init__(self, source: RepoSource, clone_dir: str) -> None:
        self.source = source
        self.clone_dir = Path(clone_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sync(self) -> bool:
        """Pull the latest changes for this source.

        For remote repos, clone if the directory does not exist, then pull.
        For local repos that are git repos, run git pull. For plain
        directories, do nothing.

        Returns True if there were any changes (new commits or first clone).
        """
        if self.source.is_remote:
            return self._sync_remote()
        return self._sync_local()

    def get_files(self) -> list[Path]:
        """Return all files under the repo root that match the configured glob patterns."""
        root = self.get_repo_path()
        if not root.exists():
            logger.warning(
                "Repo path does not exist: '%s' for source '%s'. "
                "If this is a remote source, the initial clone may have failed. "
                "If this is a local source, check that the path is correct and the "
                "directory is mounted/accessible.",
                root,
                self.source.name,
                extra={"event": "repo_path_missing", "source": self.source.name, "path": str(root)},
            )
            return []

        if not root.is_dir():
            logger.error(
                "Repo path '%s' for source '%s' exists but is not a directory "
                "(it is a %s). Expected a directory containing documentation files.",
                root,
                self.source.name,
                "symlink" if root.is_symlink() else "file",
                extra={"event": "repo_path_not_dir", "source": self.source.name, "path": str(root)},
            )
            return []

        matched: list[Path] = []
        patterns = self.source.glob_patterns or ["**/*.md"]
        for pattern in patterns:
            pattern_matches = [p for p in root.glob(pattern) if p.is_file()]
            logger.debug(
                "Source '%s': glob pattern '%s' matched %d file(s) under '%s'",
                self.source.name,
                pattern,
                len(pattern_matches),
                root,
            )
            for p in pattern_matches:
                if p not in matched:
                    matched.append(p)

        if not matched:
            # Provide detailed diagnostics when no files match
            try:
                top_level_entries = sorted(root.iterdir())
                top_level_names = [
                    f"{'[dir] ' if e.is_dir() else ''}{e.name}" for e in top_level_entries[:30]
                ]
                if len(top_level_entries) > 30:
                    top_level_names.append(f"... and {len(top_level_entries) - 30} more")
            except PermissionError:
                top_level_names = ["<permission denied — cannot list directory>"]

            # Check for common docs directory names
            common_doc_dirs = ["docs", "doc", "documentation", "wiki", "content", "pages"]
            found_doc_dirs = [d for d in common_doc_dirs if (root / d).is_dir()]

            logger.warning(
                "No files matched for source '%s'. Searched directory: '%s'. "
                "Glob patterns tried: %s. "
                "Top-level contents of '%s': [%s]. "
                "%s"
                "Possible fixes: (1) Check that the glob patterns match the actual file layout, "
                "(2) Verify the repo was cloned/mounted correctly, "
                "(3) If docs are in a subdirectory, use a pattern like 'docs/**/*.md'.",
                self.source.name,
                root,
                patterns,
                root,
                ", ".join(top_level_names) if top_level_names else "<empty directory>",
                f"Found potential documentation directories: {found_doc_dirs}. "
                f"Consider updating patterns to include them (e.g. '{found_doc_dirs[0]}/**/*.md'). "
                if found_doc_dirs else
                "No common documentation directories (docs/, doc/, wiki/, etc.) found at the repo root. ",
                extra={
                    "event": "no_files_matched",
                    "source": self.source.name,
                    "path": str(root),
                    "patterns": patterns,
                    "top_level_contents": top_level_names,
                    "found_doc_dirs": found_doc_dirs,
                },
            )

        return matched

    def get_repo_path(self) -> Path:
        """Return the local filesystem path to the repo content."""
        if self.source.is_remote:
            # Remote repos are cloned into <clone_dir>/<source_name>
            return self.clone_dir / self.source.name
        return Path(self.source.path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_remote(self) -> bool:
        repo_path = self.get_repo_path()
        if not repo_path.exists():
            # Redact credentials from clone URL for logging
            display_url = re.sub(r"://[^@]+@", "://<redacted>@", self.source.path)
            logger.info(
                "Cloning remote repo '%s' from %s (branch: %s) into %s",
                self.source.name,
                display_url,
                self.source.branch,
                repo_path,
                extra={"event": "clone_start", "source": self.source.name},
            )
            repo_path.mkdir(parents=True, exist_ok=True)
            branch = self.source.branch or "main"
            try:
                Repo.clone_from(self.source.path, repo_path, branch=branch)
            except Exception:
                logger.exception(
                    "Failed to clone remote repo '%s' from %s (branch: %s) into '%s'. "
                    "Possible causes: (1) the repository URL is incorrect or inaccessible, "
                    "(2) credentials in the URL are invalid or expired, "
                    "(3) the branch '%s' does not exist in the remote repository, "
                    "(4) network connectivity issues (DNS resolution, firewall, proxy), "
                    "(5) insufficient disk space at '%s'. "
                    "The empty clone directory will be removed to allow a retry on the next cycle.",
                    self.source.name,
                    display_url,
                    branch,
                    repo_path,
                    branch,
                    repo_path,
                    extra={
                        "event": "clone_error",
                        "source": self.source.name,
                        "branch": branch,
                        "path": str(repo_path),
                    },
                )
                # Remove the empty directory so next cycle retries the clone
                shutil.rmtree(repo_path, ignore_errors=True)
                return False
            logger.info(
                "Clone complete for '%s'",
                self.source.name,
                extra={"event": "clone_done", "source": self.source.name},
            )
            return True

        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            logger.error(
                "Remote clone directory '%s' for source '%s' exists but is not a valid "
                "git repository. This can happen if a previous clone was interrupted or "
                "the directory was corrupted. Try deleting '%s' and restarting the server "
                "to trigger a fresh clone.",
                repo_path,
                self.source.name,
                repo_path,
                extra={"event": "invalid_clone", "source": self.source.name, "path": str(repo_path)},
            )
            return False

        try:
            branch = self.source.branch or "main"
            origin = repo.remotes.origin

            # Read current HEAD for change detection. This can fail if the repo
            # is in a corrupt state (e.g. invalid branch refs). In that case,
            # delete the clone and re-clone fresh — corrupt refs persist through
            # fetch+reset and GitPython cannot operate on them.
            try:
                old_head = repo.head.commit.hexsha
            except (ValueError, TypeError):
                display_url = re.sub(r"://[^@]+@", "://<redacted>@", self.source.path)
                logger.warning(
                    "Remote repo '%s' at '%s' has corrupt git references and cannot be "
                    "read. Deleting the clone and re-cloning from %s to recover.",
                    self.source.name,
                    repo_path,
                    display_url,
                    extra={"event": "corrupt_head", "source": self.source.name, "path": str(repo_path)},
                )
                repo.close()
                shutil.rmtree(repo_path, ignore_errors=True)
                repo_path.mkdir(parents=True, exist_ok=True)
                Repo.clone_from(self.source.path, repo_path, branch=branch)
                logger.info(
                    "Re-clone complete for '%s' — repo recovered.",
                    self.source.name,
                    extra={"event": "clone_done", "source": self.source.name},
                )
                return True

            origin.fetch()
            repo.head.reset(f"origin/{branch}", index=True, working_tree=True)
            new_head = repo.head.commit.hexsha
            changed = old_head != new_head
            if changed:
                logger.info(
                    "Fetched and reset remote repo '%s' to origin/%s (%s -> %s).",
                    self.source.name,
                    branch,
                    old_head[:8],
                    new_head[:8],
                )
            else:
                logger.debug("No changes for remote repo '%s'.", self.source.name)
            return changed
        except Exception:
            display_url = re.sub(r"://[^@]+@", "://<redacted>@", self.source.path)
            logger.exception(
                "Failed to fetch remote repo '%s' (url: %s, branch: %s, clone dir: '%s'). "
                "This could be due to: network connectivity issues, invalid or expired "
                "credentials in the repo URL, the branch no longer existing, or the "
                "remote server being unavailable. Continuing with stale data.",
                self.source.name,
                display_url,
                self.source.branch,
                repo_path,
                extra={
                    "event": "fetch_error",
                    "source": self.source.name,
                    "branch": self.source.branch,
                    "path": str(repo_path),
                },
            )
            return False
        finally:
            repo.close()

    def _sync_local(self) -> bool:
        repo_path = Path(self.source.path)
        if not repo_path.exists():
            # Check parent to give better guidance
            parent = repo_path.parent
            parent_exists = parent.exists()
            logger.warning(
                "Local source '%s' path does not exist: '%s'. "
                "Parent directory '%s' %s. "
                "Check that: (1) the path in sources.yaml is correct, "
                "(2) the directory is mounted into the container (if running in Docker), "
                "(3) the volume mount path matches the configured path.",
                self.source.name,
                repo_path,
                parent,
                "exists" if parent_exists else "also does not exist — the mount may be missing entirely",
                extra={
                    "event": "local_path_missing",
                    "source": self.source.name,
                    "path": str(repo_path),
                    "parent_exists": parent_exists,
                },
            )
            return False

        if not repo_path.is_dir():
            logger.error(
                "Local source '%s' path '%s' exists but is not a directory. "
                "Expected a directory containing documentation files.",
                self.source.name,
                repo_path,
                extra={"event": "local_path_not_dir", "source": self.source.name, "path": str(repo_path)},
            )
            return False

        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            # Plain directory — nothing to pull; not an error.
            logger.debug(
                "Local source '%s' at '%s' is not a git repo; treating as static directory. "
                "Files will be read directly without git-based change detection.",
                self.source.name,
                repo_path,
            )
            return False

        try:
            if not repo.remotes:
                logger.debug(
                    "Local repo '%s' at '%s' has no remotes configured; skipping fetch. "
                    "Files will be read from the current working tree.",
                    self.source.name,
                    repo_path,
                )
                return False
            branch = self.source.branch or "main"
            try:
                old_head = repo.head.commit.hexsha
            except (ValueError, TypeError):
                logger.warning(
                    "Local repo '%s' at '%s' has corrupt git references. "
                    "Attempting to recover by checking out origin/%s.",
                    self.source.name,
                    repo_path,
                    branch,
                    extra={"event": "corrupt_head", "source": self.source.name, "path": str(repo_path)},
                )
                old_head = None

            repo.remotes.origin.fetch()

            if old_head is None:
                # HEAD is corrupt — use subprocess to fix it since GitPython
                # cannot operate on repos with invalid ref names.
                subprocess.run(
                    ["git", "checkout", "-B", branch, f"origin/{branch}"],
                    cwd=str(repo_path),
                    capture_output=True,
                    check=True,
                )
                logger.info(
                    "Recovered local repo '%s' by checking out origin/%s.",
                    self.source.name,
                    branch,
                    extra={"event": "corrupt_head_recovered", "source": self.source.name},
                )
                return True

            repo.head.reset(f"origin/{branch}", index=True, working_tree=True)
            new_head = repo.head.commit.hexsha
            changed = old_head != new_head
            if changed:
                logger.info(
                    "Fetched and reset local repo '%s' to origin/%s (%s -> %s).",
                    self.source.name,
                    branch,
                    old_head[:8],
                    new_head[:8],
                )
            return changed
        except Exception:
            logger.exception(
                "Failed to fetch local repo '%s' at '%s'. "
                "The repo exists and has remotes, but the fetch/reset operation failed. "
                "This could be due to: network issues, authentication problems, "
                "or a corrupted git state. "
                "Continuing with the existing (possibly stale) data.",
                self.source.name,
                repo_path,
                extra={"event": "fetch_error", "source": self.source.name, "path": str(repo_path)},
            )
            return False
        finally:
            repo.close()


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------


class DocumentParser:
    """Parses individual markdown files into document dicts ready for the KB."""

    def parse_markdown(self, file_path: Path, source_name: str, repo_root: Path) -> ParsedDocument:
        """Parse *file_path* and return a document dict.

        Keys returned:
          - doc_id        "{source_name}:{relative_path}"
          - content       full text of the file
          - metadata      dict with: source, file_path, title, created_at,
                          modified_at, size_bytes
        """
        relative = file_path.relative_to(repo_root)
        doc_id = f"{source_name}:{relative}"

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File {file_path} is {file_size} bytes, exceeds {MAX_FILE_SIZE} byte limit"
            )

        content = file_path.read_text(encoding="utf-8", errors="replace")

        title = self._extract_title(content, file_path)
        created_at = self._git_created_at(file_path, repo_root)
        modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC).isoformat()
        size_bytes = file_path.stat().st_size

        return {
            "doc_id": doc_id,
            "content": content,
            "metadata": {
                "source": source_name,
                "file_path": str(relative),
                "title": title,
                "created_at": created_at,
                "modified_at": modified_at,
                "size_bytes": size_bytes,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(content: str, file_path: Path) -> str:
        """Return the first ATX heading, or fall back to the filename stem."""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return file_path.stem

    @staticmethod
    def _git_created_at(file_path: Path, repo_root: Path) -> str | None:
        """Return the ISO-8601 timestamp of the commit that first added the file.

        Uses ``git log --follow --diff-filter=A`` so renames are tracked.
        Returns None if git is unavailable or the file is untracked.
        """
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--follow",
                    "--diff-filter=A",
                    "--format=%aI",
                    "--",
                    str(file_path),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip()
            # git log returns most-recent first; with --diff-filter=A there
            # should be at most one line, but take the last just in case.
            lines = [line for line in output.splitlines() if line.strip()]
            return lines[-1] if lines else None
        except Exception:
            logger.debug("Could not determine git creation time for %s.", file_path, exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_LIST_ITEM_RE = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s")

OVERLAP_SIZE = 100  # characters of overlap between consecutive chunks


def _normalise_repo_url(url: str) -> str:
    """Normalise a git repo URL for comparison.

    Strips credentials, trailing .git, trailing slashes, and lowercases
    the host so that ``https://token@github.com/Foo/Bar.git`` and
    ``https://github.com/foo/bar`` compare equal.
    """
    # Strip credentials (anything between :// and @)
    url = re.sub(r"://[^@]+@", "://", url)
    # Strip trailing .git and slashes
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Lowercase for case-insensitive host comparison
    return url.lower()


@dataclass
class Chunk:
    """A chunk of document text with its section context."""

    text: str
    section_path: str  # e.g. "Setup > Ports > Firewall Rules"


def _parse_sections(content: str) -> list[Section]:
    """Parse markdown into a list of sections, each with heading context.

    Returns a list of dicts: {heading_path: str, blocks: list[str]}
    where each block is a contiguous unit (paragraph, list, code fence, etc.)
    """
    lines = content.split("\n")
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    sections: list[Section] = []
    current_blocks: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False

    def _flush_lines() -> None:
        if current_lines:
            block = "\n".join(current_lines).strip()
            if block:
                current_blocks.append(block)
            current_lines.clear()

    def _heading_path() -> str:
        return " > ".join(text for _, text in heading_stack)

    for line in lines:
        # Track code fences — don't split inside them
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_fence:
                # End of code fence — include closing ``` in current block
                current_lines.append(line)
                _flush_lines()
                in_code_fence = False
                continue
            else:
                _flush_lines()
                in_code_fence = True
                current_lines.append(line)
                continue

        if in_code_fence:
            current_lines.append(line)
            continue

        # Check for heading
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            _flush_lines()
            # Save current section if it has content
            if current_blocks:
                sections.append(
                    {
                        "heading_path": _heading_path(),
                        "blocks": current_blocks,
                    }
                )
                current_blocks = []

            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()

            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            continue

        # Blank line = paragraph boundary (but keep list items together)
        if not stripped:
            is_list = current_lines and _LIST_ITEM_RE.match(current_lines[0])
            if not is_list:
                _flush_lines()
            else:
                current_lines.append(line)
            continue

        current_lines.append(line)

    # Flush remaining
    _flush_lines()
    if current_blocks:
        sections.append(
            {
                "heading_path": _heading_path(),
                "blocks": current_blocks,
            }
        )

    return sections


def _chunk_content(
    content: str,
    target_size: int = CHUNK_TARGET_SIZE,
    overlap_size: int = OVERLAP_SIZE,
) -> list[Chunk]:
    """Split markdown content into chunks with section context and overlap.

    Strategy:
      1. Parse into sections based on headings.
      2. Within each section, group blocks (paragraphs, lists, code fences)
         into chunks of ~target_size characters.
      3. Prepend each chunk with its heading path for context.
      4. Add overlap from the end of the previous chunk.
    """
    sections = _parse_sections(content)

    if not sections:
        return [Chunk(text=content, section_path="")]

    chunks: list[Chunk] = []
    prev_tail = ""  # last N chars of previous chunk for overlap

    def _emit(parts: list[str], heading: str) -> None:
        nonlocal prev_tail
        body = "\n\n".join(parts)

        # Add overlap from previous chunk
        if prev_tail and chunks:
            body = f"[...]{prev_tail}\n\n{body}"

        # Prepend section context
        text = f"[{heading}]\n\n{body}" if heading else body

        chunks.append(Chunk(text=text, section_path=heading))

        # Save tail for next chunk's overlap
        prev_tail = body[-overlap_size:] if len(body) > overlap_size else body

    for section in sections:
        heading_path = section["heading_path"]
        current_parts: list[str] = []
        current_size = 0

        for block in section["blocks"]:
            block_size = len(block)

            if current_parts and current_size + 2 + block_size > target_size:
                _emit(current_parts, heading_path)
                current_parts = [block]
                current_size = block_size
            else:
                current_parts.append(block)
                current_size += (2 if current_parts else 0) + block_size

        if current_parts:
            _emit(current_parts, heading_path)

    return chunks or [Chunk(text=content, section_path="")]


# ---------------------------------------------------------------------------
# Ingester
# ---------------------------------------------------------------------------


class Ingester:
    """Orchestrates ingestion across all configured sources."""

    def __init__(self, config: Config, kb: KnowledgeBase) -> None:
        self.config = config
        self.kb = kb
        self._parser = DocumentParser()
        self._scheduler = BackgroundScheduler()
        self._managers: dict[str, RepoManager] = {}
        self._build_managers()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def cleanup_orphaned_sources(self) -> dict[str, int]:
        """Remove KB entries and clone dirs for sources no longer in the config.

        Before deleting, checks if an orphaned source was renamed by matching
        git remote URLs (for remote sources) or filesystem paths (for local
        sources). Renamed sources are migrated in-place to avoid expensive
        re-cloning and re-embedding.

        Returns a dict of {source_name: docs_deleted_or_migrated} for each
        orphaned source.
        """
        configured_names = {s.name for s in self.config.sources}
        kb_names = self.kb.get_all_source_names()
        orphaned = kb_names - configured_names

        if not orphaned:
            return {}

        # Build a lookup from normalised URL/path → configured source for rename detection.
        configured_by_url: dict[str, RepoSource] = {}
        for src in self.config.sources:
            key = _normalise_repo_url(src.path)
            configured_by_url[key] = src

        clone_dir = Path(self.config.data_dir) / "clones"
        result: dict[str, int] = {}

        for name in orphaned:
            new_source = self._detect_rename(name, clone_dir, configured_by_url)
            if new_source is not None:
                count = self._migrate_renamed_source(name, new_source, clone_dir)
                result[name] = count
            else:
                count = self.kb.delete_source_documents(name)
                logger.info(
                    "Cleaned up orphaned source '%s': deleted %d docs from KB",
                    name,
                    count,
                    extra={"event": "orphan_cleanup", "source": name, "deleted": count},
                )
                result[name] = count

        # Clean up orphaned clone directories (only those not handled by rename)
        if clone_dir.exists():
            for entry in clone_dir.iterdir():
                if entry.is_dir() and entry.name not in configured_names:
                    logger.info(
                        "Removing orphaned clone directory '%s'",
                        entry,
                        extra={"event": "orphan_cleanup_dir", "path": str(entry)},
                    )
                    shutil.rmtree(entry, ignore_errors=True)

        return result

    def _detect_rename(
        self,
        orphan_name: str,
        clone_dir: Path,
        configured_by_url: dict[str, RepoSource],
    ) -> RepoSource | None:
        """Check if an orphaned source was renamed by matching its repo URL.

        Returns the new RepoSource if a rename is detected, otherwise None.
        """
        orphan_clone = clone_dir / orphan_name
        if not orphan_clone.exists():
            return None

        try:
            repo = Repo(orphan_clone)
            remote_url = repo.remotes.origin.url
            repo.close()
        except Exception:
            return None

        key = _normalise_repo_url(remote_url)
        new_source = configured_by_url.get(key)
        if new_source is None or new_source.name == orphan_name:
            return None

        # Only match if the new name doesn't already have data in the KB
        # (i.e. it's genuinely a rename, not a URL collision with an existing source)
        existing_new = self.kb.get_all_doc_ids_for_source(new_source.name)
        if existing_new:
            return None

        logger.info(
            "Detected source rename: '%s' → '%s' (same repo URL: %s)",
            orphan_name,
            new_source.name,
            key,
            extra={
                "event": "rename_detected",
                "old_name": orphan_name,
                "new_name": new_source.name,
            },
        )
        return new_source

    def _migrate_renamed_source(
        self,
        old_name: str,
        new_source: RepoSource,
        clone_dir: Path,
    ) -> int:
        """Migrate an orphaned source to its new name: rename KB data and clone dir."""
        count = self.kb.rename_source(old_name, new_source.name)

        # Rename the clone directory so the repo isn't re-cloned
        old_clone = clone_dir / old_name
        new_clone = clone_dir / new_source.name
        if old_clone.exists() and not new_clone.exists():
            old_clone.rename(new_clone)
            logger.info(
                "Renamed clone directory '%s' → '%s'",
                old_clone,
                new_clone,
                extra={
                    "event": "rename_clone_dir",
                    "old_path": str(old_clone),
                    "new_path": str(new_clone),
                },
            )

        return count

    def run_once(self, sources: list[str] | None = None) -> dict[str, dict[str, int]]:
        """Run a full ingestion cycle across configured sources.

        Args:
            sources: Optional list of source names to restrict ingestion to.
                     If None or empty, all configured sources are ingested.

        For each source:
          1. Sync the repo (clone/pull).
          2. Enumerate matching files.
          3. Parse each file and split into chunks.
          4. Upsert every chunk (and a parent index doc) into the KB.
          5. Delete KB documents for files that no longer exist in the repo.

        Returns a dict keyed by source name with ``{upserted, deleted}``
        counts.
        """
        # Clean up sources that were removed or renamed in config.
        if not sources:
            self.cleanup_orphaned_sources()

        stats: dict[str, dict[str, int]] = {}

        targets = self.config.sources
        if sources:
            source_set = set(sources)
            targets = [s for s in targets if s.name in source_set]

        target_names = [s.name for s in targets]
        logger.info(
            "Ingestion cycle starting for %d source(s): %s",
            len(targets),
            target_names,
            extra={"event": "ingestion_start", "sources": target_names},
        )

        for source in targets:
            source_stats = {"upserted": 0, "deleted": 0, "skipped": 0, "new": 0, "modified": 0, "files": 0, "errors": 0}
            stats[source.name] = source_stats

            manager = self._managers.get(source.name)
            if manager is None:
                logger.error(
                    "No manager found for source '%s'; skipping.",
                    source.name,
                    extra={"event": "ingestion_error", "source": source.name},
                )
                continue

            # 1. Sync
            logger.info(
                "Syncing source '%s' (remote=%s)...",
                source.name,
                source.is_remote,
                extra={"event": "sync_start", "source": source.name},
            )
            try:
                changed = manager.sync()
                logger.info(
                    "Sync complete for '%s': changed=%s",
                    source.name,
                    changed,
                    extra={"event": "sync_done", "source": source.name, "changed": changed},
                )
            except Exception:
                display_path = re.sub(r"://[^@]+@", "://<redacted>@", source.path) if source.is_remote else source.path
                logger.exception(
                    "Unexpected error syncing source '%s' (path: %s, remote: %s, branch: %s). "
                    "This source will be skipped entirely for this ingestion cycle. "
                    "No documents from this source will be updated until the sync succeeds.",
                    source.name,
                    display_path,
                    source.is_remote,
                    source.branch,
                    extra={"event": "sync_error", "source": source.name, "path": display_path},
                )
                source_stats["errors"] += 1
                continue

            repo_root = manager.get_repo_path()

            # 2. Enumerate files
            try:
                files = manager.get_files()
            except Exception:
                logger.exception(
                    "Error listing files for source '%s' at path '%s' with patterns %s. "
                    "This source will be skipped entirely for this ingestion cycle. "
                    "Check that the directory exists, is readable, and that the "
                    "glob patterns are valid.",
                    source.name,
                    repo_root,
                    source.glob_patterns,
                    extra={"event": "ingestion_error", "source": source.name, "path": str(repo_root)},
                )
                source_stats["errors"] += 1
                continue

            source_stats["files"] = len(files)
            logger.info(
                "Found %d files in source '%s' (patterns: %s)",
                len(files),
                source.name,
                source.glob_patterns,
                extra={"event": "files_found", "source": source.name, "file_count": len(files)},
            )

            if not files:
                logger.warning(
                    "Source '%s' will not contribute any documents to the knowledge base "
                    "because no files were found. The repo at '%s' was synced successfully, "
                    "but none of the configured glob patterns (%s) matched any files. "
                    "See the preceding log messages for detailed diagnostics about what "
                    "was found in the repo directory.",
                    source.name,
                    repo_root,
                    source.glob_patterns,
                    extra={"event": "no_files", "source": source.name, "path": str(repo_root), "patterns": source.glob_patterns},
                )

            # Track which doc_ids we write this cycle so we can prune stale ones.
            seen_doc_ids: set[str] = set()

            # Look up previously indexed content hashes to skip unchanged files.
            indexed_hashes = self.kb.get_indexed_content_hashes(source.name)
            all_existing_ids = self.kb.get_all_doc_ids_for_source(source.name)
            skipped = 0
            total_files = len(files)
            processed = 0

            # 3 & 4. Parse and upsert
            for file_idx, file_path in enumerate(files, 1):
                try:
                    doc = self._parser.parse_markdown(file_path, source.name, repo_root)
                except Exception:
                    logger.exception(
                        "Failed to parse '%s' in source '%s'; skipping file.",
                        file_path,
                        source.name,
                        extra={"event": "parse_error", "source": source.name},
                    )
                    source_stats["errors"] += 1
                    continue

                base_doc_id: str = doc["doc_id"]
                content: str = doc["content"]
                base_metadata: DocumentMetadata = doc["metadata"]

                # Compute content hash for change detection.
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Skip files whose content hasn't changed since last indexing.
                prev_hash = indexed_hashes.get(base_doc_id)
                if prev_hash and prev_hash == content_hash:
                    # Mark all existing IDs as seen so they aren't pruned.
                    seen_doc_ids.add(base_doc_id)
                    for eid in all_existing_ids:
                        if eid.startswith(base_doc_id):
                            seen_doc_ids.add(eid)
                    skipped += 1
                    continue

                # Determine if this is a new or modified file.
                is_new = base_doc_id not in indexed_hashes
                change_type = "new" if is_new else "modified"

                chunks = _chunk_content(content)
                total_chunks = len(chunks)
                processed += 1

                logger.info(
                    "[%d/%d] Indexing %s file '%s' (%d chunks)",
                    file_idx,
                    total_files,
                    change_type,
                    base_metadata.get("file_path", base_doc_id),
                    total_chunks,
                    extra={
                        "event": "indexing_file",
                        "source": source.name,
                        "doc_id": base_doc_id,
                        "change_type": change_type,
                        "chunks": total_chunks,
                        "progress": f"{file_idx}/{total_files}",
                    },
                )

                # Store parent document with full content so the UI can serve
                # the original document.  Chunks are still stored separately
                # for vector search.
                parent_metadata = {
                    **base_metadata,
                    "total_chunks": total_chunks,
                    "is_chunk": False,
                    "content_hash": content_hash,
                }
                try:
                    self._kb_upsert(base_doc_id, content, parent_metadata, source_stats)
                    seen_doc_ids.add(base_doc_id)
                    source_stats[change_type] += 1
                except Exception:
                    logger.exception(
                        "Failed to upsert index doc '%s'; skipping file.",
                        base_doc_id,
                        extra={"event": "upsert_error", "source": source.name, "doc_id": base_doc_id},
                    )
                    source_stats["errors"] += 1
                    continue

                # Store each chunk.
                for idx, chunk in enumerate(chunks):
                    chunk_doc_id = f"{base_doc_id}#chunk{idx}"
                    chunk_metadata = {
                        **base_metadata,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "is_chunk": True,
                        "section_path": chunk.section_path,
                    }
                    try:
                        self._kb_upsert(chunk_doc_id, chunk.text, chunk_metadata, source_stats)
                        seen_doc_ids.add(chunk_doc_id)
                    except Exception:
                        logger.exception(
                            "Failed to upsert chunk '%s'; continuing.",
                            chunk_doc_id,
                            extra={"event": "upsert_error", "source": source.name, "doc_id": chunk_doc_id},
                        )
                        source_stats["errors"] += 1

            if skipped and processed:
                logger.info(
                    "Processed %d changed file(s), skipped %d unchanged for source '%s'",
                    processed,
                    skipped,
                    source.name,
                    extra={"event": "skip_summary", "source": source.name, "processed": processed, "skipped": skipped},
                )
            elif skipped and not processed:
                logger.info(
                    "All %d file(s) unchanged for source '%s', nothing to index",
                    skipped,
                    source.name,
                    extra={"event": "skip_summary", "source": source.name, "processed": 0, "skipped": skipped},
                )

            # 5. Delete stale documents
            try:
                existing_ids = self.kb.get_all_doc_ids_for_source(source.name)
                stale_ids = existing_ids - seen_doc_ids
                if stale_ids:
                    logger.info(
                        "Removing %d stale docs from source '%s'",
                        len(stale_ids),
                        source.name,
                        extra={"event": "stale_cleanup", "source": source.name, "stale_count": len(stale_ids)},
                    )
                for stale_id in stale_ids:
                    try:
                        self.kb.delete_document(stale_id)
                        source_stats["deleted"] += 1
                        logger.debug("Deleted stale doc '%s'.", stale_id)
                    except Exception:
                        logger.exception("Failed to delete stale doc '%s'.", stale_id)
                        source_stats["errors"] += 1
            except Exception:
                logger.exception(
                    "Failed to retrieve existing doc IDs for source '%s'.",
                    source.name,
                    extra={"event": "ingestion_error", "source": source.name},
                )

            source_stats["skipped"] = skipped
            logger.info(
                "Ingestion complete for source '%s': files=%d, new=%d, modified=%d, skipped=%d, deleted=%d, errors=%d",
                source.name,
                source_stats["files"],
                source_stats["new"],
                source_stats["modified"],
                source_stats["skipped"],
                source_stats["deleted"],
                source_stats["errors"],
                extra={"event": "ingestion_source_done", "source": source.name, "stats": source_stats},
            )

        logger.info(
            "Ingestion cycle finished: %s",
            {name: s for name, s in stats.items()},
            extra={"event": "ingestion_done", "stats": stats},
        )

        return stats

    def start(self) -> None:
        """Start the background scheduler that runs :meth:`run_once` periodically."""
        interval = self.config.poll_interval_seconds
        logger.info("Starting ingestion scheduler (interval=%ds).", interval)
        self._scheduler.add_job(
            self._run_once_safe,
            trigger="interval",
            seconds=interval,
            id="ingestion_job",
            replace_existing=True,
            next_run_time=datetime.now(tz=UTC),  # run immediately on start
        )
        self._scheduler.start()

    def stop(self) -> None:
        """Shut down the background scheduler gracefully."""
        logger.info("Stopping ingestion scheduler.")
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_managers(self) -> None:
        clone_dir = os.path.join(self.config.data_dir, "clones")
        for source in self.config.sources:
            self._managers[source.name] = RepoManager(source, clone_dir)

    def _run_once_safe(self) -> None:
        """Wrapper around run_once that swallows top-level exceptions so the
        scheduler job is never killed by an unhandled error."""
        try:
            self.run_once()
        except Exception:
            logger.exception("Unhandled error in ingestion cycle.")

    def _kb_upsert(
        self,
        doc_id: str,
        content: str,
        metadata: dict,
        stats: dict[str, int],
    ) -> None:
        """Upsert a document and increment the appropriate stat counter."""
        self.kb.upsert_document(doc_id, content, metadata)
        stats["upserted"] += 1
