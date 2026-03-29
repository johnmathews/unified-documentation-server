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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        # Always include root-level README files even when custom patterns
        # are specified — README.md is the canonical project overview.
        for name in ("README.md", "readme.md", "Readme.md"):
            readme = root / name
            if not readme.is_file():
                continue
            # Check if this file is already matched (handles case-insensitive FS)
            already = any(readme.samefile(p) for p in matched if p.is_file())
            if not already:
                matched.append(readme)
                logger.debug(
                    "Source '%s': auto-included root-level '%s'",
                    self.source.name,
                    name,
                )
                break  # Only include one README variant

        # Always include these directories even when custom patterns are
        # specified — they contain important project documentation.
        auto_include_dirs = [
            (".engineering-team", "engineering analysis docs"),
            ("documentation", "project documentation"),
        ]
        for dir_name, description in auto_include_dirs:
            auto_dir = root / dir_name
            if auto_dir.is_dir():
                for md_file in auto_dir.glob("**/*.md"):
                    if md_file.is_file():
                        already = any(md_file.samefile(p) for p in matched if p.is_file())
                        if not already:
                            matched.append(md_file)
                if any(p.is_relative_to(auto_dir) for p in matched):
                    logger.debug(
                        "Source '%s': auto-included %s/ markdown files (%s)",
                        self.source.name,
                        dir_name,
                        description,
                    )

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

            # Ensure the clone's origin URL matches the current config.
            # The URL can drift if the user rotates tokens or changes the
            # source path in sources.yaml while the clone persists on a
            # Docker volume.
            current_url = origin.url
            if current_url != self.source.path:
                display_old = re.sub(r"://[^@]+@", "://<redacted>@", current_url)
                display_new = re.sub(r"://[^@]+@", "://<redacted>@", self.source.path)
                logger.info(
                    "Updating origin URL for '%s': %s -> %s",
                    self.source.name,
                    display_old,
                    display_new,
                    extra={"event": "origin_url_update", "source": self.source.name},
                )
                origin.set_url(self.source.path)

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

            fetch_info = origin.fetch()
            fetch_summary = []
            for fi in fetch_info:
                flag_names = []
                if fi.flags & fi.NEW_HEAD:
                    flag_names.append("NEW_HEAD")
                if fi.flags & fi.FAST_FORWARD:
                    flag_names.append("FAST_FORWARD")
                if fi.flags & fi.NEW_TAG:
                    flag_names.append("NEW_TAG")
                if fi.flags & fi.HEAD_UPTODATE:
                    flag_names.append("HEAD_UPTODATE")
                fetch_summary.append(f"{fi.ref}[{','.join(flag_names) or 'flags=' + str(fi.flags)}]")
            if fetch_summary:
                logger.info(
                    "Fetch results for '%s': %s",
                    self.source.name,
                    "; ".join(fetch_summary),
                    extra={"event": "fetch_info", "source": self.source.name},
                )

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
                logger.info(
                    "No new commits for remote repo '%s' (HEAD=%s).",
                    self.source.name,
                    old_head[:8],
                    extra={"event": "sync_unchanged", "source": self.source.name, "head": old_head[:8]},
                )
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

    # File extensions that are stored as raw binary (no text extraction or chunking).
    BINARY_EXTENSIONS = frozenset({".pdf"})

    def parse_binary(
        self,
        file_path: Path,
        source_name: str,
        repo_root: Path,
        created_at: str | None = None,
    ) -> ParsedDocument:
        """Parse a binary file (e.g. PDF) — metadata only, no text content.

        The file is indexed with empty content so it appears in the sidebar
        tree but is not searchable via vector search. The raw file is served
        separately by the ``/api/files/`` endpoint.
        """
        relative = file_path.relative_to(repo_root)
        doc_id = f"{source_name}:{relative}"

        stat = file_path.stat()
        file_size = stat.st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File {file_path} is {file_size} bytes, exceeds {MAX_FILE_SIZE} byte limit"
            )

        title = file_path.stem
        if created_at is None:
            created_at = self._git_created_at(file_path, repo_root)
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

        return {
            "doc_id": doc_id,
            "content": "",
            "metadata": {
                "source": source_name,
                "file_path": str(relative),
                "title": title,
                "created_at": created_at,
                "modified_at": modified_at,
                "size_bytes": file_size,
            },
        }

    def parse_markdown(
        self,
        file_path: Path,
        source_name: str,
        repo_root: Path,
        created_at: str | None = None,
    ) -> ParsedDocument:
        """Parse *file_path* and return a document dict.

        Keys returned:
          - doc_id        "{source_name}:{relative_path}"
          - content       full text of the file
          - metadata      dict with: source, file_path, title, created_at,
                          modified_at, size_bytes

        If *created_at* is provided, it is used directly instead of running
        ``git log`` per file. Pass pre-computed values from
        :meth:`_bulk_git_created_at` for better performance.
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
        if created_at is None:
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

    @staticmethod
    def _bulk_git_created_at(file_paths: list[Path], repo_root: Path) -> dict[Path, str | None]:
        """Return {file_path: ISO-8601 timestamp} for all files in one git call.

        Uses ``git log --diff-filter=A --name-only`` to get creation dates for
        all files in the repo with a single subprocess call, then matches against
        the requested file paths. Falls back to per-file lookup for files not
        found in the bulk result (e.g. renamed files that need --follow).
        """
        result: dict[Path, str | None] = {}

        try:
            proc = subprocess.run(
                [
                    "git",
                    "log",
                    "--diff-filter=A",
                    "--format=%aI",
                    "--name-only",
                    "--reverse",
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse output: alternating date lines and filename lines
            # Format: date\n\nfile1\nfile2\n\ndate\n\nfile3\n...
            created_dates: dict[str, str] = {}  # relative path -> date
            current_date: str | None = None
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                # ISO dates start with a digit (e.g. 2024-...)
                if stripped[0].isdigit() and "T" in stripped:
                    current_date = stripped
                elif current_date and stripped not in created_dates:
                    # Only record the first (earliest) date for each file
                    created_dates[stripped] = current_date
        except Exception:
            logger.debug("Bulk git created_at failed; will fall back to per-file.", exc_info=True)
            created_dates = {}

        # Match requested files against bulk results
        missing: list[Path] = []
        for fp in file_paths:
            try:
                rel = str(fp.relative_to(repo_root))
            except ValueError:
                rel = str(fp)
            if rel in created_dates:
                result[fp] = created_dates[rel]
            else:
                missing.append(fp)

        # Fall back to per-file for any misses (renamed files, etc.)
        for fp in missing:
            result[fp] = DocumentParser._git_created_at(fp, repo_root)

        return result


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

    def run_once(self, sources: list[str] | None = None, *, force: bool = False) -> dict[str, dict[str, int]]:
        """Run a full ingestion cycle across configured sources.

        Args:
            sources: Optional list of source names to restrict ingestion to.
                     If None or empty, all configured sources are ingested.
            force: If True, re-index all files regardless of content hash.

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

        # 1. Sync all sources in parallel (git fetch/pull is I/O-bound).
        sync_results: dict[str, bool | None] = {}  # name -> changed (None = failed)

        def _sync_source(source: RepoSource) -> tuple[str, bool | None]:
            manager = self._managers.get(source.name)
            if manager is None:
                logger.error(
                    "No manager found for source '%s'; skipping.",
                    source.name,
                    extra={"event": "ingestion_error", "source": source.name},
                )
                return source.name, None

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
                return source.name, changed
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
                return source.name, None

        if len(targets) > 1:
            with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as pool:
                futures = {pool.submit(_sync_source, s): s for s in targets}
                for future in as_completed(futures):
                    name, changed = future.result()
                    sync_results[name] = changed
        else:
            for s in targets:
                name, changed = _sync_source(s)
                sync_results[name] = changed

        for source in targets:
            source_stats = {"upserted": 0, "deleted": 0, "skipped": 0, "new": 0, "modified": 0, "files": 0, "errors": 0}
            stats[source.name] = source_stats

            # Skip sources that failed to sync.
            if sync_results.get(source.name) is None:
                source_stats["errors"] += 1
                continue

            manager = self._managers.get(source.name)
            if manager is None:
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

            # Bulk-fetch git creation dates for all files in one subprocess call.
            git_dates = DocumentParser._bulk_git_created_at(files, repo_root) if files else {}

            # Track which doc_ids we write this cycle so we can prune stale ones.
            seen_doc_ids: set[str] = set()

            # Look up previously indexed content hashes to skip unchanged files.
            indexed_hashes = self.kb.get_indexed_content_hashes(source.name)
            all_existing_ids = self.kb.get_all_doc_ids_for_source(source.name)
            skipped = 0
            total_files = len(files)
            processed = 0

            # 3 & 4. Parse, chunk, and batch-upsert
            # Collect all items for batch upsert to leverage ChromaDB's embedding batching.
            upsert_batch: list[tuple[str, str, dict]] = []
            BATCH_FLUSH_SIZE = 64  # flush every N items to bound memory
            total_upserted = 0

            def _flush_batch(
                _source: RepoSource = source,
                _stats: dict[str, int] = source_stats,
            ) -> None:
                """Flush the current upsert batch to the KB."""
                nonlocal upsert_batch, total_upserted
                if not upsert_batch:
                    return
                batch_size = len(upsert_batch)
                chunk_count = sum(1 for _, _, m in upsert_batch if m.get("is_chunk"))
                try:
                    self.kb.upsert_documents_batch(upsert_batch)
                    _stats["upserted"] += batch_size
                    total_upserted += batch_size
                    logger.info(
                        "Flushed batch for '%s': %d items (%d chunks), %d total upserted so far",
                        _source.name,
                        batch_size,
                        chunk_count,
                        total_upserted,
                        extra={
                            "event": "batch_flush",
                            "source": _source.name,
                            "batch_size": batch_size,
                            "chunk_count": chunk_count,
                            "total_upserted": total_upserted,
                        },
                    )
                except Exception:
                    logger.exception(
                        "Batch upsert failed for source '%s' (%d items); falling back to individual upserts.",
                        _source.name,
                        batch_size,
                        extra={"event": "batch_upsert_error", "source": _source.name},
                    )
                    for item_id, item_content, item_meta in upsert_batch:
                        try:
                            self.kb.upsert_document(item_id, item_content, item_meta)
                            _stats["upserted"] += 1
                            total_upserted += 1
                        except Exception:
                            logger.exception("Failed to upsert '%s'.", item_id)
                            _stats["errors"] += 1
                upsert_batch = []

            for file_idx, file_path in enumerate(files, 1):
                is_binary = file_path.suffix.lower() in DocumentParser.BINARY_EXTENSIONS

                try:
                    if is_binary:
                        doc = self._parser.parse_binary(
                            file_path, source.name, repo_root,
                            created_at=git_dates.get(file_path),
                        )
                    else:
                        doc = self._parser.parse_markdown(
                            file_path, source.name, repo_root,
                            created_at=git_dates.get(file_path),
                        )
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
                # For binary files, hash the raw bytes; for text, hash the content string.
                if is_binary:
                    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                else:
                    content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Skip files whose content hasn't changed since last indexing.
                prev_hash = indexed_hashes.get(base_doc_id)
                if not force and prev_hash and prev_hash == content_hash:
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

                # Binary files are stored as metadata only — no chunking.
                if is_binary:
                    chunks = []
                    total_chunks = 0
                else:
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

                # Queue parent document.
                parent_metadata = {
                    **base_metadata,
                    "total_chunks": total_chunks,
                    "is_chunk": False,
                    "content_hash": content_hash,
                }
                upsert_batch.append((base_doc_id, content, parent_metadata))
                seen_doc_ids.add(base_doc_id)
                source_stats[change_type] += 1

                # Queue each chunk.
                for idx, chunk in enumerate(chunks):
                    chunk_doc_id = f"{base_doc_id}#chunk{idx}"
                    chunk_metadata = {
                        **base_metadata,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "is_chunk": True,
                        "section_path": chunk.section_path,
                    }
                    upsert_batch.append((chunk_doc_id, chunk.text, chunk_metadata))
                    seen_doc_ids.add(chunk_doc_id)

                # Flush batch periodically to bound memory usage.
                if len(upsert_batch) >= BATCH_FLUSH_SIZE:
                    _flush_batch()

            # Flush remaining items.
            _flush_batch()

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

