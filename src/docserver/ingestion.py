"""
Ingestion layer for the documentation MCP server.

Watches git repositories, parses markdown files, chunks them, and upserts
them into the knowledge base. A background APScheduler job runs the cycle
on the configured poll interval.
"""

from __future__ import annotations

import logging
import os
import re
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
            logger.warning("Repo path does not exist yet: %s", root)
            return []

        matched: list[Path] = []
        patterns = self.source.glob_patterns or ["**/*.md"]
        for pattern in patterns:
            for p in root.glob(pattern):
                if p.is_file() and p not in matched:
                    matched.append(p)
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
            Repo.clone_from(self.source.path, repo_path, branch=branch)
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
                "Remote clone path '%s' exists but is not a git repo; skipping sync.",
                repo_path,
            )
            return False

        try:
            origin = repo.remotes.origin
            fetch_infos = origin.pull()
            changed = any(fi.flags & fi.NEW_HEAD for fi in fetch_infos)
            if changed:
                logger.info("Pulled changes for remote repo '%s'.", self.source.name)
            else:
                logger.debug("No changes for remote repo '%s'.", self.source.name)
            return changed
        except Exception:
            logger.exception(
                "Failed to pull remote repo '%s'; continuing with stale data.",
                self.source.name,
            )
            return False
        finally:
            repo.close()

    def _sync_local(self) -> bool:
        repo_path = Path(self.source.path)
        if not repo_path.exists():
            logger.warning("Local source path '%s' does not exist; skipping.", repo_path)
            return False

        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            # Plain directory — nothing to pull; not an error.
            logger.debug(
                "Local source '%s' is not a git repo; treating as static directory.",
                self.source.name,
            )
            return False

        try:
            if not repo.remotes:
                logger.debug("Local repo '%s' has no remotes; skipping pull.", self.source.name)
                return False
            fetch_infos = repo.remotes.origin.pull()
            changed = any(fi.flags & fi.NEW_HEAD for fi in fetch_infos)
            if changed:
                logger.info("Pulled changes for local repo '%s'.", self.source.name)
            return changed
        except Exception:
            logger.exception(
                "Failed to pull local repo '%s'; continuing with stale data.",
                self.source.name,
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
                logger.exception(
                    "Unexpected error syncing source '%s'; skipping.",
                    source.name,
                    extra={"event": "sync_error", "source": source.name},
                )
                source_stats["errors"] += 1
                continue

            repo_root = manager.get_repo_path()

            # 2. Enumerate files
            try:
                files = manager.get_files()
            except Exception:
                logger.exception(
                    "Error listing files for source '%s'; skipping.",
                    source.name,
                    extra={"event": "ingestion_error", "source": source.name},
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
                    "No files matched for source '%s' at %s — check glob patterns",
                    source.name,
                    repo_root,
                    extra={"event": "no_files", "source": source.name},
                )

            # Track which doc_ids we write this cycle so we can prune stale ones.
            seen_doc_ids: set[str] = set()

            # Look up previously indexed modification times to skip unchanged files.
            indexed_mtimes = self.kb.get_indexed_modified_times(source.name)
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

                # Skip files that haven't changed since last indexing.
                prev_mtime = indexed_mtimes.get(base_doc_id)
                current_mtime = base_metadata.get("modified_at")
                if prev_mtime and current_mtime and prev_mtime == current_mtime:
                    # Mark all existing IDs as seen so they aren't pruned.
                    seen_doc_ids.add(base_doc_id)
                    for eid in all_existing_ids:
                        if eid.startswith(base_doc_id):
                            seen_doc_ids.add(eid)
                    skipped += 1
                    continue

                # Determine if this is a new or modified file.
                is_new = base_doc_id not in indexed_mtimes
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

                # Store a parent index document (no content body) so that
                # structured queries (e.g. "when was X created") work.
                parent_metadata = {
                    **base_metadata,
                    "total_chunks": total_chunks,
                    "is_chunk": False,
                }
                try:
                    self._kb_upsert(base_doc_id, "", parent_metadata, source_stats)
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
