"""One-shot ingestion worker.

Spawned by the docserver to run a single ingestion cycle as a separate OS
process. Keeping the heavy embedding work out of the request-serving
process is the primary defence against (a) Docker OOM-killing the server
when ingestion's RSS spikes, and (b) GIL contention pausing the asyncio
event loop long enough for upstream timeouts.

Memory bound: this process shares the docserver container's cgroup. When
the container's mem_limit fires, Docker's OOM killer picks the highest-RSS
process in the cgroup — the worker, since it loads the embedding model and
holds repo objects. The server parent stays alive. We deliberately do NOT
set an in-process RLIMIT_AS: ONNX Runtime mmaps model files, which counts
against virtual address space well before pages are resident, so an rlimit
small enough to actually bound RSS will fail loads at unpredictable points.
The cgroup limit is the right boundary.

Usage::

    python -m docserver.ingestion_worker [--source NAME ...] [--force]

Reads the same ``DOCSERVER_*`` environment variables as the server. Opens
the shared SQLite (in WAL mode) + ChromaDB sidecar. Runs exactly one
cycle via :class:`docserver.ingestion.Ingester` and then exits. A single
JSON line is emitted on stdout immediately before exit, tagged with
``"event": "ingestion_cycle_complete"`` so the supervising process can
ingest it back into its own log stream.

Exit codes:
    0 — cycle ran (even if individual sources failed; per-source errors
        are recorded in the cycle stats)
    1 — fatal error (config load, KB open, unhandled exception)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from docserver.config import load_config, load_doc_types_config
from docserver.ingestion import Ingester
from docserver.knowledge_base import KnowledgeBase
from docserver.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="docserver.ingestion_worker",
        description="Run one ingestion cycle and exit.",
    )
    _ = parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Restrict to a specific source (may be passed multiple times).",
    )
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index every file regardless of content hash.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    setup_logging(
        level=os.environ.get("DOCSERVER_LOG_LEVEL", "INFO"),
        json_output=os.environ.get("DOCSERVER_LOG_FORMAT", "json") == "json",
    )

    # Optional CPU yield. The supervisor sets this in the worker's env so
    # the server's request handlers stay snappy on a contended core.
    nice_offset = os.environ.get("DOCSERVER_INGEST_NICE")
    if nice_offset:
        try:
            _ = os.nice(int(nice_offset))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not apply DOCSERVER_INGEST_NICE=%r: %s",
                nice_offset,
                exc,
                extra={"event": "ingestion_worker_nice_failed"},
            )

    try:
        config = load_config()
    except Exception:
        logger.exception("Failed to load config in ingestion worker.")
        return 1

    try:
        kb = KnowledgeBase(
            config.data_dir,
            chroma_host=config.chroma_host,
            chroma_port=config.chroma_port,
        )
    except Exception:
        logger.exception("Failed to open KnowledgeBase in ingestion worker.")
        return 1

    def _emit_progress(payload: dict[str, str | int]) -> None:
        # Dedicated IPC channel to the supervisor: a single-line JSON event
        # with "event": "scan_progress". The supervisor parses these out of
        # the worker's stdout to populate /health's current_progress field.
        # Kept separate from the structured logger so it survives even if
        # the operator changes the log format.
        print(
            json.dumps({"event": "scan_progress", **payload}),
            flush=True,
        )

    # Load the doc-type classifier from disk in the worker process. The
    # server's init_app does the same thing for the in-process Ingester it
    # builds, but the subprocess does not inherit Python state — it must
    # re-load the config here or every doc ingested in Docker falls back to
    # the default 'documentation' type.
    try:
        doc_types_config = load_doc_types_config(
            known_source_names={s.name for s in config.sources}
        )
    except Exception:
        logger.exception("Failed to load doc_types config in ingestion worker.")
        kb.close()
        return 1

    try:
        ingester = Ingester(config, kb, doc_types_config=doc_types_config)
        try:
            stats = ingester.run_once(
                sources=args.source,
                force=args.force,
                progress_callback=_emit_progress,
            )
        except Exception:
            logger.exception("Unhandled exception during ingestion cycle.")
            return 1

        # Emit the final-line marker after Ingester's normal logs so the
        # supervisor can spot it. Use a stable "event" tag and keep the
        # payload small enough to fit in one line.
        print(
            json.dumps(
                {
                    "event": "ingestion_cycle_complete",
                    "stats": stats,
                    "metrics": ingester._last_ingestion,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        kb.close()


if __name__ == "__main__":
    sys.exit(main())
