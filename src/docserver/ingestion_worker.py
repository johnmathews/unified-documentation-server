"""One-shot ingestion worker.

Spawned by the docserver to run a single ingestion cycle as a separate OS
process. Keeping the heavy embedding work out of the request-serving
process is the primary defence against (a) Docker OOM-killing the server
when ingestion's RSS spikes, and (b) GIL contention pausing the asyncio
event loop long enough for upstream timeouts.

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

from docserver.config import load_config
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

    try:
        ingester = Ingester(config, kb)
        try:
            stats = ingester.run_once(sources=args.source, force=args.force)
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
