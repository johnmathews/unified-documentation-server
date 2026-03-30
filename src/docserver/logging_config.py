"""Structured JSON logging configuration for Docker container output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import override


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields set by the caller (beyond standard LogRecord attrs)
        standard_attrs = logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        for key, val in record.__dict__.items():  # pyright: ignore[reportAny]
            if key not in standard_attrs and key not in log_entry and val is not None:
                log_entry[key] = val  # pyright: ignore[reportAny]

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure root logger.

    Args:
        level: Log level string.
        json_output: If True, use JSON formatter (for Docker). If False, use
                     human-readable format (for local dev).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )

    root.addHandler(handler)

    # Quiet down noisy libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
