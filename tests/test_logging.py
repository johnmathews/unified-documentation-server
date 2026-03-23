"""Tests for structured logging configuration."""

import json
import logging

from docserver.logging_config import JSONFormatter, setup_logging


def test_json_formatter_basic():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"
    assert "timestamp" in parsed


def test_json_formatter_extra_fields():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="search done",
        args=(),
        exc_info=None,
    )
    record.event = "search"
    record.duration_ms = 42
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["event"] == "search"
    assert parsed["duration_ms"] == 42


def test_json_formatter_exception():
    formatter = JSONFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="something failed",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_json_formatter_arbitrary_extra_fields():
    """Formatter should include any extra field, not just a fixed allowlist."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="ingestion done",
        args=(),
        exc_info=None,
    )
    record.source = "my-repo"
    record.change_type = "modified"
    record.progress = "3/10"
    record.custom_field = {"nested": True}
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["source"] == "my-repo"
    assert parsed["change_type"] == "modified"
    assert parsed["progress"] == "3/10"
    assert parsed["custom_field"] == {"nested": True}


def test_json_formatter_no_extra_fields():
    """Standard fields only — no spurious keys in output."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="plain message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert set(parsed.keys()) == {"timestamp", "level", "logger", "message"}


def test_setup_logging_json():
    setup_logging(level="DEBUG", json_output=True)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JSONFormatter)


def test_setup_logging_human():
    setup_logging(level="INFO", json_output=False)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert not isinstance(root.handlers[0].formatter, JSONFormatter)
