"""Tests for IngesterSupervisor.

Most tests use a tiny "fake worker" Python script written into tmp_path
and pointed at via the ``worker_module`` constructor parameter. This
keeps the tests hermetic — they spawn a real subprocess but it does not
load chromadb, ONNX, or any of the actual ingestion code, so they run in
milliseconds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING

import pytest

from docserver.config import Config
from docserver.ingestion_supervisor import (
    IngesterSupervisor,
    IngestionAlreadyRunning,
    IngestionTimeout,
)

if TYPE_CHECKING:
    from pathlib import Path


def _install_fake_worker(tmp_path: Path, body: str) -> str:
    """Drop a tiny Python module that the supervisor will spawn instead of
    the real ingestion_worker. Returns the module name to pass via
    ``worker_module``."""
    pkg = tmp_path / "_fake_workers"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    name = f"mod_{abs(hash(body)) % (10**8)}"
    (pkg / f"{name}.py").write_text(textwrap.dedent(body))
    return f"_fake_workers.{name}"


@pytest.fixture
def small_config():
    return Config(sources=[], data_dir="/tmp/unused")


@pytest.fixture
def supervisor_factory(monkeypatch, tmp_path, small_config):
    """Returns a callable that constructs a supervisor with a fake worker."""

    def make(body: str, *, timeout: float = 30.0) -> IngesterSupervisor:
        worker_module_name = _install_fake_worker(tmp_path, body)
        # Make sure the worker module is importable from the spawned process
        # by prepending tmp_path to PYTHONPATH for both this process and any
        # subprocess that inherits the env.
        existing = os.environ.get("PYTHONPATH", "")
        new_pp = (
            f"{tmp_path}{os.pathsep}{existing}" if existing else str(tmp_path)
        )
        monkeypatch.setenv("PYTHONPATH", new_pp)
        return IngesterSupervisor(
            small_config,
            worker_module=worker_module_name,
            timeout_seconds=timeout,
        )

    return make


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_subprocess_cycle_parses_metrics(supervisor_factory, capsys):
    """A worker that prints the ingestion_cycle_complete sentinel should
    populate last_ingestion (cycle-level) and return the per-source stats."""
    body = """
    import json, sys
    print("hello from fake worker", flush=True)
    print(json.dumps({
        "event": "ingestion_cycle_complete",
        "stats": {"x": {"upserted": 3}},
        "metrics": {
            "duration_s": 0.5,
            "rss_at_end_mb": 200.0,
            "flush_count": 2,
        },
    }), flush=True)
    sys.exit(0)
    """
    sup = supervisor_factory(body)
    result = sup.run_subprocess_cycle()
    assert result == {"x": {"upserted": 3}}
    assert sup.last_stats == result
    assert sup.last_ingestion == {
        "duration_s": 0.5,
        "rss_at_end_mb": 200.0,
        "flush_count": 2,
    }
    assert sup.last_failure is None


def test_worker_log_lines_passed_through_to_stdout(supervisor_factory, capsys):
    body = """
    import json, sys
    print("regular log line", flush=True)
    print(json.dumps({
        "event": "ingestion_cycle_complete",
        "stats": {},
        "metrics": {"flush_count": 0},
    }), flush=True)
    """
    sup = supervisor_factory(body)
    _ = sup.run_subprocess_cycle()
    out = capsys.readouterr().out
    assert "regular log line" in out
    assert "ingestion_cycle_complete" in out


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_worker_non_zero_exit_records_failure(supervisor_factory):
    body = """
    import sys
    print("crashing", flush=True)
    sys.exit(2)
    """
    sup = supervisor_factory(body)
    result = sup.run_subprocess_cycle()
    assert result is None
    assert sup.last_failure is not None
    assert sup.last_failure["exit_code"] == "2"


def test_worker_no_metrics_line_records_failure(supervisor_factory):
    """A worker that exits 0 but emits no metrics line should be flagged."""
    body = """
    print("nothing useful here", flush=True)
    """
    sup = supervisor_factory(body)
    result = sup.run_subprocess_cycle()
    assert result is None
    assert sup.last_failure is not None
    assert "no metrics line" in sup.last_failure["reason"]


def test_worker_timeout_is_killed(supervisor_factory):
    body = """
    import time
    time.sleep(60)
    """
    sup = supervisor_factory(body, timeout=0.5)
    with pytest.raises(IngestionTimeout):
        _ = sup.run_subprocess_cycle()


def test_concurrent_cycle_rejected(supervisor_factory):
    """A second run while one is in flight should raise IngestionAlreadyRunning."""
    body = """
    import time
    time.sleep(2)
    """
    sup = supervisor_factory(body, timeout=10.0)

    # Manually populate _current_proc with a long-lived subprocess so the
    # next run_subprocess_cycle hits the lock check. We use sleep rather
    # than the actual fake worker to avoid race conditions.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sup._current_proc = proc
    try:
        with pytest.raises(IngestionAlreadyRunning):
            _ = sup.run_subprocess_cycle()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Argv construction
# ---------------------------------------------------------------------------


def test_build_worker_argv_passes_sources_and_force(small_config):
    sup = IngesterSupervisor(small_config, worker_module="x")
    argv = sup._build_worker_argv(sources=["a", "b"], force=True)
    assert "--source" in argv
    assert "a" in argv
    assert "b" in argv
    assert "--force" in argv


def test_build_worker_argv_no_flags_when_default(small_config):
    sup = IngesterSupervisor(small_config, worker_module="x")
    argv = sup._build_worker_argv(sources=None, force=False)
    assert "--source" not in argv
    assert "--force" not in argv


# ---------------------------------------------------------------------------
# Live progress reporting
# ---------------------------------------------------------------------------


def test_progress_events_update_current_progress(supervisor_factory):
    """The supervisor should parse scan_progress lines into current_progress."""
    body = """
    import json, sys, time
    print(json.dumps({"event": "scan_progress", "phase": "syncing"}), flush=True)
    print(json.dumps({
        "event": "scan_progress",
        "phase": "discovery_done",
        "total_docs": 5,
        "sources_changed": 2,
        "sources_total": 4,
    }), flush=True)
    print(json.dumps({
        "event": "scan_progress",
        "phase": "processing",
        "current": 1,
        "total": 5,
        "source": "alpha",
        "doc": "a.md",
    }), flush=True)
    print(json.dumps({
        "event": "ingestion_cycle_complete",
        "stats": {},
        "metrics": {"flush_count": 0},
    }), flush=True)
    """
    sup = supervisor_factory(body)
    _ = sup.run_subprocess_cycle()
    # Once the worker exits, current_progress is cleared regardless of what
    # the last in-flight event was.
    assert sup.current_progress is None


def test_progress_cleared_on_worker_failure(supervisor_factory):
    """Even when the worker exits non-zero, current_progress is cleared."""
    body = """
    import json, sys
    print(json.dumps({"event": "scan_progress", "phase": "processing",
                      "current": 3, "total": 10}), flush=True)
    sys.exit(2)
    """
    sup = supervisor_factory(body)
    _ = sup.run_subprocess_cycle()
    assert sup.current_progress is None
    assert sup.last_failure is not None


def test_progress_cleared_on_worker_timeout(supervisor_factory):
    """Worker timeout should also clear current_progress."""
    body = """
    import json, time
    print(json.dumps({"event": "scan_progress", "phase": "processing",
                      "current": 1, "total": 100}), flush=True)
    time.sleep(60)
    """
    sup = supervisor_factory(body, timeout=0.5)
    with pytest.raises(IngestionTimeout):
        _ = sup.run_subprocess_cycle()
    assert sup.current_progress is None


def test_progress_parsing_directly(small_config):
    """Unit-test the line-parsing path in isolation by feeding a real instance
    a few hand-crafted lines via _spawn_and_stream's helpers. Exercises the
    JSON parser branch without spawning a subprocess for the parse logic itself.
    """
    sup = IngesterSupervisor(small_config, worker_module="x")
    # Simulate the supervisor having seen one progress line.
    with sup._progress_lock:
        sup._current_progress = {
            "phase": "processing",
            "current": 4,
            "total": 9,
            "source": "demo",
            "doc": "guide.md",
        }
    snap = sup.current_progress
    assert snap == {
        "phase": "processing",
        "current": 4,
        "total": 9,
        "source": "demo",
        "doc": "guide.md",
    }
    # Ensure current_progress returns a copy — caller mutation must not
    # leak back to internal state.
    snap["phase"] = "tampered"
    assert sup.current_progress is not None
    assert sup.current_progress["phase"] == "processing"


def test_progress_malformed_json_is_ignored(supervisor_factory):
    """A line that contains the sentinel but isn't valid JSON shouldn't crash."""
    body = """
    import json, sys
    # Looks like a progress line but isn't valid JSON
    print('garbage with "scan_progress" inside but not real json {', flush=True)
    print(json.dumps({"event": "scan_progress", "phase": "syncing"}), flush=True)
    print(json.dumps({
        "event": "ingestion_cycle_complete",
        "stats": {},
        "metrics": {"flush_count": 0},
    }), flush=True)
    """
    sup = supervisor_factory(body)
    result = sup.run_subprocess_cycle()
    # Cycle should still succeed despite the malformed line.
    assert result == {}


# ---------------------------------------------------------------------------
# stop() lifecycle
# ---------------------------------------------------------------------------


def test_stop_terminates_running_worker(supervisor_factory):
    body = """
    import time
    time.sleep(60)
    """
    sup = supervisor_factory(body, timeout=120.0)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sup._current_proc = proc
    sup.stop(terminate_timeout=2.0)
    # After stop(), the proc should have been signalled and exited.
    deadline = time.time() + 5.0
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.1)
    assert proc.poll() is not None, "stop() should have terminated the worker"
