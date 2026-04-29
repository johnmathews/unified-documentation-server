"""Supervises the ingestion worker subprocess.

Replaces the in-process scheduled ingestion path. Owns the APScheduler
timer in the server's process, but each tick spawns a fresh
``python -m docserver.ingestion_worker`` and waits for it to exit. The
worker's RSS is fully released to the OS on exit, so the long-running
server stays at its small steady-state working set even if a cycle peaks
high. If the worker crashes or is OOM-killed, the server's process is
unaffected — only the cycle is lost.

The worker writes its results directly to the shared SQLite + ChromaDB
sidecar; the supervisor's job is purely:

  1. Spawn the subprocess at the right time (on schedule or on
     ``/rescan``).
  2. Stream the worker's stdout into the container's log stream so the
     operator sees a single coherent log.
  3. Parse the worker's final ``ingestion_cycle_complete`` JSON line into
     ``self._last_ingestion`` so ``/health`` can surface it.
  4. Enforce a hard timeout so a runaway worker is killed rather than
     pinning resources forever.
  5. Refuse a concurrent cycle (one worker at a time per supervisor).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler

if TYPE_CHECKING:
    from docserver.config import Config

logger = logging.getLogger(__name__)


# Sentinel used to spot the worker's final metrics line in stdout.
_WORKER_FINAL_EVENT = '"ingestion_cycle_complete"'


class IngestionAlreadyRunning(RuntimeError):
    """Raised when a second cycle is requested while one is in flight."""


class IngestionTimeout(RuntimeError):
    """Raised when the worker exceeds the supervisor's hard timeout."""


class IngesterSupervisor:
    """Schedules subprocess ingestion cycles and tracks their results."""

    def __init__(
        self,
        config: Config,
        *,
        worker_module: str = "docserver.ingestion_worker",
        timeout_seconds: float = 600.0,
    ) -> None:
        self.config = config
        self._worker_module = worker_module
        self._timeout = timeout_seconds
        self._scheduler = BackgroundScheduler()
        self._proc_lock = threading.Lock()
        self._current_proc: subprocess.Popen[str] | None = None
        # Most recent successful cycle's metrics, surfaced via /health.
        self._last_ingestion: dict[str, str | int | float] | None = None
        # Most recent failure (timeout, crash, non-zero exit). Cleared on
        # next success. Surfaced via /health for operator visibility.
        self._last_failure: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin scheduled ingestion. Runs the first cycle immediately."""
        interval = self.config.poll_interval_seconds
        logger.info(
            "Starting ingestion supervisor (interval=%ds, timeout=%.0fs).",
            interval,
            self._timeout,
            extra={"event": "supervisor_start", "interval": interval},
        )
        _ = self._scheduler.add_job(
            self._run_cycle_safe,
            trigger="interval",
            seconds=interval,
            next_run_time=datetime.now(UTC),
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()

    def stop(self, *, terminate_timeout: float = 30.0) -> None:
        """Cancel future cycles and stop any running worker."""
        logger.info("Stopping ingestion supervisor.", extra={"event": "supervisor_stop"})
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        with self._proc_lock:
            proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            _ = proc.wait(timeout=terminate_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Ingestion worker did not exit within %.0fs of SIGTERM; sending SIGKILL.",
                terminate_timeout,
                extra={"event": "supervisor_kill_worker"},
            )
            proc.kill()
            _ = proc.wait(timeout=5)

    # ------------------------------------------------------------------
    # Public cycle entry points
    # ------------------------------------------------------------------

    def run_subprocess_cycle(
        self,
        sources: list[str] | None = None,
        *,
        force: bool = False,
    ) -> dict[str, str | int | float] | None:
        """Run one cycle synchronously. Returns the metrics dict on success.

        Raises ``IngestionAlreadyRunning`` if another cycle is in flight,
        and ``IngestionTimeout`` if the worker exceeds the configured
        timeout. Other crashes return ``None`` and set ``_last_failure``.
        """
        argv = list(self._build_worker_argv(sources=sources, force=force))
        rc, metrics = self._spawn_and_stream(argv, timeout=self._timeout)
        if rc == 0 and metrics is not None:
            self._last_ingestion = metrics.get("metrics") or metrics
            self._last_failure = None
            return self._last_ingestion
        # Non-zero exit or no metrics line — treat as failure but don't raise
        # for the scheduler path. Caller (run_subprocess_cycle's user) decides.
        self._last_failure = {
            "completed_at": datetime.now(UTC).isoformat(),
            "exit_code": str(rc),
            "reason": "worker exited non-zero" if rc != 0 else "no metrics line on stdout",
        }
        return None

    # ------------------------------------------------------------------
    # Read-side accessors used by /health
    # ------------------------------------------------------------------

    @property
    def last_ingestion(self) -> dict[str, str | int | float] | None:
        return self._last_ingestion

    @property
    def last_failure(self) -> dict[str, str] | None:
        return self._last_failure

    # ------------------------------------------------------------------
    # Internal: subprocess plumbing
    # ------------------------------------------------------------------

    def _run_cycle_safe(self) -> None:
        """Wrapper for the scheduler that swallows exceptions."""
        try:
            _ = self.run_subprocess_cycle()
        except IngestionAlreadyRunning:
            logger.info(
                "Skipping ingestion tick: worker already running.",
                extra={"event": "supervisor_skip_overlap"},
            )
        except IngestionTimeout as exc:
            logger.error(
                "Ingestion worker timed out: %s",
                exc,
                extra={"event": "supervisor_timeout"},
            )
        except Exception:
            logger.exception("Unhandled error in ingestion supervisor cycle.")

    def _build_worker_argv(
        self, *, sources: list[str] | None, force: bool
    ) -> list[str]:
        argv = [sys.executable, "-m", self._worker_module]
        if sources:
            for s in sources:
                argv.extend(["--source", s])
        if force:
            argv.append("--force")
        return argv

    def _spawn_and_stream(
        self, argv: list[str], *, timeout: float
    ) -> tuple[int, dict[str, str | int | float] | None]:
        """Spawn the worker, stream its output, return (exit_code, metrics).

        Enforces the timeout via a watchdog thread that SIGKILLs the worker
        once the deadline passes. The main thread iterates the worker's
        stdout, so as long as the kernel closes the pipe on process death,
        the read loop exits naturally after the kill.
        """
        env = self._worker_env()

        with self._proc_lock:
            if self._current_proc is not None and self._current_proc.poll() is None:
                raise IngestionAlreadyRunning(
                    "another ingestion cycle is already in flight"
                )
            proc = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
            self._current_proc = proc

        # ``stop_event`` lets us cancel the watchdog the instant the worker
        # exits, so a healthy quick cycle does not pay the timeout cost.
        timed_out = threading.Event()
        stop_event = threading.Event()

        def _killer() -> None:
            # stop_event.wait returns True on early signal, False on timeout.
            # We only kill when the timeout fires AND the process is still
            # running.
            if not stop_event.wait(timeout) and proc.poll() is None:
                timed_out.set()
                proc.kill()

        watchdog = threading.Thread(target=_killer, daemon=True)
        watchdog.start()

        metrics_payload: dict[str, str | int | float] | None = None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # Pass through to the container's stdout so all worker logs
                # appear in `docker logs` alongside the server's own.
                sys.stdout.write(line)
                sys.stdout.flush()
                if _WORKER_FINAL_EVENT in line:
                    # If a normal log line happens to contain the sentinel,
                    # just keep going — bad parse is not fatal.
                    with contextlib.suppress(json.JSONDecodeError):
                        metrics_payload = json.loads(line)
            rc = proc.wait()
        finally:
            stop_event.set()
            watchdog.join(timeout=1.0)
            with self._proc_lock:
                if self._current_proc is proc:
                    self._current_proc = None

        if timed_out.is_set():
            raise IngestionTimeout(
                f"worker exceeded {timeout:.0f}s and was killed"
            )

        return rc, metrics_payload

    def _worker_env(self) -> dict[str, str]:
        """Build the env passed to the worker subprocess."""
        return dict(os.environ)
