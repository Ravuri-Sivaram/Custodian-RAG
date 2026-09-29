"""Observability (DESIGN D11): request logging (JSONL) plus in-process metrics. Good enough for
our needs, without pulling in a monitoring stack.

Privacy and security boundary: logs **never persist the key itself** (only the identity's name is
recorded); queries are logged by default (truncated to 120 chars, prioritizing internal debugging
value), and CUSTODIAN_LOG_QUERIES=off turns that off. An observability failure never affects the
service (a log-write exception is only counted, never propagated).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque

log = logging.getLogger("custodian")


class Stats:
    """In-process metrics: per-endpoint counts/error counts/a ring buffer of latencies
    (p50/p95). Resets on restart (acceptable for now)."""

    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self._n = defaultdict(int)
        self._err = defaultdict(int)
        self._lat = defaultdict(lambda: deque(maxlen=window))
        self.started = time.time()
        self.log_write_failures = 0

    def record(self, ep: str, ms: float, error: bool) -> None:
        with self._lock:
            self._n[ep] += 1
            if error:
                self._err[ep] += 1
            self._lat[ep].append(ms)

    @staticmethod
    def _pct(sorted_vals, p):
        if not sorted_vals:
            return None
        i = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
        return round(sorted_vals[i], 1)

    def snapshot(self) -> dict:
        with self._lock:
            eps = {}
            for ep, n in sorted(self._n.items()):
                lat = sorted(self._lat[ep])
                eps[ep] = {"n": n, "errors": self._err[ep],
                           "p50_ms": self._pct(lat, 50), "p95_ms": self._pct(lat, 95),
                           "max_ms": round(lat[-1], 1) if lat else None}
            return {"uptime_s": round(time.time() - self.started, 1),
                    "log_write_failures": self.log_write_failures, "endpoints": eps}


class RequestLog:
    """Append-only JSONL log. An empty dir means logging is disabled. Single file (fine at team
    scale); rotation is left to logrotate (see OPERATIONS).

    Writing to disk goes through a **single background writer thread plus a bounded queue** (per
    the phase-F review): `write` is called from the async middleware `_observe`, and if it did a
    synchronous open/append/close on the event-loop thread, an I/O stall on a shared bind-mounted
    volume would freeze the entire event loop -- taking down every endpoint on the replica
    (including health-check responses) at once. Instead, `write` only does a non-blocking enqueue
    (a full queue just drops the record and counts it, preserving the guarantee that "observability
    never drags down the service"), and the actual disk write happens on a separate daemon thread
    -- so the event loop is never blocked by disk I/O."""

    def __init__(self, log_dir: str, log_queries: bool = True, queue_max: int = 4096):
        self.enabled = bool(log_dir)
        self.log_queries = log_queries
        self.path = ""
        self._q: queue.Queue | None = None
        self._writer: threading.Thread | None = None
        if self.enabled:
            d = os.path.expanduser(log_dir)
            os.makedirs(d, exist_ok=True)
            self.path = os.path.join(d, "requests.jsonl")
            self._q = queue.Queue(maxsize=queue_max)
            self._writer = threading.Thread(target=self._drain, name="reqlog-writer", daemon=True)
            self._writer.start()

    def _drain(self) -> None:
        """The single background writer thread: writes to disk serially (a single thread is
        naturally serial, so no lock is needed). stats is only used to count write failures."""
        while True:
            rec, stats = self._q.get()
            try:
                line = json.dumps(rec, ensure_ascii=False)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                if stats is not None:
                    stats.log_write_failures += 1
            finally:
                self._q.task_done()

    def flush(self, timeout: float | None = None) -> None:
        """Wait for all currently-queued records to be written to disk (for tests/graceful
        shutdown; never called on the production hot path). A no-op if there's no queue (logging
        disabled). timeout=None waits indefinitely (used for synchronous test assertions); with a
        timeout given, it waits against a deadline instead -- the stdlib queue.join() doesn't
        support a timeout, and if the writer thread is stuck in open/write (exactly the
        bind-mount I/O hang scenario this class's docstring already acknowledges), an unbounded
        join would freeze graceful shutdown until stop_grace_period expires and it gets SIGKILLed.
        On timeout, log a warning and give up (the log records stuck behind a hung disk weren't
        going to get written anyway)."""
        if self._q is None:
            return
        if timeout is None:
            self._q.join()
            return
        deadline = time.monotonic() + timeout
        with self._q.all_tasks_done:              # the same condition variable queue.join() uses, but with wait() given the remaining deadline
            while self._q.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("reqlog flush timed out (%.1fs), giving up with %d records unwritten",
                                timeout, self._q.unfinished_tasks)
                    return
                self._q.all_tasks_done.wait(remaining)

    def write(self, rec: dict, stats: Stats | None = None) -> None:
        if not self.enabled:
            return
        # The privacy boundary is enforced here in this layer (per review: truncation used to
        # happen only in _observe, so any new caller would have missed it):
        # log_queries=off removes query entirely; otherwise it's uniformly truncated to 120 chars.
        # The enqueue call is the single enforcement point for this privacy boundary (done before
        # enqueueing, so it doesn't cost the writer thread anything).
        if not self.log_queries:
            rec.pop("query", None)
        elif rec.get("query") is not None:
            rec["query"] = str(rec["query"])[:120]
        try:
            self._q.put_nowait((rec, stats))         # non-blocking enqueue: a full queue immediately raises Full, never blocking the caller (the event loop)
        except queue.Full:
            if stats is not None:                     # an observability failure doesn't affect the service, just counts it for /v1/stats to expose (the newest record is dropped)
                stats.log_write_failures += 1
