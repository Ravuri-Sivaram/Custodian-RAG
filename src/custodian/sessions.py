"""Per-session registry of already-delivered chunks (returned_keys).

In the engine's stdio server, "process = session", so a single process-level set is enough. The
Custodian daemon is shared across many sessions, though, so it must isolate this state by
X-Custodian-Session -- otherwise a passage already fetched by session A would get wrongly marked
already_returned for session B (its body text would be stripped out, even though B never actually
received it). This is exactly the pitfall the engine's server.py flagged in an earlier comment
warning that a shared per-process set would misbehave once used by multiple sessions; this module
is where that gets fixed properly.

Semantics: **dedup is opt-in** -- a request without an X-Custodian-Session header gets no
cross-call dedup (returned_keys=None), which suits one-off curl/script calls; the MCP adapter
generates a uuid per process, so it naturally gets session-level dedup.
"""
from __future__ import annotations

import threading
from collections import OrderedDict


class SessionRegistry:
    """A bounded LRU: at most max_sessions sessions, each with its own returned_keys set (the
    per-set size cap is enforced as a backstop by toolcore)."""

    def __init__(self, max_sessions: int = 64):
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, set] = OrderedDict()

    def get(self, session_id: str) -> set:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            else:
                self._sessions[session_id] = set()
                while len(self._sessions) > self._max:
                    self._sessions.popitem(last=False)   # evict the least-recently-used session (losing its dedup state doesn't affect correctness)
            return self._sessions[session_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
