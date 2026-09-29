"""SessionRegistry: same session gets the same set, LRU eviction, bounded capacity."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from custodian.sessions import SessionRegistry


def test_same_session_same_set():
    reg = SessionRegistry()
    s = reg.get("a")
    s.add(("d1", "c1"))
    assert ("d1", "c1") in reg.get("a")          # same session shares the registration
    assert ("d1", "c1") not in reg.get("b")      # isolated across sessions


def test_lru_eviction_bounded():
    reg = SessionRegistry(max_sessions=3)
    for i in range(5):
        reg.get(f"s{i}")
    assert len(reg) == 3
    # The least-recently-used s0/s1 get evicted; getting s2 again afterward yields a fresh empty
    # set (losing dedup state doesn't affect correctness)
    reg.get("s2").add("x")
    reg.get("s0")                                # s0 comes back in, evicting the next oldest
    assert len(reg) == 3


def test_touch_refreshes_order():
    reg = SessionRegistry(max_sessions=2)
    reg.get("a").add("ka")
    reg.get("b")
    reg.get("a")                                 # touch a -> b becomes the oldest
    reg.get("c")                                 # evicts b
    assert "ka" in reg.get("a")                  # a is still there (not evicted, registration kept)
