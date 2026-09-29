"""Concurrency regression tests for the M1 lock-sinking work (pure CPU, doesn't touch a real
GPU/Qdrant/HTTP). Revised after adversarial review (fixing tests that were falsely green):
 ① remote encode backoff happens outside **search_with_context** (the method surface the old
    LockedRetriever genuinely serialized) -> doesn't block other queries;
 ②/②b Store._lock serializes the single Qdrant client's **read (hybrid_search) and write
    (upsert)** paths (peak concurrency == 1);
 ③/③b Dense/Reranker._fwd_lock serializes the local GPU forward pass (peak == 1);
 ⑦ encode_query's two-phase design: encoding outside the lock doesn't block another key's read
    phase (the cache lock's real value -- under CPython's GIL a single dict operation is already
    atomic, so there's no corruption to test here; see scratchpad/diag_sensitivity.py
    counter-example 1; this lock instead guarantees two-phase consistency + targets
    free-threading builds);
 ⑤/⑨ Dense/Reranker._load are single-flight; ⑥ RemoteDense/RemoteReranker's forward/load
    locks are nullcontext.
The concurrency-peak probe uses threading.Barrier to force overlap -> removing the lock
deterministically makes peak==N (a deterministic failure), not something relying on sleep timing luck."""
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from embedder.config import EmbedConfig
from embedder.dense import Dense
from embedder.types import User


class _OverlapProbe:
    """A concurrency-peak probe. Barrier(n) forces n threads to rendezvous at the entry point: with
    no lock -> all n enter at once -> peak==n; with a lock -> only 1 can enter, so the barrier
    never fills and times out -> peak==1. Removing the lock always fails this (deterministic, not
    relying on sleep timing luck)."""
    def __init__(self, n: int, result, hold: float = 0.05, timeout: float = 0.4):
        self.n = n
        self._result = result
        self.hold = hold
        self.timeout = timeout
        self.peak = self.cur = 0
        self._lk = threading.Lock()
        self._barrier = threading.Barrier(n)

    def __call__(self, *a, **k):
        try:
            self._barrier.wait(timeout=self.timeout)   # no lock: n threads rendezvous; with a lock serializing them: never fills -> BrokenBarrier
        except threading.BrokenBarrierError:
            pass
        with self._lk:
            self.cur += 1
            self.peak = max(self.peak, self.cur)
        time.sleep(self.hold)
        with self._lk:
            self.cur -= 1
        return self._result()


# ---------- ① M1 core: remote encode backoff doesn't block other queries (through search_with_context -- the exact method surface the old big lock genuinely serialized) ----------
def test_remote_encode_backoff_not_block_others():
    from embedder.retriever import Retriever
    started, release = threading.Event(), threading.Event()

    class SlowRemoteDense:                       # simulates RemoteDense: the first query's encode gets stuck in backoff (HTTP, outside the lock)
        def __init__(self):
            self.n = 0
            self._lk = threading.Lock()

        def encode_query(self, q):
            with self._lk:
                self.n += 1
                first = self.n == 1
            if first:
                started.set()
                release.wait(3)                  # gets stuck (simulates a remote backoff sleep)
            return SimpleNamespace(tolist=lambda: [0.0] * 8)

    r = Retriever(EmbedConfig(inference_url="http://x"),
                  store=SimpleNamespace(hybrid_search=lambda *a, **k: []),
                  dense=SlowRemoteDense(), reranker=SimpleNamespace())
    fast_done = threading.Event()

    def slow():
        r.search_with_context("q1", User("t", []), assemble=False)   # exactly the method surface the old LockedRetriever serialized

    def fast():
        started.wait(2)
        r.search_with_context("q2", User("t", []), assemble=False)   # after the fix: not blocked by slow's backoff
        fast_done.set()

    ts, tf = threading.Thread(target=slow), threading.Thread(target=fast)
    ts.start(); tf.start()
    try:
        assert fast_done.wait(2), "fast's search_with_context was blocked by slow's encode backoff (M1: the big lock serialized the whole method surface)"
    finally:
        release.set(); ts.join(3); tf.join(1)


# ---------- ②/②b Store._lock serializes the read (hybrid_search) and write (upsert) paths ----------
def _fresh_store():
    from embedder.store import Store
    s = Store.__new__(Store)                     # bypasses __init__ (doesn't open a real Qdrant)
    s.cfg = EmbedConfig()
    s._lock = threading.Lock()
    return s


def test_store_lock_serializes_hybrid_search():
    s = _fresh_store()
    probe = _OverlapProbe(4, lambda: SimpleNamespace(points=[]))
    s.client = SimpleNamespace(query_points=probe)
    ts = [threading.Thread(target=lambda: s.hybrid_search([0.0] * 8, None, User("t", []), strategy="dense")) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"the Store lock did not serialize hybrid_search, peak={probe.peak}"


def test_store_lock_serializes_upsert():        # the write path (most dangerous: np.append rebinds the array), review gap #4
    s = _fresh_store()
    probe = _OverlapProbe(4, lambda: None)
    s.client = SimpleNamespace(upsert=probe)
    ts = [threading.Thread(target=lambda: s.upsert([])) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"the Store lock did not serialize upsert (the write path!), peak={probe.peak}"


# ---------- ③/③b Dense/Reranker._fwd_lock serializes the local GPU forward pass ----------
def test_dense_fwd_lock_serializes():
    torch = pytest.importorskip("torch", reason="needs CPU torch (in the [gpu] extra, not [dev]); installed separately in CI")
    d = Dense(EmbedConfig(dense_dim=1024))
    probe = _OverlapProbe(4, lambda: torch.zeros(1, 4096))   # _mrl truncates to 1024 + renorm (real CPU torch)
    d._model = SimpleNamespace(process=probe)                # _model is already set -> _load returns quickly, doesn't touch a real GPU
    ts = [threading.Thread(target=lambda: d.encode_text(["x"])) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"Dense._fwd_lock did not serialize the GPU forward pass, peak={probe.peak}"


def test_reranker_fwd_lock_serializes():        # review gap ③: rerank has its own separately added lock, tested on its own
    from embedder.rerank import Reranker
    rr = Reranker(EmbedConfig())
    probe = _OverlapProbe(4, lambda: [0.5])     # score's contract: len(scores)==len(docs)==1
    rr._model = SimpleNamespace(process=probe)
    ts = [threading.Thread(target=lambda: rr.score("q", ["d"])) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"Reranker._fwd_lock did not serialize the GPU forward pass, peak={probe.peak}"


# ---------- ⑦ encode_query's two-phase design: encoding outside the lock doesn't block another key's read phase (the cache lock's real job, a core invariant of the M1 dense layer) ----------
# (the original test④ "cache concurrent corrupt" was removed: under CPython's GIL a single dict
# operation is atomic, so removing the lock doesn't corrupt anything -- the counter-example proved it was falsely green.)
def test_cache_read_not_blocked_by_slow_encode():
    d = Dense(EmbedConfig())
    d.encode_text = lambda texts, instruction=None: np.zeros((1, 8), dtype=np.float32)
    d.encode_query("cached")                     # warms up: stores it in the cache
    slow_in, release = threading.Event(), threading.Event()

    def slow_encode(texts, instruction=None):    # swapped for a slow encode (simulating backoff, outside _cache_lock)
        slow_in.set(); release.wait(3)
        return np.zeros((1, 8), dtype=np.float32)
    d.encode_text = slow_encode
    fast_done = threading.Event()

    def slow():
        d.encode_query("newkey")                 # a miss -> the read phase (lock) releases -> encode_text hangs outside the lock
    def fast():
        slow_in.wait(2)
        d.encode_query("cached")                 # a hit -> the read phase (lock): if two-phase works, slow isn't holding the lock -> returns immediately
        fast_done.set()

    ts, tf = threading.Thread(target=slow), threading.Thread(target=fast)
    ts.start(); tf.start()
    try:
        assert fast_done.wait(2), "a read for an already-cached key was blocked by the slow encode (encode ran inside _cache_lock = two-phase isn't working)"
    finally:
        release.set(); ts.join(3); tf.join(1)


# ---------- ⑤/⑨ _load single-flight ----------
def _load_single_flight(model_cls_setter, load_fn):
    count = {"n": 0}

    def build(**k):
        count["n"] += 1
        time.sleep(0.05)                         # a slow load, to widen the concurrency window
        return object()
    mod_name, mod = model_cls_setter(build)
    import sys
    sys.modules[mod_name] = mod
    try:
        ts = [threading.Thread(target=load_fn) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        return count["n"]
    finally:
        sys.modules.pop(mod_name, None)


def test_dense_load_single_flight():
    pytest.importorskip("torch", reason="the Dense._load path under test needs torch; CPU torch is installed separately in CI")
    d = Dense(EmbedConfig())
    d._assert_gpu = lambda: None

    def setter(build):
        m = types.ModuleType("qwen3_vl_embedding")
        m.Qwen3VLEmbedder = build
        return "qwen3_vl_embedding", m
    assert _load_single_flight(setter, d._load) == 1, "Dense._load is not single-flight (concurrent first queries reload the 8B model repeatedly)"


def test_reranker_load_single_flight():         # review gap ③: reranker's _load also needs the double-check
    pytest.importorskip("torch", reason="the Reranker._load path under test needs torch; CPU torch is installed separately in CI")
    from embedder.rerank import Reranker
    rr = Reranker(EmbedConfig())
    rr._assert_gpu = lambda: None

    def setter(build):
        m = types.ModuleType("qwen3_vl_reranker")
        m.Qwen3VLReranker = build
        return "qwen3_vl_reranker", m
    assert _load_single_flight(setter, rr._load) == 1, "Reranker._load is not single-flight"


# ---------- ⑥ the remote forward/load locks are nullcontext ----------
def test_remote_forward_locks_are_nullcontext():
    import contextlib

    from embedder.remote import RemoteDense, RemoteReranker
    rd = RemoteDense(EmbedConfig(inference_url="http://x"))
    assert isinstance(rd._fwd_lock, contextlib.nullcontext)
    assert isinstance(rd._load_lock, contextlib.nullcontext)
    assert not isinstance(rd._cache_lock, contextlib.nullcontext)   # the cache lock is still a real lock (inherited from Dense; remote also uses the LRU)
    rr = RemoteReranker(EmbedConfig(inference_url="http://x"))
    assert isinstance(rr._fwd_lock, contextlib.nullcontext)
    assert isinstance(rr._load_lock, contextlib.nullcontext)
