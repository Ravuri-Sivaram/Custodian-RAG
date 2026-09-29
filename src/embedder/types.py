"""Data contracts for embedder."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    """The retrieval subject = the input to ACL hard filtering. Tenant isolation plus
    principals (the user's groups plus their own id).
    Corresponds to chunker's acl contract: a filter admits when acl_tenant==user.tenant and
    (acl_allow intersects principals is non-empty, or visibility is public)."""
    tenant: str
    principals: list[str]


@dataclass
class Hit:
    """A single retrieval hit (after hybrid retrieval + ACL filtering). payload carries the
    fields needed for small-to-big expansion / output citation."""
    chunk_id: str
    doc_id: str
    kind: str
    text: str
    score: float
    payload: dict = field(default_factory=dict)   # doc_meta / image_path / source_indices / section_anchor / ...
    # Score convention: 'rrf' = the hybrid fusion score (the embedded local implementation uses
    # RRF with k=2, while a real server uses k~60, which drops the magnitude sharply; it tracks
    # rank order and cannot be compared across queries or treated as normalized);
    # 'rerank' = the cross-encoder sigmoid score, 0-1. After reranking, score and score_kind
    # must be rewritten together, otherwise rank order and score fall out of sync.
    score_kind: str = "rrf"
