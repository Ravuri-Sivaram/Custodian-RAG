"""Client-side ACL logic (kept in one place for easier auditing). chunk.acl is a free-form dict;
this module handles decomposing it and judging visibility.

Both of the following uses must stay **strictly semantically identical** to `store.acl_filter`
(the Qdrant server-side hard filter):
  - `acl_split`: at indexing time, splits acl into the store's 4 filterable payload fields;
  - `acl_admits`: at retrieval time, the element-by-element visibility predicate used when
    fetching small-to-big material.
If the two drift apart in meaning, small-to-big (refetching the original text for big.text by
index range) could bypass the hard filter and leak plaintext the user isn't authorized to see --
this is one of the integration invariants that must always hold.
fail-closed: any empty or missing field is treated as the strictest case (matching chunker's
RESTRICTED_ACL: unset=True / empty allow / restricted).
deny has no effect (consistent with the integration's example filter: excluding a group means
removing it from allow)."""
from __future__ import annotations

from .types import User


def acl_split(acl: dict) -> dict:
    """Splits chunk.acl into the store payload's 4 filterable ACL fields (fail-closed by default)."""
    acl = acl or {}
    tenant = acl.get("tenant") or ""
    return {
        # An empty or missing tenant is treated as unset (fail-closed): an empty-string tenant
        # would self-match against "users with an empty tenant", which would make tenant
        # isolation meaningless and leave only the public flag as a gate -- effectively fail-open.
        # A document that cannot be safely attributed to a tenant is denied by default; if
        # something genuinely needs to be "globally public", that should go through an explicit
        # mechanism (e.g. an explicit tenant value), never by relying on "no tenant".
        "acl_unset": bool(acl.get("unset")) or not acl or not tenant,
        "acl_tenant": tenant,
        "acl_allow": list(acl.get("allow") or []),
        "acl_visibility": acl.get("visibility") or "restricted",    # missing -> restricted (not public)
    }


def acl_admits(acl: dict, user: User) -> bool:
    """Whether a single acl is visible to user -- the client-side equivalent of store.acl_filter
    (must be: not unset AND same tenant AND (allow intersects principals OR public))."""
    f = acl_split(acl)
    if f["acl_unset"] or not f["acl_tenant"] or not user.tenant:    # An empty tenant on either side is denied
        return False
    if f["acl_tenant"] != user.tenant:
        return False
    return bool(set(f["acl_allow"]) & set(user.principals or [])) or f["acl_visibility"] == "public"  # principals=None is tolerated defensively
