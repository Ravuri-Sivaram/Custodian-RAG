"""ACL client-side logic unit tests (pure CPU). Focus: empty-tenant fail-closed (seal#1) + admits
and split having the same semantics + RESTRICTED defaults to deny."""


from embedder.acl import acl_admits, acl_split
from embedder.types import User


def test_empty_tenant_failclosed():
    # seal#1: a public document with no tenant can't be safely attributed to a tenant -> acl_split
    # marks unset=True (defaults to deny)
    assert acl_split({"visibility": "public", "allow": []})["acl_unset"] is True
    # An empty-tenant document is denied even to an empty-tenant user too (closes the fail-open
    # hole from an empty string matching itself)
    assert acl_admits({"visibility": "public"}, User(tenant="", principals=[])) is False
    # An empty-tenant document is denied to a real-tenant user too
    assert acl_admits({"visibility": "public"}, User(tenant="t1", principals=[])) is False


def test_restricted_acl_denied():
    RESTRICTED = {"visibility": "restricted", "allow": [], "unset": True}   # chunker RESTRICTED_ACL
    assert acl_split(RESTRICTED)["acl_unset"] is True
    assert acl_admits(RESTRICTED, User(tenant="t1", principals=["g"])) is False
    assert acl_admits({}, User(tenant="t1", principals=["g"])) is False     # an empty dict is denied


def test_normal_visibility():
    pub = {"tenant": "t1", "visibility": "public", "allow": []}
    assert acl_admits(pub, User(tenant="t1", principals=["x"])) is True     # same tenant, public
    assert acl_admits(pub, User(tenant="t2", principals=["x"])) is False    # cross-tenant public is still denied
    r = {"tenant": "t1", "visibility": "restricted", "allow": ["g_hr"]}
    assert acl_admits(r, User(tenant="t1", principals=["g_hr"])) is True    # authorized
    assert acl_admits(r, User(tenant="t1", principals=["g_x"])) is False    # same tenant, wrong group
    assert acl_admits(r, User(tenant="t2", principals=["g_hr"])) is False   # cross-tenant


def test_split_admits_same_semantics():
    # acl_admits must share the same semantics as acl_split (and by extension store.acl_filter),
    # or the small-to-big path becomes a bypass
    r = {"tenant": "t1", "visibility": "restricted", "allow": ["g_hr"]}
    f = acl_split(r)
    u = User(tenant="t1", principals=["g_hr"])
    store_pass = (not f["acl_unset"] and f["acl_tenant"] == u.tenant
                  and (bool(set(f["acl_allow"]) & set(u.principals)) or f["acl_visibility"] == "public"))
    assert store_pass == acl_admits(r, u)


def test_principals_none_tolerated():
    # B2 review nit: principals=None must not crash (set(None or []) tolerates it). public is
    # still visible, restricted is still denied.
    assert acl_admits({"tenant": "t1", "visibility": "public", "allow": []}, User("t1", None)) is True
    assert acl_admits({"tenant": "t1", "visibility": "restricted", "allow": ["g"]}, User("t1", None)) is False


if __name__ == "__main__":
    test_empty_tenant_failclosed()
    test_restricted_acl_denied()
    test_normal_visibility()
    test_split_admits_same_semantics()
    test_principals_none_tolerated()
    print("acl tests OK")
