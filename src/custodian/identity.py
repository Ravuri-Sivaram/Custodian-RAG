"""Multi-identity support (DESIGN D10): API key -> identity.

Three modes (service.py picks one automatically based on config):
  keys   -- team mode: CUSTODIAN_KEYS_FILE points at a JSON file; each request's X-API-Key resolves
            to an identity; anything unknown or missing is always 401.
  legacy -- single user / one gate: only CUSTODIAN_API_KEY is set, a single key bound to one identity
            at startup.
  open   -- neither is set: unauthenticated mode, restricted to loopback addresses only.

Fail-closed rule: a malformed keys file refuses to start (never silently degrades); binding to a
non-loopback address requires keys mode (enforced by a startup guard in service.py). Identity only
answers "who is asking"; "what they can see" is enforced by the engine's ACL logic using that
identity's User.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field

KEY_MIN_LEN = 16


@dataclass(frozen=True)
class Identity:
    name: str
    tenant: str
    principals: list[str] = field(default_factory=list)
    admin: bool = False


def load_keys(path: str) -> dict[str, Identity]:
    """Load the keys file -> {key: Identity}. Any format problem fails loudly (SystemExit); this
    never silently degrades into open mode."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"CUSTODIAN_KEYS_FILE does not exist: {path} (generate one with `custodian keys new <name>`)")
    except json.JSONDecodeError as e:
        raise SystemExit(f"keys file is not valid JSON: {path} ({e})")
    entries = data.get("keys")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"keys file must contain a non-empty keys array: {path}")
    out: dict[str, Identity] = {}
    seen_names: set[str] = set()
    for i, k in enumerate(entries):
        key = str(k.get("key") or "").strip()
        name = str(k.get("name") or "").strip()
        tenant = str(k.get("tenant") or "").strip()
        if len(key) < KEY_MIN_LEN:
            raise SystemExit(f"keys[{i}].key is too short (<{KEY_MIN_LEN} chars), refusing to start")
        if not name or not tenant:
            raise SystemExit(f"keys[{i}] is missing name/tenant (fail-closed: an incomplete identity does not go live)")
        if "|" in name:
            # name is the prefix of the session-dedup registration key f"{name}|{sid}"
            # (service._session_keys): allowing '|' in it would let namespaces collide
            # (a + "b|c" produces the same key as "a|b" + c). Forbidding '|' keeps the prefix
            # unambiguous.
            raise SystemExit(f"keys[{i}].name must not contain '|' (it is the session-isolation prefix separator)")
        if name in seen_names:
            # name must be unique: it is both the identity label used in logs and the
            # session-dedup prefix; a duplicate would let two different identities share a
            # session namespace (dedup would get mixed up) and logs would no longer be able to
            # tell them apart. Rejected fail-closed.
            raise SystemExit(f"keys[{i}].name '{name}' is duplicated (identity names must be unique)")
        if key in out:
            raise SystemExit(f"keys[{i}]'s key duplicates an earlier entry")
        seen_names.add(name)
        principals = [str(p).strip() for p in (k.get("principals") or []) if str(p).strip()]
        out[key] = Identity(name=name, tenant=tenant, principals=principals, admin=bool(k.get("admin")))
    return out


def new_key(name: str) -> str:
    """Generate a key (pk_<name>_<32hex>, cryptographically random via secrets). It is printed
    only once at generation time; the keys file is the only place it's persisted."""
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "user"
    return f"pk_{safe}_{secrets.token_hex(16)}"


def append_key(path: str, *, name: str, tenant: str, principals: list[str], admin: bool = False) -> str:
    """Append a new identity to the keys file (creating it if it doesn't exist), returning the
    generated key. Best-effort chmod 600."""
    path = os.path.expanduser(path)
    data = {"keys": []}
    if os.path.exists(path):
        load_keys(path)                             # reuse serve-side validation: corruption, duplicate names, or '|' in a name all fail loudly instead of raising a bare traceback
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    if "|" in name:
        raise SystemExit("name must not contain '|' (it is the session-isolation prefix separator)")
    if any(e.get("name") == name for e in data["keys"]):
        raise SystemExit(f"name '{name}' already exists (identity names must be unique)")
    key = new_key(name)
    data["keys"].append({"key": key, "name": name, "tenant": tenant,
                         "principals": principals, "admin": admin})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)                       # tighten permissions on the keys file (best-effort on Windows)
    except OSError:
        pass
    return key


def is_loopback(host: str) -> bool:
    return (host or "").strip().lower() in ("127.0.0.1", "localhost", "::1")
