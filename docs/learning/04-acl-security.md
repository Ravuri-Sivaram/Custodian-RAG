# 04 Enterprise ACL and Security Model

> **How to read this piece**
> This piece covers how custodian handles permissions inside RAG: from the ACL stamp at the chunker, the hard filter and exit re-check at recall in the embedder, and equivalence-class material-pulling in small-to-big, through to the three identity modes and no-leakage rules at the service layer — and the testing methodology for "how do you prove that one layer of a defense-in-depth stack is actually effective on its own."
> **Interview weight: high.** "How do you do permissions in a multi-tenant RAG system" is a must-ask question in enterprise RAG interviews, and "an embedded vector store dropping filter clauses under fusion" is a rare, first-hand piece of evidence caught by real measurement.
> **Prerequisite reading**: it helps to already understand the basic shape of small-to-big retrieval (index small chunks, deliver large ones); for the evaluation basis, see [07 Evaluation Methodology](07-evaluation.md).

---

## 1. Conceptual foundation: why permissions in RAG are harder than they look

Independent of this project, let's get the problem itself straight first.

**RAG shatters the existing permission boundary.** In a traditional document system, permissions hang off the "file": if you don't have access, you can't open the file, end of story. RAG's first step is to cut the file into hundreds of chunks, encode them into vectors, and mix them into one big store — and then use "semantic similarity," not "permission," to decide what to show a user. This creates a counterintuitive danger: **the more relevant unauthorized content is to a query, the more likely it is to be recalled.** An ordinary employee asking "executive compensation plan" — vector retrieval will faithfully rank the HR-confidential document first — unless something stops it.

**The mainstream spectrum of approaches** (from weakest to strongest isolation):

| Approach | How it works | Problem |
|---|---|---|
| **Post-filtering** | Recall the top-k by similarity first, then filter out anything unauthorized | ① unauthorized results contaminate the limit: 8 of the top 10 get filtered out, leaving the user with only 2; ② a bug in the filtering code leaks directly (fail-open); ③ "fetch it first, then discard it" is itself one instance of unauthorized reading |
| **Pre-filtering at recall** | Encode permissions as vector-store metadata filters, pushed down into the retrieval call | Depends on the correctness and expressiveness of the vector store's filter; once the permission model gets complex, filter construction gets error-prone |
| **Physical isolation** | A separate collection or index per tenant/permission domain | Strongest isolation, but operational cost is O(number of tenants), and documents shared across domains (e.g. a company-wide announcement) have to be duplicated |

**RAG has three particular ways to be bypassed**, which is what makes this harder here than "row-level permissions in a database":

1. **Context expansion can bypass retrieval filtering.** Mechanisms like small-to-big, "pull back the original text around a hit," reach back into the original document and re-pull material *after* the hard filter has already run — the filter stops "an unauthorized chunk being retrieved," but not "starting from an authorized chunk and pulling an unauthorized neighbor's original text into the context."
2. **By-id direct-read interfaces.** Tools that fetch content directly by chunk_id / doc_id (expand, get_document) don't go through the retrieval path at all — the retrieval layer's filter has no effect on them.
3. **Error messages leak existence.** The difference between "404, doesn't exist" and "403, no access" is itself information: an attacker can use it to enumerate "there's a document called X here that I can't see."

Two general concepts run throughout this piece:

- **Fail-open vs. fail-closed**: when permission information is missing/corrupted/ambiguous, does the default behavior allow access or deny it? The iron rule of a security system is fail-closed — any "incomplete configuration" should show up as "can't see it" or "won't start," never as "can see everything."
- **Authentication (AuthN) vs. authorization (AuthZ), layered**: "who's asking" and "what can they see" are two orthogonal questions. The former is solved at the service boundary (API key, SSO), the latter at the data layer (ACL filtering). A system that mixes the two has to rewrite its permission logic every time the access method changes (HTTP → MCP → CLI).

---

## 2. How Custodian does it

### 2.0 The big picture: the life of one ACL

```
manifest declares acl ──► chunker stamps it (default = RESTRICTED, fail-closed)
                        │  every chunk gets a deep copy of the acl; acl_index() records {element idx → acl}
                        ▼
                embedder, index time: acl_split breaks it into 4 filterable Qdrant payload fields
                        │  the raw acl is also stored in the payload (for exit re-checks); acl_index goes into the sidecar
                        ▼
        ┌── Gate 1: hard filter at the recall layer, acl_filter (pushed down into each prefetch)
        │
        ├── Gate 2: small-to-big material-pulling gate (acl_index same-ACL equivalence class)
        │
        └── Gate 3: per-result acl_admits re-check at the exit (every delivery path, including by-id direct reads)
                        ▲
   Service-layer identity (keys/legacy/open) resolves a User{tenant, principals} for every request, fed to the three gates above
```

The layering principle comes from [DESIGN.md D5/D12](../DESIGN.md): **identity lives in the service layer, ACL lives in the retrieval layer.** The service layer only answers "who's asking," turning it into a `User(tenant, principals)`; as for "what can they see," that's uniformly resolved in one place, the embedder. This layering lets the HTTP, MCP adapter, and stdio direct-connect entry points all share the same fail-closed model, instead of each writing its own.

### 2.1 Index time: stamping, splitting, and preventing smuggling

**Stamping is fail-closed.** When `Chunker.chunk` is called without an acl, it stamps the default `RESTRICTED_ACL = {"visibility": "restricted", "allow": [], "unset": True}` — a document with no declared permissions is visible to no one, not to everyone ([src/chunker/core.py:36](../../src/chunker/core.py#L36), [src/chunker/core.py:340](../../src/chunker/core.py#L340)). Every chunk gets a `deepcopy` ([src/chunker/core.py:348](../../src/chunker/core.py#L348)), allowing per-chunk tightening without cross-contamination.

**acl and doc_meta are separate channels.** doc_meta is a convenience channel for "filtering + citation," never a security channel. `extract_doc_meta` uses an `_ACL_KEYS` blacklist to refuse letting the manifest smuggle permission-semantic keys like `acl/tenant/allow/visibility` into doc_meta ([src/chunker/meta.py:18](../../src/chunker/meta.py#L18), [src/chunker/meta.py:62-63](../../src/chunker/meta.py#L62-L63)) — otherwise, if some downstream component mistakenly reads doc_meta as permissions, a poisoned manifest could forge a policy.

**acl_split: turning a free-form dict into filterable fields.** At index time, [src/embedder/embed.py:69](../../src/embedder/embed.py#L69) calls `acl_split`, breaking chunk.acl into 4 payload fields: `acl_unset / acl_tenant / acl_allow / acl_visibility` ([src/embedder/acl.py:14-26](../../src/embedder/acl.py#L14-L26)). There's a detail here that adversarial review caught (seal#1): **an empty-string tenant is always marked `unset=True`** — if an empty tenant were allowed through, it would self-match against "a user with an empty tenant," and tenant isolation would be effectively meaningless, leaving only the public visibility flag as the sole gate — i.e. fail-open. A document that can't be safely attributed to a tenant is denied by default; to genuinely make it global-public, that has to be declared explicitly.

### 2.2 Gate 1: the hard filter at recall, and the "embedded fusion drops should" pitfall

**Filter structure.** `acl_filter` constructs: `must=[acl_unset==False, acl_tenant==user.tenant, a nested Filter(should=[allow∩principals, visibility==public])]` ([src/embedder/store.py:60-81](../../src/embedder/store.py#L60-L81)). Note that `(allow OR public)` uses a **nested Filter**, not a should clause flattened to the top level — the semantics of mixing must+should at the top level aren't guaranteed to be consistent across engines/versions, and "parentheses cannot be flattened" is treated as an explicit contract. Optional filters like doc_ids/doc_type/kind are always appended into must: AND'd with the ACL, only ever narrowing, never loosening.

**The most important pitfall in this piece: embedded QdrantLocal, in RRF fusion mode, silently drops the top-level `query_filter`'s should clause.** Hybrid retrieval's structure is dense/sparse each getting their own Prefetch recall, fused by a top-level `FusionQuery(RRF)`. Measurement (a minimal reproduction from a diagnostic script) found: in fusion mode, only the top-level filter's must equality conditions remain in effect, and `(allow OR public)` vanishes entirely — the ACL degrades to filtering by tenant only, and **within the same tenant, unauthorized documents keep getting recalled as normal — fail-open.** A single-route direct query (no fusion) doesn't hit this pitfall, so testing dense alone would completely miss it.

The fix isn't to route around fusion — it's to push the ACL filter **down into every single Prefetch** ([src/embedder/store.py:118-120](../../src/embedder/store.py#L118-L120)): fusion only fuses already-filtered results, filtering still happens at the recall layer, and the limit isn't contaminated by unauthorized results; the top-level `query_filter` is kept as a second layer of insurance ([src/embedder/store.py:121-123](../../src/embedder/store.py#L121-L123)). When Stage D migrated to Qdrant server mode, the server's behavior was not assumed to match the embedded mode's — a raw fusion probe (asserting directly against the server's raw output, bypassing the exit re-check) re-verified it, and `test_server_fusion_no_should_leak_raw` is now a standing regression test.

The general lesson from this story: **the infrastructure component carrying your security semantics — its behavioral differences (embedded vs. server, version upgrades) are themselves a security surface.** "The filter was passed in" doesn't mean "the filter took effect" — it has to be measured with a leak assertion.

### 2.3 Same semantics, two implementations: acl_admits and acl_filter as mirror images

The ACL semantics are concentrated in a file under 40 lines long ([src/embedder/acl.py](../../src/embedder/acl.py)): `acl_split` is the index-time decomposition, `acl_admits` is the client-side equivalent predicate at query time ([src/embedder/acl.py:29-36](../../src/embedder/acl.py#L29-L36)), and the two are **strictly the same semantics** as the server-side `store.acl_filter`. Why must they be the same semantics? Because small-to-big re-pulls original text from the sidecar by idx range, and the judgment of "can this element go into the big-block" uses `acl_admits` — if it's even slightly looser than the server-side filter, it's a bypass of the hard filter. This isomorphism is locked down by a dedicated unit test (`tests/engine/test_acl.py::test_split_admits_same_semantics`); changing one side without the other turns it red immediately.

### 2.4 Gate 2: small-to-big's acl_index equivalence-class material-pulling

This is the core finding of the chunker-side security audit; the audit's own phrase for it was "**production is closed, retrieval is open.**" Both chunk-production paths are fail-closed, but `assemble_big` re-pulls material from the original elements by idx range *after* the hard filter has run, and can cross chunk boundaries — **same document ≠ same ACL.** Hitting one public section, small-to-big will pull sibling sections in the same idx range that were tightened per-chunk right into big.text, in plain text. The old contract, "a big-block only ever pulls from within the same document, so it can't escalate privilege," was empirically disproven.

The fix is a three-part set:

1. **acl_index**: `ChunkResult.acl_index()` builds a `{element idx → the acl of the chunk that produced it}` map; if the same idx appears in two chunks, the stricter one is kept (`_stricter`, compared by openness, [src/chunker/types.py:7-19](../../src/chunker/types.py#L7-L19), [src/chunker/types.py:93-104](../../src/chunker/types.py#L93-L104)). It's persisted with the sidecar ([src/embedder/embed.py:116-117](../../src/embedder/embed.py#L116-L117)).
2. **Equivalence-class gating**: `assemble_big` by default only takes elements where `acl_index[i] == hit_acl` — only admitted if it's **exactly equal** to the hit chunk's ACL, with unknown idx's fail-closed excluded ([src/chunker/retrieve.py:102-105](../../src/chunker/retrieve.py#L102-L105)). Equivalence-class equality (list-order-sensitive) is deliberately conservative: the direction favors safety, at the cost of pulling in slightly less material in extreme cases.
3. **A verifiable acl marker at the exit**: `BigBlock.acl` records the assembled text's effective ACL; the legacy unguarded mode explicitly sets `acl=None`, meaning "this text has **not been access-checked**" ([src/chunker/types.py:116-118](../../src/chunker/types.py#L116-L118)). There's also a misuse guard: if `admit=` (a caller-supplied custom visibility predicate) is passed, acl_index must be passed too, or it raises immediately ([src/chunker/retrieve.py:94-98](../../src/chunker/retrieve.py#L94-L98)) — because without a per-element acl, admit can only judge by hit_acl, yet it would still stamp big.acl as "verified," constituting a silent cross-ACL leak.

Measured effect (a real corpus, a government document, 568 elements / 80 chunks, with one chunk artificially tightened): the legacy path leaked plaintext of tightened content in 3/79 big-blocks; after the fix, **0/79, and 78/79 blocks still grew normally** — safety wasn't achieved at the cost of recall. There's also a reverse "contract lock" test: `test_assemble_big_legacy_path_is_unguarded` asserts that the legacy path **necessarily leaks** — if anyone in the future silently makes the unguarded mode the default, the test turns red. This technique of using a test to lock down the shape of a known vulnerability is itself worth remembering.

A supporting integrity assertion: both `assemble_big` and sidecar loading enforce `elements[i].idx == i` (dense and ordered, [src/chunker/retrieve.py:118-120](../../src/chunker/retrieve.py#L118-L120), [src/embedder/retrieve.py:89-90](../../src/embedder/retrieve.py#L89-L90)) — material-pulling and ACL gating both index by position, so a sparse/out-of-order sidecar would misassign unauthorized elements as visible; it's better to fail loudly than risk that.

### 2.5 Gate 3: a second check at the exit (all delivery paths)

Gate 1 blocks retrieval, Gate 2 blocks material-pulling, but three more kinds of path could still bypass them: by-id direct reads, engine behavioral differences under fusion, and sidecar assembly. So **every bare-hit/payload delivery path re-checks `acl_admits` per result at the exit**, a mode-agnostic hard backstop (INTEGRATION iron rule 5):

- `hybrid_search` re-checks the nested acl of every point before returning; a missing acl is treated as `{}` → denied ([src/embedder/store.py:125-129](../../src/embedder/store.py#L125-L129));
- `list_documents` re-checks every entry during the scroll — the nested should's behavior under the scroll path has never been independently verified, and fail-closed doesn't bet on it ([src/embedder/store.py:131-158](../../src/embedder/store.py#L131-L158));
- `get_by_chunk_id` fetches directly in O(1) by `uuid5(chunk_id)`, **completely bypassing acl_filter, so it must be re-checked**; both "doesn't exist" and "no access" uniformly return None, without distinguishing the two, to prevent probing ([src/embedder/store.py:160-172](../../src/embedder/store.py#L160-L172));
- `search_with_context` checks `big.acl` on the assembled big-block, and `acl=None` (unverified) is always refused for delivery, marked with the status `single_chunk_acl` ([src/embedder/retrieve.py:157-160](../../src/embedder/retrieve.py#L157-L160));
- `expand` is a complete microcosm of the three gates: `get_by_chunk_id` re-checks → `assemble_big`'s per-element `==hit_acl` gate → the exit `acl_admits(big.acl)` ([src/embedder/retrieve.py:244-259](../../src/embedder/retrieve.py#L244-L259)).

A subtle counterexample shows this system isn't just "lock everything, mindlessly": `_load_sidecar`'s document-level precheck is used **only by doc-id direct-read tools** (triggered when get_document/get_outline pass a user); hit-driven paths (_assemble/expand) **deliberately don't pass a user** ([src/embedder/retrieve.py:76-83](../../src/embedder/retrieve.py#L76-L83), [src/embedder/retrieve.py:186-197](../../src/embedder/retrieve.py#L186-L197)) — because `_stricter` will rewrite the hit chunk idx's acl in acl_index into a stricter sibling block's acl, and a doc-level precheck on the hit path would wrongly reject a public hit that should have been delivered (confirmed empirically in the B2 adversarial review). **Fail-closed doesn't mean "block everywhere, one more time"** — blocking in the wrong place manufactures false-denies, and the urge to fix a false-deny (by loosening the predicate) is exactly the seed of a future leak.

### 2.6 "Zero recall even with the exit gate disabled": how to prove one layer of defense-in-depth is effective on its own

This subsection stands on its own because it's the most methodologically valuable step in this whole security design.

**Defense-in-depth has a testing paradox: a backstop can mask a regression in the first line of defense.** Suppose the exit re-check (Gate 3) is working correctly — then regardless of whether the recall-layer push-down (Gate 1) is correct, end-to-end tests will always show "0 leakage" — Gate 1 could quietly break (say, a future refactor moves the filter from prefetch back to the top level) and every test would still be green. This is exactly what the R5.M2 adversarial review pointed out: the existing unauthorized-access assertions trivially pass when the exit gate is present, and can't prove the true statement "the prefetch push-down itself is what's blocking unauthorized access."

The solution is **isolated testing**: section 5 of [eval/acl_regression.py](../../eval/acl_regression.py) monkeypatches the exit `acl_admits` to always return True — **deliberately disabling the backstop** — and then, using the sentinel original text of four documents across five identities as exact queries, asserts that cross-tenant/unauthorized content still recalls 0 results ([eval/acl_regression.py:123-138](../../eval/acl_regression.py#L123-L138)):

```python
_store.acl_admits = lambda acl, user: True   # disable the exit re-check
# ... assert the leaked set under hybrid (RRF fusion) is still empty → proves the prefetch push-down itself is effective
```

The whole proof structure of acl_regression is worth remembering in full — it has four interlocking layers of assertions:

1. **Sentinel exact queries**: search for each document's unique original text as the query — even an exact match must recall 0 results, proving "the ACL hard filter comes before relevance," rather than passing by the luck of "it happened not to be found";
2. **Positive authorization**: an authorized identity must be able to find what it should be able to see — guarding against "false-pass by denying everyone" (a system that rejects everyone equally would also pass a leakage test);
3. **Direct-read surface coverage**: get_document with no access → PermissionError, expand across ACL boundaries → None;
4. **Isolated testing with the exit gate disabled**: as above, independently proving the first layer is effective on its own.

The same idea was reused when migrating to Qdrant server mode in Stage D: `test_server_fusion_no_should_leak_raw` bypasses the exit re-check and asserts directly that the server's fusion **raw output** contains no unauthorized points — proving that the "embedded mode drops should" pitfall didn't reappear on the server, and that this isn't just the exit gate masking it. **Security assertions must test true statements: every line of defense needs a test where "the other lines of defense are not present."**

### 2.7 Service layer: three identity modes + the fail-closed iron rule

Everything at the retrieval layer takes `User(tenant, principals)` as its input, and the service layer's job is to authoritatively produce it ([docs/DESIGN.md D10](../DESIGN.md)). Three modes:

- **keys** (for teams, the recommended default): `CUSTODIAN_KEYS_FILE` points to a JSON file, and every request's `X-API-Key` is resolved into an `Identity{name, tenant, principals, admin}`; unknown/missing keys always return 401, and it never leaks "whether a key ever existed" ([src/custodian/service.py:148-164](../../src/custodian/service.py#L148-L164));
- **legacy**: a single `CUSTODIAN_API_KEY` threshold, with a single identity bound at startup;
- **open**: neither is set, restricted to loopback addresses only.

Fail-closed operates on three levels, all failing loudly **at startup** rather than silently allowing access at runtime:

1. **Tenant unset** → every toolcore tool entry point checks `user.tenant` first, and returns an empty `no_identity` result if it's empty ([src/custodian/toolcore.py:215-216](../../src/custodian/toolcore.py#L215-L216));
2. **Binding to a non-loopback address without keys mode** → `create_app` calls SystemExit directly, refusing to start ([src/custodian/service.py:93-95](../../src/custodian/service.py#L93-L95)). This is called "**authorization at deployment**": whoever can connect to the port = the full visible content of that identity, so it's never allowed to expose the whole knowledge base to a LAN under a single identity or with no authentication;
3. **Any malformed keys file** (too-short keys, missing name/tenant, duplicate names, a name containing `|`, duplicate keys) → SystemExit, never silently degraded ([src/custodian/identity.py:29-64](../../src/custodian/identity.py#L29-L64)).

In keys mode, `_current_user` builds a fresh engine User **per request**, based on the identity resolved ([src/custodian/service.py:128-133](../../src/custodian/service.py#L128-L133)) — there's no process-level shared state anywhere along the chain from the HTTP header to the Qdrant filter, and alice's and bob's concurrent requests each carry their own tenant into the retrieval stack (test_team.py has an assertion that identity really does flow all the way into the engine).

### 2.8 Session isolation: a namespace proof behind one input validation

Per-session dedup (a segment already delivered is only referred to by a pointer next time) is a convenience feature, but under multi-user usage it becomes an information boundary: content that user A already retrieved must never be mislabeled `already_returned` for user B (who never received the body text at all). The design:

- Dedup is **opt-in**: it only activates if the request carries an `X-Custodian-Session` header; without it, no dedup happens (a one-off curl call shouldn't carry cross-call state). The MCP adapter generates a uuid per process for the session header ([src/custodian/mcp_adapter.py:26](../../src/custodian/mcp_adapter.py#L26)), and under stdio, "one process = one session" holds naturally;
- The registration key is `f"{identity name}|{session id}"` ([src/custodian/service.py:204-208](../../src/custodian/service.py#L204-L208)): **even if different users forge the same session id, they remain invisible to each other**;
- SessionRegistry is a bounded LRU (64 sessions), and eviction only loses the dedup convenience, not correctness ([src/custodian/sessions.py:16-32](../../src/custodian/sessions.py#L16-L32)).

Now look back at two seemingly trivial identity-validation rules: names can't contain `|`, and names must be unique ([src/custodian/identity.py:51-58](../../src/custodian/identity.py#L51-L58)). These aren't style pedantry — they're **derivable**: the registration key is a `name|sid` concatenation, and if a name contains `|`, then `("a", "b|c")` and `("a|b", "c")` collide on the same key — a namespace collision; if names duplicate, two different identities would share a dedup namespace — cross-contamination. Being able to reverse-engineer "a provable, unambiguous namespace" from "one input validation rule" is good material for demonstrating design depth in an interview.

### 2.9 No-leakage rules: errors, probes, and logs

The last mile of a permission system is "say nothing extra on failure":

- **No access and doesn't exist get the same response**: `_safe_doc_call` maps a PermissionError to `no_access`, with the message uniformly "no access permission, **or it doesn't exist**" ([src/custodian/toolcore.py:198-206](../../src/custodian/toolcore.py#L198-L206)); the store layer's get_by_chunk_id does the same (None doesn't distinguish the two cases). At the same time, note it maps FileNotFoundError/ValueError to `config_error`, **not** no_access — a corrupted sidecar is "visible but the index is broken," and disguising that as "no access" would dress up an operational problem as a permissions problem; these are two opposite-direction errors and must never be merged;
- **Exceptions are never thrown raw at the agent**: a ValueError from sidecar version drift once bubbled its exception text, including absolute paths, up to the agent (an info leak); after the fix it degrades to `single_chunk_degraded`, with the details going only into server-side logs ([src/embedder/retrieve.py:153-156](../../src/embedder/retrieve.py#L153-L156));
- **Probe information boundary**: /healthz and /readyz are unauthenticated (orchestration/nginx need to probe with no key), so their response bodies are a reconnaissance surface — readyz's error case doesn't return `str(e)` (or it would leak the internal qdrant/inference host:port), and a missing collection returns only `collection_missing`, never the collection's name ([src/custodian/service.py:224-256](../../src/custodian/service.py#L224-L256));
- **Log privacy**: the key itself is never written to disk (only the identity name is logged), the query is truncated to 120 characters by default and can be turned off entirely, with truncation done in a single place before entering the queue;
- **Aggregate information is also gated**: `/v1/stats` requires an admin key in keys mode — "who's querying which endpoint at what frequency" is itself information ([src/custodian/service.py:258-268](../../src/custodian/service.py#L258-L268)).

---

## 3. Why it's designed this way: rejected alternatives

| Alternative | Reason for rejection | Evidence |
|---|---|---|
| **Client-side filtering after recall** | A fail-open window (any bug in the filter code leaks directly) + limit contamination (unauthorized results occupy top-k slots). Doing it at the recall layer is what makes it fail-closed: unauthorized content never enters the candidate set at all | The embedded-fusion-drops-should incident is itself proof that "a client-side backstop isn't reliable" — what actually blocked the leak was the prefetch push-down, not the exit gate |
| **Flattening must+should at the top level** (not using a nested Filter) | Mixing must+should at the top level has semantics that aren't guaranteed to be consistent across engines/versions — "parentheses cannot be flattened" is an explicit contract | Confirmed by the fusion-drops-should measurement; with the nested structure + push-down, acl_regression's 65 assertions show 0 leakage |
| **small-to-big using `admit=acl_admits(user)`** (take every element visible to the user) | seal#2: under the admit path, `big.acl` can only be stamped as hit_acl, under-reporting the stricter content actually present in big.text, making the exit check meaningless; equivalence-class material-pulling makes `big.acl = hit_acl` naturally accurate and verifiable at the exit | Real-corpus leakage went from 3/79 → 0/79, with 78/79 blocks growing normally (recall was barely hurt) |
| **Implementing deny semantics** | Consistent with the chunker INTEGRATION contract: to exclude a group, remove it from allow. deny was a candidate for "the most poisonous kind of contract failure" — a field that exists but silently doesn't work is more dangerous than not having it at all, so it was removed from the schema entirely with an explicit warning | A contract-layer decision (see [INTEGRATION.md](../components/chunker/INTEGRATION.md)) |
| **Wrapping RESTRICTED_ACL in a MappingProxyType to prevent tampering** (a suggestion from security review) | `deepcopy(mappingproxy)` raises TypeError on Py3.12, which would turn the fail-closed default path into a crash path; the existing deepcopy isolation was measured to withstand five classes of contamination attacks | Even a review's suggestion gets adversarially verified, not accepted wholesale |
| **SSO/OIDC, storing keys in a database, hot-reloading keys** | Not worth introducing an IdP dependency at the current team scale; a file + restart (seconds) is sufficient; hot-reload adds a state-consistency surface | [DESIGN.md D10](../DESIGN.md) rejection list |
| **A separate collection per tenant (physical isolation)** | A single collection + push-down filtering has already been proven, by regression, to have 0 leakage at the current scale; the operational cost of physical isolation (migration/multiple replicas/evaluation baselines × number of tenants) is disproportionate; but it remains an available upgrade path for stronger isolation | A scale trade-off, not a categorical rejection |

One sentence summarizing the design philosophy: **converge security semantics into the fewest possible places (one file, acl.py, plus three gates), then lock every one of them down with "a test that can turn red"**, rather than scattering permission checks throughout the codebase.

---

## 4. Real-world retrospective: ACL/security-related items from this review round

This writing-phase adversarial review round (34 confirmed) was routed by discipline: behavior-neutral robustness fixes landed directly, while anything that would change output content, or needed GPU verification, was deferred. Security-related items happen to have hit all three outcomes, each its own teaching point.

### Fixed: consolidating healthz information (fixes_applied #8)

- **Symptom**: the unauthenticated `/healthz` endpoint exposed fields like `collection` and `llm_model`; the same file's `/readyz`, per the sec-2 review conclusion, deliberately **doesn't return the collection name and doesn't return `str(e)` on error**.
- **Root cause**: the same information boundary was implemented to two different standards across two endpoints — the hard-won "unauthenticated probes don't leak internal information" on readyz was simply bypassed by healthz. This kind of "inconsistency" is more worth worrying about than a single-point defect: it shows no one treated the boundary as a systemic discipline — each endpoint just did its own thing.
- **Fix**: healthz was consolidated to a minimal liveness response `{status, service, version, tenant_bound, uptime_s}` ([src/custodian/service.py:211-222](../../src/custodian/service.py#L211-L222)), with sensitive fields moved into the admin-gated `/v1/stats` ([src/custodian/service.py:264-267](../../src/custodian/service.py#L264-L267)); docs/API.md was updated to match.
- **Test**: a negative assertion on the healthz response body (doesn't contain collection/llm_model), guarding against regression.

The same batch also has a security-adjacent item, #7: the stats keys previously used the raw URL path, and **even unauthorized 401 requests were counted** — anyone on the network could slowly bloat the process's memory using random paths (a low-rate DoS). The fix switches to using the route template as the key, merging unmatched routes into a fixed bucket, so the key set is naturally bounded ([src/custodian/service.py:185-191](../../src/custodian/service.py#L185-L191)); `test_stats_unauthorized_requests_bounded` locks this down.

### Deferred: big-blocks on the ACL-aware path systematically losing headings (deferred chunker#0)

- **The confirmed facts**: heading elements never enter any chunk's source_indices, and so aren't in acl_index; the equivalence-class gate fail-closed excludes unknown idx's → the production default path's big.text **contains no heading lines at all**. The same root cause was already patched in get_document (additionally including headings for any subsection visible in its own-section context, [src/embedder/retrieve.py:212-232](../../src/embedder/retrieve.py#L212-L232)), but the assemble_big/expand path was never synced up.
- **Why it's confirmed but not fixed right away**: fixing it would change big-block content → both retrieval delivery and the evaluation numbers would move → it needs a bump of SIDECAR_VERSION, an index rebuild, and a GPU eval re-run before it can land. Fixing it "along the way" within the writing window would mean already-published evaluation baselines drift out of sync. **"Confirm → schedule → attach a fix sketch" is itself an engineering judgment**: the fail-closed direction error here (leaking out, not leaking in) is a quality defect, not a security vulnerability, and it can wait for a proper rebuild window.
- **Teaching point**: this is the **cost side** of fail-closed. The direction was chosen correctly (headings leak out, rather than restricted headings leaking in), but there's a real cost — the context the LLM receives is missing section-boundary signals, and when climbing to a parent section, sibling body text gets joined together with nothing between it. The quality cost of a safe default has to be managed explicitly, not pretended away.

### Refuted: readyz bypassing Store._lock (service#0)

An analyst reported that "/readyz directly accesses the embedded Qdrant client, bypassing Store._lock, racing against the business path on a non-thread-safe object." Adversarial verification **refuted** it: the embedded QdrantLocal's unsafety comes only from the write path (np.append rebinding the array), and during the serve process's runtime there is **zero write path** (indexing happens in a separate CLI process, mutually exclusive via a file lock), and `collection_exists` is only an in-process dict membership read — read+read is safe within the declared concurrency model. This was logged for the record as "if an online write endpoint is introduced in the future, it must switch to a lock-holding method." — **The value of adversarial review isn't just catching bugs, it's also blocking "looks safer" fixes that would add no real value**; every unnecessary lock is a tax on availability.

---

## 5. How to pitch this in an interview

### 30-second version (elevator pitch)

> I built complete enterprise-grade ACL into a RAG system: permission filtering happens at the **recall layer**, not after recall, with the filter pushed down into every prefetch of the vector store; because small-to-big re-pulls material from the original text after the hard filter runs, we designed acl_index equivalence-class gating, plus a second check at the exit of every delivery path — defense in depth across three gates. During this, measurement caught embedded Qdrant **silently dropping the filter's should clause under RRF fusion**, degrading the ACL to fail-open; the fix for it also produced a testing methodology: monkeypatch the exit backstop to always return True, running an isolated test to prove the first line of defense is effective on its own, rather than being masked by the backstop. Real-corpus regression: cross-tenant/unauthorized content recalls 0 results, and small-to-big leakage went 3/79 → 0/79, with recall barely hurt.

### 3-minute version (structured expansion)

1. **Frame the problem** (30s): RAG shatters file-level permissions into a vector store, and similarity-based retrieval naturally "favors" unauthorized content; and RAG has three particular bypass surfaces — context expansion, by-id direct reads, and error messages leaking existence. So single-point filtering isn't enough — you need defense in depth.
2. **Layering** (30s): identity at the service layer (API key → Identity, three modes, non-loopback binding forces multi-identity auth, a bad config refuses to start), authorization at the retrieval layer (User{tenant, principals} → the hard filter). The two layers are orthogonal, and the three entry points (HTTP/MCP/stdio) share the same model.
3. **The three gates** (60s): ① the recall-layer filter pushed into every prefetch — tell the story of the fail-open incident with embedded fusion dropping should, and its fix; ② small-to-big's acl_index equivalence-class material-pulling — "same document ≠ same ACL," data point: real-corpus leakage went 3/79→0/79 with 78/79 blocks growing normally; ③ the per-result exit re-check, with by-id direct reads giving "no access and doesn't exist" the same response to prevent probing.
4. **Proof method** (45s): the testing paradox of defense-in-depth — a backstop masks a regression in the first layer. acl_regression's four-layer assertions: sentinel exact queries proving "filtering comes before relevance," positive authorization guarding against false-pass-by-denying-everyone, direct-read surface coverage, and **the isolated test with the exit gate disabled**; when migrating to server mode, a raw fusion probe re-verifies it without assuming the behavior is the same.
5. **Closing** (15s): fail-closed has a real cost (the ACL path losing headings in big-blocks, already confirmed, scheduled for a fix) — we manage that cost explicitly instead of hiding it — this sentence usually steers the interviewer right into the honest-boundaries section you've already prepared.

---

## 6. Rehearsing follow-up questions

**Q1: Why must filtering happen at the recall layer? Wouldn't a bigger buffer at post-filtering work?**
Point to make: three unfixable flaws — ① fail-open: any bug in the filtering code leaks directly, whereas under recall-layer filtering, a bug shows up as "fewer results" (fail-closed); ② the limit/buffer is a guess: the proportion of unauthorized documents isn't bounded, and top-50 might be entirely unauthorized; ③ "fetch it, then discard it" is itself unauthorized reading, and won't pass an audit/compliance review. Keywords: fail-open vs. fail-closed, limit contamination. You can add: custodian does have an exit re-check, but its role is a backstop, not the primary line of defense.

**Q2: How was the fusion-drops-should pitfall discovered? How do you know it's engine behavior and not your own code's bug?**
Point to make: while validating BM25's Modifier.IDF support, the hybrid unauthorized-access assertion was run alongside it and found unauthorized documents in the same tenant being recalled; a **minimal reproduction script** isolated the variable down to "fusion mode × should clause" (single-route direct query doesn't hit it, must equality conditions still work) → pinpointed to embedded QdrantLocal's fusion implementation dropping the top-level query_filter's should clause. The fix chose "push down into prefetch" rather than "avoid fusion," because filtering has to stay at the recall layer. Keywords: minimal reproduction, variable isolation, behavioral differences are themselves a security surface.

**Q3: Could the three gates end up masking each other? How do you know each one is actually working?**
Point to make: this is exactly what the R5.M2 review caught — with the exit gate present, the recall layer's regression tests trivially pass. The solution is isolated testing: monkeypatch the exit `acl_admits` to always return True, and assert 0 leakage still holds → proving the prefetch push-down is effective on its own; when migrating to server mode, a raw probe bypasses the exit and looks directly at the engine's raw output. General principle: **every line of defense needs a test where "the other lines of defense are not present."** The other half is guarding against "false-pass by denying everyone" — a positive-authorization assertion that an authorized identity must actually be able to recall its own content.

**Q4: Why does small-to-big bypass the hard filter? Why is equivalence-class material-pulling safer than "take every element visible to the user"?**
Point to make: big-blocks pull material from the original elements by idx range, crossing chunk boundaries — "same document ≠ same ACL." The fatal flaw of the admit (user-visible) approach is that big.acl can't accurately express the effective ACL of mixed content (it can only be stamped as hit_acl, under-reporting stricter content present within it), and the exit check gets hollowed out; the equivalence class (==hit_acl) makes big.acl naturally accurate and verifiable at the exit, with unknown idx's fail-closed. Data: 3/79→0/79, with 78/79 growing normally. Keyword: verifiability at the exit takes priority over completeness of the pulled material.

**Q5: Why must "no access" and "doesn't exist" get the same response? Isn't distinguishing them more user-friendly?**
Point to make: distinguishing them leaks existence — an attacker can enumerate chunk_ids/doc_ids to learn "there's an X here I can't see." For trusted operators, the details go into server-side logs; for untrusted agents/clients, it's uniformly no_access/None. Note the counter-example in the other direction: a corrupted sidecar maps to config_error, not no_access — disguising an operational failure as a permissions issue would send someone down the wrong troubleshooting path — "no leakage" doesn't mean "pretend every error is a permissions issue."

**Q6: Why not use SSO/OIDC, RBAC, or database row-level permissions?**
Point to make: this is a trade-off based on scale, with an upgrade path left in place. The identity layer is a thin, replaceable layer (a keys file → swapping in an IdP later doesn't touch the retrieval layer, because the layering is genuinely orthogonal); the authorization model is tenant + groups∩allow + public, sufficient to cover "department/project team" granularity; what's rejected is operational cost (an IdP dependency, the state-consistency surface of hot-reload), not the concept itself. A good self-test question: if SSO had to be integrated tomorrow, which lines would change? The answer is: swap the implementation of the identity module, and the `_current_user` contract stays the same — which shows the layering is real.

**Q7: How do you guarantee this ACL setup doesn't regress across multiple replicas / a different storage backend?**
Point to make: ① the ACL semantics are single-sourced (acl.py) + the two implementations are locked isomorphic by a unit test; ② acl_regression is an end-to-end regression (a real Chunker+Embedder populating a synthetic 2-tenant store, a five-identity matrix), rerun after switching backends; ③ no behavioral assumptions are made about a new backend — server mode specifically added a raw fusion probe; ④ cross-replica state like session dedup is explicitly declared "a convenience, not a correctness requirement" — degrading to 1/N under round-robin is acceptable (correctness doesn't depend on it).

**Q8: What are this design's known weaknesses right now?** (leads into the next section)

---

## 7. Hands-on experiments

### Lab 1 (CPU): reproduce the "embedded fusion drops should" fail-open by hand

Prerequisite: the repo already `pip install -e .`'d (needs qdrant-client, jieba; under WSL, `conda activate custodian`). The principle is in §2.2; the script uses an `:memory:` store and random dense vectors (no GPU/model needed), with five points covering public/restricted/cross-tenant/unset ACL types:

```bash
python - <<'PY'
import random
from qdrant_client import models
from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User
DIM = 8; _v = lambda: [random.random() for _ in range(DIM)]
A = lambda t, a, vis, u=False: {"tenant": t, "allow": a, "visibility": vis, "unset": u}
def pt(i, text, acl):
    sp = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
          "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _v(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **sp})
s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
s.ensure_collection()
s.upsert([pt(1, "public revenue", A("t1", [], "public")),
          pt(2, "hr revenue",  A("t1", ["g_hr"], "restricted")),
          pt(3, "fin revenue", A("t1", ["g_fin"], "restricted")),
          pt(4, "t2 revenue",  A("t2", ["g_hr"], "restricted")),
          pt(5, "unset revenue", A("t1", [], "restricted", True))])
acl = s.acl_filter(User("t1", ["g_hr"])); qs = query_sparse("revenue")
leak = s.client.query_points("t",
    prefetch=[models.Prefetch(query=_v(), using="dense", limit=20),
              models.Prefetch(query=qs, using="sparse", limit=20)],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=acl, limit=10, with_payload=True).points
print("top-level filter only (not pushed down):", sorted(p.payload["chunk_id"] for p in leak))
ok = s.client.query_points("t",
    prefetch=[models.Prefetch(query=_v(), using="dense", filter=acl, limit=20),
              models.Prefetch(query=qs, using="sparse", filter=acl, limit=20)],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=acl, limit=10, with_payload=True).points
print("filter pushed into every prefetch:", sorted(p.payload["chunk_id"] for p in ok))
PY
```

**Expected**: the first line (relying only on the top-level query_filter) will include `c3` — a same-tenant document whose user is not in the allow group still gets recalled, because the nested should was silently dropped by fusion, leaving only must (tenant/unset) in effect — i.e. fail-open; the second line (pushed down into every prefetch) leaves only `c1/c2` — 0 leakage. What you're seeing with your own eyes is exactly the pitfall described in the [store.py:103-105](../../src/embedder/store.py#L103-L105) comment, and why the top-level filter can only ever be a second layer of insurance.

### Lab 2 (CPU): walk through the guard tests for all three gates with pytest

```bash
# ① ACL semantics: dual-implementation isomorphism + fail-closed decomposition
python -m pytest -q tests/engine/test_acl.py -v
# ② small-to-big three states: the equivalence class blocking cross-ACL siblings / the legacy path's "must leak" contract lock / admit requires acl_index
python -m pytest -q tests/engine/test_core.py -k "cross_acl or legacy_path or admit_requires" -v
# ③ store layer: hard filter / by-id re-check (no access and doesn't exist both return None) / list_documents scoping
python -m pytest -q tests/engine/test_store.py -k "acl" -v
# ④ service layer: keys validation (including '|'/duplicate-name rejection at startup) / cross-user isolation of forged session ids / logs don't record keys
python -m pytest -q tests/test_team.py -v
```

Worth reading closely: `test_assemble_big_legacy_path_is_unguarded` (locking down a known vulnerability's shape with a test) and `test_keys_mode_session_isolated_across_users` (two users forging the same session id remain invisible to each other, verifying the `identity name|` prefix). All of it runs with no GPU, no network.

### Lab 3 (GPU/WSL): the end-to-end ACL regression, including the "exit gate disabled" isolation section

Prerequisite: WSL + `conda activate custodian` (needs the Qwen3-VL models and a 4090; the script builds a fresh 2-tenant synthetic store and populates it with real vectors):

```bash
conda activate custodian && python eval/acl_regression.py
```

**Expected**: all 65 assertions pass, exit code 0. Pay special attention to section 5's output — the assertions prefixed `[exit gate disabled]` show 0 leakage even after `acl_admits` has been forced to always return True — this is exactly the "proving the first line of defense is effective on its own" discussed in §2.6.

---

## 8. Honest boundaries

Proactively admitting these in an interview is far stronger than having them dug out of you:

1. **Big-blocks on the ACL-aware path lose all headings** (already confirmed, deferred for a fix). The fail-closed direction is correct (leaking out, not leaking in), but the context the LLM receives is missing section-boundary signals, with only the breadcrumb offering partial compensation; get_document has already been fixed, and assemble_big/expand are waiting for the same window (bump SIDECAR_VERSION + rebuild + GPU eval) to land together. Talking point: "This is an explicit cost of fail-closed. We chose to log it and schedule it, rather than quietly changing output outside the evaluation baseline."
2. **The permission model is coarse-grained**: tenant + group∩allow + public, with no deny, no field-level/row-level permissions, no time-windowed authorization. deny was deliberately not implemented (silently-not-working is more poisonous than not having it), but something like "a user removed from a group whose session is already cached" — revocation timeliness — has no dedicated mechanism; keys-file rotation relies on a restart, which is seconds-scale but not real-time.
3. **The embedded mode's scroll/should behavior hasn't been exhaustively verified**: list_documents' handling of nested should under the scroll path follows a "don't bet on it, re-check every entry" strategy — that is, it relies on the exit gate rather than a verified first line of defense — this is a declared conservative choice, not a proven equivalence.
4. **The query-vector cache is user-agnostic** (ACL is applied after recall), which is safe, but it means "user A's query warms up user B's cache" — an extremely small amount of information leaks through a timing side channel; this hasn't been specifically evaluated.
5. **The threat model has boundaries**: it defends against "authorized-but-over-reaching reads" and "silently running wide open due to an incomplete configuration," not against an attacker who has gotten a server shell (both the sidecar JSON and the Qdrant payload are plaintext), not against poisoned parsed output (only img_path gets path sanitization), and the security of the key file depends on filesystem permissions (chmod 600, best effort — weaker still on Windows).
6. **The exit-gate-disabled test runs in the GPU regression suite, not CI**: acl_regression needs real vectors, and CPU CI only covers per-layer unit tests; "re-proving the first line of defense on every commit" isn't achievable today, and is backstopped by change discipline (must run whenever store/acl is touched).

---

*Anchor line numbers verified against the actual code as of 2026-07-07 (after the adversarial review's fixes had landed); if a future refactor causes line numbers to drift, treat the same-named symbol in the linked file as authoritative.*
