# Integration guide

For **Aditya** (main backend) and **Saravanan** (OpenClaw tooling). This is the
complete contract of the RAG service. Nothing else in this repository needs to
be read to integrate against it.

Base URL: `http://127.0.0.1:8001` (configurable via `RAG_HOST` / `RAG_PORT`).
OpenAPI schema: `GET /openapi.json` · Swagger UI: `/docs`.

---

There are two ways to consume this service. Pick one:

- **HTTP** (§1–8) — the service runs on its own port, you call it with `requests`.
- **In-process** (§11) — you `import retriever` and call three functions. Use
  this if the pipeline already imports a `retriever` module.

Both are the same pipeline underneath and return the same guarantees.

---

## 1. The minimum path

```python
import requests

BASE = "http://localhost:8001"

# 1. ingest once per contract
with open("contract.pdf", "rb") as handle:
    document = requests.post(
        f"{BASE}/v1/documents/ingest",
        files={"file": handle},
        data={"document_id": "contract_123"},   # optional
    ).json()

# 2. extract obligations
result = requests.post(
    f"{BASE}/v1/documents/{document['document_id']}/extract",
    json={},                                    # {} = all obligation types
).json()

if result["can_approve"]:
    show_approve_button(result["obligations"])
else:
    show_blocked_review(result["obligations"], result["failures"])
```

`extract` is synchronous and re-runs the model each time it is called. With the
real model expect seconds, not milliseconds — call it once per contract and
store the result, or read it back later with
`GET /v1/documents/{id}/obligations` (no model call).

---

## 2. `ExtractionResult`

```jsonc
{
  "document_id": "contract_123",
  "obligations": [ /* CandidateObligation, see §3 */ ],
  "can_approve": true,          // false if ANY obligation failed
  "failures": [                 // things that never became obligations
    {
      "obligation_type": "notice_deadline",
      "reason": "missing_verified_inputs",
      "detail": "notice_deadline was not computed because these verified inputs are missing: contract_end_date"
    }
  ],
  "llm_mode": "gpt-oss-120b",   // or "fake" when USE_FAKE_LLM=true
  "retrieval_mode": "bm25_only",// or "hybrid_bm25_vector_rrf"
  "elapsed_seconds": 4.128
}
```

**`can_approve` is the gate.** It is `true` only when at least one obligation
was produced and every one of them is `verified`, `computed` or
`human_verified`. Do not re-derive it client-side; do not let a user approve
when it is `false`.

---

## 3. `CandidateObligation`

```jsonc
{
  "id": "contract_123:termination_notice_period",  // stable: document_id + type
  "document_id": "contract_123",
  "obligation_type": "termination_notice_period",
  "raw_value": "sixty (60) days",                  // as the model read it
  "normalized_value": "P60D",                      // machine-usable, may be null
  "source_evidence": {
    "quote": "…not less than sixty (60) days prior to the end of the then-current term.",
    "page": 3,                                     // 1-based, as printed
    "chunk_id": "chunk_2",
    "start_offset": 251,                           // into GET /pages text
    "end_offset": 437
  },
  "status": "verified",
  "verification_method": "normalized_exact_match",
  "verification_reason": null,                     // populated on failure
  "computation_formula": null,                     // computed fields only
  "computation_inputs": null
}
```

### Statuses

| `status` | Meaning | UI |
|---|---|---|
| `verified` | Quote found on the claimed page and the value parsed | teal badge, approvable |
| `computed` | Produced by Python from verified inputs — `source_evidence` is `null` | amber badge, annotate "calculated in code, not model output" |
| `failed` | Evidence missing/wrong page, or the value could not be parsed | red badge, **blocks approval** |
| `proposed` | Internal transitional state; you should not normally see it | — |
| `human_verified` | A person confirmed or edited the value | set by *your* UI, never by this service |

### `verification_method`

| Value | Meaning |
|---|---|
| `normalized_exact_match` | Quote matched the page exactly after Unicode/whitespace normalization |
| `fuzzy_match` | Matched within the configured similarity threshold (PDF artifact); `verification_reason` carries the score |
| `deterministic_computation` | Not quoted — calculated in code |
| `none` | Nothing verified this value; only ever paired with `failed` |

### `verification_reason` values on failure

| Reason prefix | What happened | Suggested UI |
|---|---|---|
| `quote_not_found_on_page` | The quote is not in the document as claimed (hallucination) | red, require manual entry |
| `unchecked_option` | The quote is real but comes from a form option that was **not** ticked (`☐`) — the value does not apply | red, show the quote so the reviewer sees the empty box |
| `wrong_page: … found on page N` | The quote is real but on another page | offer a one-click page fix |
| `page_out_of_range` | The model cited a page that does not exist | red |
| `quote_too_short` | Under the minimum quotable length | red |
| `value_not_normalizable` | The quote verified but the value could not be parsed (e.g. a date that is not a date) | let a human type the value |

### Obligation types

Canonical names, all `snake_case`:

`contract_start_date` · `contract_end_date` · `automatic_renewal` ·
`renewal_duration` · `termination_notice_period` · `notice_deadline` ·
`renewal_option_notice` · `renewal_option_deadline` ·
`payment_obligation` · `fee_escalation` · `indemnification` ·
`liability_cap` · `governing_law`

**Two notice families, opposite consequences** — surface them differently:

| | Notice | Deadline | Miss it and… |
|---|---|---|---|
| Evergreen contract | `termination_notice_period` | `notice_deadline` | it renews and you pay for another term |
| Renewal option | `renewal_option_notice` | `renewal_option_deadline` | the option lapses and you **lose** the contract |

A contract normally has one family or the other. `renewal_option_deadline`
carries `computation_inputs.consequence` = *"the renewal option lapses after
this date"* so the row can be labelled without inspecting the type.

Aliases accepted on input (`term_end` → `contract_end_date`,
`termination_notice` → `termination_notice_period`, `term_start`, `notice`,
`renewal_term`, `auto_renewal`, `payment`, `fee_increase`, `indemnity`,
`liability`, `law`). Responses always use the canonical name.

### `normalized_value` conventions

| Type | Format | Example |
|---|---|---|
| dates | `YYYY-MM-DD` | `2026-03-31` |
| durations | ISO-8601 | `P60D`, `P12M`, `P1Y6M` |
| money | `<CCY> <amount>` | `USD 120000.00` |
| percentages | `<n>%` | `4%` |
| `automatic_renewal` | boolean string | `"true"` / `"false"` |
| free text | whitespace-collapsed string | `Delaware` |

`normalized_value` is `null` only on `failed` obligations.

---

## 4. Computed fields

`notice_deadline` is **never** taken from the model. It is recomputed from the
verified `contract_end_date` and `termination_notice_period`:

```jsonc
{
  "obligation_type": "notice_deadline",
  "raw_value": "2026-01-30",
  "normalized_value": "2026-01-30",
  "source_evidence": null,
  "status": "computed",
  "verification_method": "deterministic_computation",
  "verification_reason": "calculated in code, not model output",
  "computation_formula": "notice_deadline = contract_end_date - termination_notice_period",
  "computation_inputs": {
    "contract_end_date": "2026-03-31",
    "termination_notice_period": "P60D",
    "operation": "subtract",
    "evaluated": "2026-03-31 - 60 days = 2026-01-30",
    "source_obligation_ids": [
      "contract_123:contract_end_date",
      "contract_123:termination_notice_period"
    ]
  }
}
```

`computation_inputs.evaluated` is a human-readable one-liner suitable for
display next to the value. `source_obligation_ids` lets the UI link the derived
date back to the two quotes it rests on.

If either input is missing or failed, **no deadline is produced** and a
`missing_verified_inputs` entry appears in `failures`.

---

## 5. Highlighting a quote in the UI

`start_offset` / `end_offset` are character offsets into the stored text of the
cited page:

```python
pages = requests.get(f"{BASE}/v1/documents/contract_123/pages").json()["pages"]
page_text = next(p["text"] for p in pages if p["page"] == evidence["page"])
before  = page_text[: evidence["start_offset"]]
marked  = page_text[evidence["start_offset"] : evidence["end_offset"]]
after   = page_text[evidence["end_offset"] :]
```

The offsets are mapped back onto the *original* page text, so they survive the
Unicode/whitespace normalization used during matching.

---

## 6. Re-verifying an edited value

When a human edits a value or fixes a page reference, re-verify before
promoting the row to `human_verified`:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/evidence/verify \
  -H 'Content-Type: application/json' \
  -d '{"document_id": "contract_123", "quote": "written notice of termination", "page": 3}'
```

```jsonc
{
  "document_id": "contract_123",
  "verified": true,
  "page": 3,
  "verification_method": "normalized_exact_match",
  "reason": null,
  "matched_text": "written notice of termination",
  "start_offset": 316,
  "end_offset": 345,
  "similarity": 1.0
}
```

Omit `page` to search the whole document — the response reports where the quote
was found.

---

## 7. Computing dates yourself

Anything date-shaped the backend needs (renewal dates, alternative notice
windows) should go through the same deterministic code rather than being
reimplemented:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/deadlines/compute \
  -H 'Content-Type: application/json' \
  -d '{"operation": "renewal_date", "anchor_date": "2026-03-31", "duration": "twelve (12) months"}'
```

`operation` is `notice_deadline` (subtract) or `renewal_date` (add).
`anchor_date` and `duration` accept both ISO forms and contract wording
(`"March 31, 2026"`, `"sixty (60) days"`). On unparseable input the response is
`status: "failed"` with `result_date: null` and an `error` string — never a
guess.

---

## 8. Errors

| Status | When | Body |
|---|---|---|
| `400` | Empty upload | `{"detail": "empty_file"}` |
| `415` | Not a `.pdf` / `.txt` / `.md` | `{"detail": "unsupported_file_type: .docx; expected …"}` |
| `422` | No text layer (scanned PDF), or an unknown `obligation_type` | `{"detail": "no_extractable_text: …"}` |
| `404` | Unknown `document_id` | `{"detail": "unknown_document: contract_x"}` |
| `200` | Model unreachable | `can_approve: false`, `failures: [{"reason": "llm_unavailable", …}]` |

A dead model server is **not** an HTTP error: you get a well-formed
`ExtractionResult` with no obligations and an `llm_unavailable` failure, so the
queue UI can show the contract as "processing failed" rather than crashing.

---

## 9. Running it (for Saravanan)

The service is a plain ASGI app with no privileged requirements:

```bash
uvicorn obligation_rag.api:app --host 127.0.0.1 --port 8001
```

- **Outbound network:** one destination only, `LLM_BASE_URL`. Nothing else is
  contacted at runtime — no telemetry, no model downloads. Compatible with a
  default-deny egress policy that allows the gateway only.
- **Filesystem:** everything (uploads, SQLite, vector indexes) is written under
  `RAG_DATA_DIR`. Point it at the mounted `/srv/ledger` volume; no other write
  path is used.
- **State:** SQLite file plus `.npy` index files. Deleting `RAG_DATA_DIR` is a
  full reset.
- **Docker:** see the README; the image is arm64-compatible, runs as UID 10001
  and declares `/srv/ledger/rag` as its volume.

### Exposing it to OpenClaw

Each endpoint maps cleanly to a tool. Suggested definitions:

| Tool | Endpoint | Notes |
|---|---|---|
| `contract_search(document_id, query, top_k)` | `POST /v1/documents/{id}/search` | Read-only, safe to expose broadly |
| `contract_extract(document_id, obligation_types)` | `POST /v1/documents/{id}/extract` | Calls the model; rate-limit it |
| `verify_quote(document_id, quote, page)` | `POST /v1/evidence/verify` | Pure function, no model |
| `compute_deadline(operation, anchor_date, duration)` | `POST /v1/deadlines/compute` | Pure function, no model |

The agent should never be given a "commit to ledger" tool that reads from here
directly — approval is a human action by design.

---

## 10. Local development without the model server

```bash
USE_FAKE_LLM=true uvicorn obligation_rag.api:app --port 8001
```

The fake adapter is rule-based over the *retrieved text*, so the quotes it
returns are genuine spans of the document and the verification path is
exercised for real. Aditya can build the entire approval UI against it before
`gpt-oss-120b` is up, and the JSON shape is identical.

---

## 11. In-process integration (`retriever.py`)

[src/obligation_rag/retriever.py](../src/obligation_rag/retriever.py) implements
the module contract the pipeline expects — same functions, same conventions
(int `contract_id`, called after a successful parse, never raises):

```python
def index(contract_id: int, doc: ParsedDoc) -> int
def retrieve(query: str, k: int = 8, contract_id: int | None = None) -> list[Passage]
def extract(contract_id: int, doc: ParsedDoc) -> ExtractionResult   # <- added
```

`ParsedDoc` and `Passage` are imported from your `ingest.py` when present, so
your definitions stay authoritative. Copy the module in, or install this
package and re-export:

```python
# retriever.py in your repo
from obligation_rag.retriever import index, retrieve, extract  # noqa: F401
```

### Coordinate system

Every offset returned — `Passage.char_start/char_end` and
`Evidence.char_start/char_end` — indexes **`ParsedDoc.text`**, the same
normalised string your UI displays:

```python
doc.text[passage.char_start : passage.char_end] == passage.text          # always true
doc.text[ev.char_start : ev.char_end] == ev.quote                        # always true
```

The document is never re-parsed or re-normalised on this side: page text is
sliced out of `doc.text` using `Page.char_start`, so the indexed text, the text
quotes are verified against, and the text the reviewer reads are the same
string. If you change normalisation in `ingest.py`, nothing here needs updating.

### Why `extract()` and not just `retrieve()`

`Passage` carries `text, page, char_start, char_end, score`. It has no place to
put a verification status, so with `retrieve()` alone the trust guarantees never
reach the UI, and the Register view (spec §6.2) cannot render:

- a per-field badge `verified` / `computed` / `failed`
- the supporting quote with its page
- an **Approve button disabled when any field failed**

`extract()` returns exactly that:

```python
@dataclass
class Evidence:
    quote: str; page: int
    char_start: int | None; char_end: int | None   # into ParsedDoc.text

@dataclass
class Obligation:
    field: str                  # 'contract_end_date', 'notice_deadline', ...
    value: str | None           # normalised: '2026-03-31', 'P60D', 'USD 120000.00'
    status: str                 # 'verified' | 'computed' | 'failed'
    evidence: Evidence | None   # None for computed fields
    reason: str | None          # why it failed, for the reviewer
    formula: str | None         # computed fields only
    inputs: dict | None         # what the formula was evaluated on

@dataclass
class ExtractionResult:
    obligations: list[Obligation]
    can_approve: bool           # False if any field failed - gate the button on this
    failures: list[str]
```

These live in `retriever.py` today; move them to `ingest.py` if you prefer to
keep all shared types on your side — the field names are what matter.

### Usage

```python
import retriever

n_chunks = retriever.index(contract.id, doc)      # after parse, before extract
result   = retriever.extract(contract.id, doc)    # the Register view

for obligation in result.obligations:
    if obligation.evidence:
        span = doc.text[obligation.evidence.char_start : obligation.evidence.char_end]
        render_row(obligation.field, obligation.value, obligation.status, span)
    else:
        render_computed_row(obligation.field, obligation.value, obligation.formula)

approve_button.enabled = result.can_approve

# Ask tab / "where does this come from"
passages = retriever.retrieve("what renews before November?", k=8)  # whole corpus
```

### Guarantees and limits

- `index()` is idempotent per `contract_id`: re-indexing replaces the previous
  pages, chunks and vectors instead of layering on top.
- `retrieve()` and `extract()` never raise. `retrieve()` returns `[]`;
  `extract()` returns `can_approve=False` with a `failures` entry.
- `retrieve(contract_id=None)` searches the whole corpus, with BM25 statistics
  computed corpus-wide so scores are comparable across contracts. Only
  documents ingested through this module are included.
- `extract()` will index the document itself if `index()` was skipped.
- Everything is stored under `RAG_DATA_DIR`, independent of your own database.
