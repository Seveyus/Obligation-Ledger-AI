# Obligation Ledger — RAG service

A local, air-gapped RAG service that reads contract PDFs and proposes dated
obligations **where every value carries a quote that has been verified against
the source page in code, and every derived date is computed in Python.**

This repository is the **RAG component only**. It is a standalone FastAPI
service on port `8001` that the main backend calls over HTTP.

```
Aditya's backend ──HTTP──▶ this service
                            ├── parse PDF (page numbers preserved)
                            ├── chunk + index (BM25 + local vectors)
                            ├── retrieve the relevant clauses
                            ├── ask gpt-oss-120b for values + quotes
                            ├── verify every quote in Python
                            └── compute deadlines in Python
```

## Scope

| Owner | Responsibility |
|---|---|
| **Yoann (this repo)** | PDF ingestion, retrieval, structured obligation extraction, evidence verification, deterministic deadline computation |
| Aditya | Frontend and main application backend |
| Sai | Contract dataset, video presentation |
| Saravanan | OpenClaw, NVIDIA NemoClaw, OpenShell, local model runtime |

**Not in this repository:** frontend, approval UI, main backend, the committed
ledger and its hash chain, OpenClaw / NemoClaw / OpenShell, the `gpt-oss-120b`
model server, and the video. The model server is external — this service only
knows a base URL.

## The three trust rules, and where they live in the code

The product's bet is that engineering *around* the model is what makes the
output trustworthy. All three rules are enforced in code, not in UI copy:

| Rule | Implementation |
|---|---|
| **Every value quotes the contract** | [verification.py](src/obligation_rag/verification.py) — normalized exact match against the claimed page, then a bounded fuzzy fallback. The model is *never* asked to verify its own quote. |
| **Code does the math, not the model** | [date_math.py](src/obligation_rag/date_math.py) — a model-proposed `notice_deadline` is discarded on principle and recomputed from verified inputs, with the formula stored alongside the result. |
| **Nothing commits without a person** | This service only ever returns `status: proposed / verified / computed / failed`. It sets `can_approve: false` whenever any field fails. Committing is the caller's job. |

## Quickstart

```bash
git clone https://github.com/Seveyus/Obligation-Ledger-AI.git
cd Obligation-Ledger-AI

uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

cp .env.example .env          # then set USE_FAKE_LLM=true to run without a model server
```

Start the API:

```bash
.venv/bin/uvicorn obligation_rag.api:app --host 127.0.0.1 --port 8001
```

Interactive docs: <http://127.0.0.1:8001/docs>

Run the whole pipeline from the terminal, no server needed:

```bash
.venv/bin/python scripts/test_pipeline.py              # fixture + deterministic fake model
.venv/bin/python scripts/test_pipeline.py --ablation   # hallucinating model: approval blocked
.venv/bin/python scripts/test_pipeline.py contract.pdf --real   # real gpt-oss-120b
```

```
FIELD                        VALUE                  STATUS    PAGE  EVIDENCE
----------------------------------------------------------------------------
contract_end_date            2026-03-31             verified  p.2   "The initial term of this Agreement expires on March 31, 2026…"
termination_notice_period    P60D                   verified  p.3   "Either party may elect not to renew this Agreement by delive…"
notice_deadline              2026-01-30             computed  —     notice_deadline = contract_end_date - termination_notice_period (calculated in code)

can_approve = True  ->  APPROVAL UNLOCKED
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + which LLM/retrieval mode is active |
| `POST` | `/v1/documents/ingest` | Upload a PDF: parse, chunk, index |
| `POST` | `/v1/documents/{id}/search` | Hybrid retrieval over one document |
| `POST` | `/v1/documents/{id}/extract` | Full extraction → verified obligations |
| `POST` | `/v1/evidence/verify` | Re-verify a quote (used when a human edits a value) |
| `POST` | `/v1/deadlines/compute` | Deterministic date arithmetic |
| `GET` | `/v1/documents` | List ingested documents |
| `GET` | `/v1/documents/{id}` | Document metadata |
| `GET` | `/v1/documents/{id}/pages` | Page text — the exact string evidence offsets refer to |
| `GET` | `/v1/documents/{id}/obligations` | Last extraction, without re-running the model |
| `DELETE` | `/v1/documents/{id}` | Drop a document and its index |

### Ingestion

```bash
curl -s -X POST http://127.0.0.1:8001/v1/documents/ingest \
  -F "file=@data/samples/sample_contract.txt" \
  -F "document_id=contract_123"
```

```json
{
  "document_id": "contract_123",
  "filename": "sample_contract.txt",
  "page_count": 6,
  "chunk_count": 6
}
```

`document_id` is optional; without it the service derives a content-addressed
id (`contract_<sha256[:12]>`), which makes re-uploading the same file
idempotent.

### Retrieval

```bash
curl -s -X POST http://127.0.0.1:8001/v1/documents/contract_123/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "termination notice period", "top_k": 3}'
```

```json
{
  "document_id": "contract_123",
  "query": "termination notice period",
  "retrieval_mode": "bm25_only",
  "chunks": [
    {
      "id": "chunk_2",
      "document_id": "contract_123",
      "page": 3,
      "text": "3. TERMINATION\n\n3.1 Termination for Cause. …",
      "lexical_score": 1.0,
      "vector_score": 0.0,
      "fused_score": 0.016393
    }
  ]
}
```

### Extraction

```bash
curl -s -X POST http://127.0.0.1:8001/v1/documents/contract_123/extract \
  -H 'Content-Type: application/json' \
  -d '{"obligation_types": ["term_end", "termination_notice", "automatic_renewal"]}'
```

```json
{
  "document_id": "contract_123",
  "obligations": [
    {
      "id": "contract_123:termination_notice_period",
      "document_id": "contract_123",
      "obligation_type": "termination_notice_period",
      "raw_value": "P60D",
      "normalized_value": "P60D",
      "source_evidence": {
        "quote": "Either party may elect not to renew this Agreement by delivering\nwritten notice of termination to the other party not less than sixty (60)\ndays prior to the end of the then-current term.",
        "page": 3,
        "chunk_id": "chunk_2",
        "start_offset": 251,
        "end_offset": 437
      },
      "status": "verified",
      "verification_method": "normalized_exact_match",
      "verification_reason": null,
      "computation_formula": null,
      "computation_inputs": null
    }
  ],
  "can_approve": true,
  "failures": [],
  "llm_mode": "fake",
  "retrieval_mode": "bm25_only",
  "elapsed_seconds": 0.017
}
```

Omit `obligation_types` to extract all eleven. Full field-by-field contract:
[docs/INTEGRATION.md](docs/INTEGRATION.md).

### Evidence verification and deadlines

```bash
curl -s -X POST http://127.0.0.1:8001/v1/evidence/verify \
  -H 'Content-Type: application/json' \
  -d '{"document_id": "contract_123", "quote": "written notice of termination", "page": 3}'

curl -s -X POST http://127.0.0.1:8001/v1/deadlines/compute \
  -H 'Content-Type: application/json' \
  -d '{"operation": "notice_deadline", "anchor_date": "March 31, 2026", "duration": "sixty (60) days"}'
```

```json
{
  "operation": "notice_deadline",
  "result_date": "2026-01-30",
  "status": "computed",
  "computation_formula": "notice_deadline = contract_end_date - termination_notice_period",
  "computation_inputs": {
    "contract_end_date": "March 31, 2026",
    "termination_notice_period": "sixty (60) days",
    "contract_end_date_iso": "2026-03-31",
    "termination_notice_period_iso": "P60D",
    "operation": "subtract",
    "evaluated": "2026-03-31 - 60 days = 2026-01-30"
  },
  "error": null
}
```

## Configuration

Everything is environment-driven ([.env.example](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible endpoint for the local model |
| `LLM_API_KEY` | `local` | Ignored by most local servers, required by the client |
| `LLM_MODEL` | `gpt-oss-120b` | Model name passed through to the server |
| `USE_FAKE_LLM` | `false` | Run the pipeline with the deterministic rule-based adapter |
| `EMBEDDING_MODEL_PATH` | *(empty)* | Local sentence-transformers directory; empty ⇒ BM25-only |
| `USE_FAKE_EMBEDDINGS` | `false` | Deterministic hashed vectors, for testing hybrid retrieval without weights |
| `RAG_DATA_DIR` | `./runtime-data` | Uploads, SQLite database, vector indexes |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8001` | Bind address |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | `1200` / `200` | Chunking |
| `DEFAULT_TOP_K` / `RRF_K` | `6` / `60` | Retrieval |
| `FUZZY_MATCH_THRESHOLD` / `FUZZY_MAX_LENGTH_RATIO` | `0.92` / `1.35` | Verification strictness |

No weights are ever downloaded, and no request ever leaves the machine: the
only outbound connection is to `LLM_BASE_URL`.

## How it works

```
src/obligation_rag/
├── config.py        environment-driven settings
├── schemas.py       the Pydantic contract shared with the backend
├── pdf_parser.py    PyMuPDF text extraction, page numbers preserved
├── chunking.py      page-aware chunks with offsets and overlap
├── embeddings.py    sentence-transformers | hashed | none, behind one interface
├── retrieval.py     BM25 + cosine, fused with Reciprocal Rank Fusion
├── llm_client.py    real / fake / hallucinating adapters for gpt-oss-120b
├── ingestion.py     parse → chunk → persist → index (shared by API and CLI)
├── extraction.py    retrieve → propose → verify → normalize → compute
├── verification.py  deterministic source-evidence checking
├── date_math.py     deterministic date arithmetic
├── storage.py       SQLite + .npy vector files
└── api.py           FastAPI surface
```

**Retrieval.** BM25 (lexical) and local embeddings (semantic) are fused with
Reciprocal Rank Fusion rather than a weighted score blend — BM25 scores and
cosine similarities live on different, document-dependent scales, so any fixed
weighting is a guess, while ranks are directly comparable. Contract clauses are
lexically distinctive ("sixty (60) days' written notice"), so **the service
works in BM25-only mode** when no embedding model is present; the dense side is
there for paraphrases ("cancel the agreement" ↔ "terminate this Agreement").

**Verification.** The claimed quote and the claimed page are normalized
(Unicode punctuation, ligatures, NBSP, soft hyphens, whitespace runs), then:
normalized exact substring match first; a *bounded* fuzzy fallback second, for
PDF extraction artifacts only, anchored on the longest common block and capped
by both a similarity threshold and a length ratio. A quote that is real but on
a different page **fails**, with a reason naming the page it was actually found
on. Offsets are mapped back onto the original page text, so the approval UI can
highlight the exact span.

**Date math.** The model reads a term-end date and a notice duration off the
page; Python computes the deadline. Month arithmetic clamps to the end of short
months (31 Jan + 1 month = 28 Feb). Every computed value stores its formula and
inputs, including the ids of the verified obligations it was derived from.

## Tests

```bash
.venv/bin/python -m pytest                              # 128 tests
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format src tests scripts
```

Integration tests run the whole pipeline against the fake model, so the suite
needs no model server, no network and no weights. They include the spec's
**ablation test** (§10.1): with a hallucinating model, verification catches the
invented date and `can_approve` goes to `false`.

The fixture is a synthetic six-page contract
([data/samples/sample_contract.txt](data/samples/sample_contract.txt)) — no real
contracts are committed, and `.gitignore` excludes PDFs, databases, indexes,
weights and `.env`.

## Docker

```bash
docker build -t obligation-rag .
docker run --rm -p 8001:8001 \
  -v "$PWD/runtime-data:/srv/ledger/rag" \
  -e LLM_BASE_URL=http://host.docker.internal:8000/v1 \
  obligation-rag
```

The image is multi-arch (arm64 for the GB10 target box, plus x86_64), runs as a
non-root user, and keeps all state under the single mounted volume.

## Assumptions and open questions

Resolved here, worth confirming with the team:

- **Verification algorithm** (spec §7, `[OPEN]`) — resolved as normalized exact
  match with a bounded fuzzy fallback, never a model self-check. Thresholds are
  configurable.
- **`human_verified` status** (spec §7, `[OPEN]`) — present in the schema; this
  service never sets it, the approval UI does after a human edit. Re-verify
  edited quotes through `POST /v1/evidence/verify`.
- **Binary vs. confidence score** (spec §7, `[OPEN]`) — binary. A similarity
  score is returned for transparency but never turns a failure into a pass.
- **OCR / scanned contracts** (spec §8, `[OPEN]`) — out of scope. A PDF with no
  text layer is rejected with `422 no_extractable_text` rather than silently
  indexed as empty.
- **Obligation type names** — canonical `snake_case`; the deck's shorter labels
  (`term_end`, `termination_notice`, …) are accepted as aliases.
- **Ingest also accepts `.txt`** so fixtures and pipelines can be exercised
  without binary files in git.
- **Ambiguous `dd/mm` vs `mm/dd` dates** — read as US format, falling back to
  day-first when the US reading is impossible. Worth revisiting with real data.

The full product spec this component was built against lives in
[docs/obligation-ledger-spec.md](docs/obligation-ledger-spec.md).
