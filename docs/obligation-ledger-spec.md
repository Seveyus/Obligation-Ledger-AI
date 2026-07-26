# Obligation Ledger — Product & Technical Spec

> Source: derived from the project's pitch-deck mockup (`Obligation Ledger` HTML deck). Items marked **[ASSUMED]** are inferred from the visual design and are not yet confirmed — flag these for discussion before building against them. Items marked **[OPEN]** are unresolved decisions the team still needs to make.

---

## 1. One-line pitch

A local, air-gapped assistant that reads contracts (PDFs), extracts dated financial/legal obligations, verifies every extracted value against a direct quote from the source document, and turns them into a trustworthy, human-approved register of deadlines and terms.

## 2. Problem statement

Contracts bury obligations — renewal windows, notice periods, fee escalators, uncapped indemnities — inside dense legal text. These facts exist only in the PDF: they aren't in anyone's calendar or accounting system until a human manually reads and transcribes them. Missed notice deadlines and unnoticed clauses translate directly into cost (auto-renewals, fee creep, unmanaged liability).

## 3. Core value proposition / thesis

An LLM alone is only **~⅔ accurate** at this kind of extraction (per deck) — good enough to draft, not good enough to trust blindly. The product's bet is that **engineering around the model** (not a better model) is what makes the output trustworthy:

- Every extracted value must carry a **direct quote** from the source document as evidence.
- Any **date arithmetic** (e.g., deadline = term_end − notice_period) is computed in **deterministic code**, never left to the model.
- Nothing is committed to the system of record without **explicit human approval** — the model only *proposes*.

This reframes the product from "AI that reads contracts" to "a verification pipeline that happens to use AI for the first draft."

## 4. Users / use case

- **[ASSUMED]** Primary user: someone responsible for contract/vendor management (ops, legal ops, finance, or a small business owner) who needs to track renewal/notice deadlines without a legal team.
- Deployment implies a **privacy-sensitive** buyer: legal documents, willing to run local hardware rather than send contracts to a cloud API. **[ASSUMED — worth confirming: is air-gapping a hard requirement or a differentiator/selling point?]**

## 5. End-to-end pipeline

```
Drop → Read → Verify → Approve → Ledger
```

1. **Drop** — user places a contract file (PDF) into a watched folder.
2. **Read** — pipeline parses the document and the model extracts candidate obligations/fields.
3. **Verify** — each extracted field is checked against the source text; matching logic confirms the value is actually present as quoted. Any date math is recomputed in code from verified inputs, not taken from the model's output directly.
4. **Approve** — a human reviews a diff-like view of proposed values, each with its supporting quote and a verified/computed/failed status. Approval is a distinct, required action.
5. **Ledger** — once approved, values are committed to a persistent, tamper-evident register (SQLite + hash chain per architecture slide) that downstream features (deadlines feed, natural-language Q&A, calendar reminders, memos) read from.

Turnaround for a new file to appear in the queue: **~2 minutes [ASSUMED to be a target/observed latency, not yet a guarantee].**

## 6. Product surfaces (UI)

A single web app (`ledger.local`, served over LAN, e.g. `:8443` **[ASSUMED port from architecture diagram]**) with four tabs:

### 6.1 Queue
- List of contracts awaiting processing/review, each showing entity name, filename, page count, time since upload, and a status badge: **Proposed** (amber) / **Committed** (teal) / presumably **Failed/Rejected** (red).
- The most recently arrived item is visually highlighted ("hot" row).

### 6.2 Register (detail / review view)
- Per-contract review screen. Each extracted field is a row with:
  - **Key** (e.g. "Term end", "Notice", "Deadline", "Renewal term")
  - **Value** + a status badge: `verified` (teal), `computed` (amber), `failed` (red)
  - **Supporting quote** — the exact source sentence with the relevant span highlighted (`<mark>`), plus a page reference (e.g. "p.9")
  - Computed fields (e.g. Deadline = term_end − notice days) are visually distinguished and annotated "calculated in code, not model output"
- **Gating rule:** if **any** field fails verification, the **Approve** action is disabled until a human resolves it (edits, confirms manually, or rejects). This is the core trust mechanism of the product.
- Actions available: **Reject**, **Edit**, **Approve** (disabled when a field has failed).

### 6.3 Deadlines
- Rolling view (e.g. "Next 90 days") of upcoming obligations across all committed contracts, sorted by date, with urgent ones flagged (red dot/text) — e.g. "notice due in 54 days."
- Aggregated count of open items.

### 6.4 Ask
- Read-only natural-language query interface over the committed register (e.g. "what renews before November?").
- Answers are grounded in the register data (not a free-form model answer over raw contract text) — **[ASSUMED, worth confirming implementation: is this a retrieval-over-structured-register query, or RAG over raw docs? The "read-only" badge and register-first design suggests the former.]**

### 6.5 Other outputs (mentioned, not detailed in mockup)
- Calendar reminders (format/integration **[OPEN]**)
- A "one-page memo" export (format **[OPEN]** — PDF? Markdown?)
- Tamper-evident audit line, viewable/verifiable on screen

## 7. Trust rules (product principles, enforced in code — not just UI copy)

1. **Every value quotes the contract.** No quote → no commit. Verification is presumably a string/substring match (or fuzzy match) between the model's claimed value and the actual document text, not just the model's say-so.
2. **Code does the math, not the model.** Any derived date/number is computed by a deterministic function operating on verified inputs.
3. **Nothing commits without a person.** The model's output is always "Proposed" status until a human approves; approval is blocked automatically if any field fails verification.

**[OPEN — implementation questions for these rules:]**
- What's the exact verification algorithm? Exact substring match, normalized/fuzzy match, or a secondary "does this quote support this value" model check?
- What happens on Reject — is the contract requeued, discarded, or sent for manual entry?
- What happens on Edit — does an edited value get a new status (e.g. "human-verified") distinct from "verified" (model+quote) and "computed"?
- Is there a confidence score anywhere, or is it strictly binary verified/failed?

## 8. Architecture

**Target hardware [ASSUMED, per deck]:** Dell Pro Max, GB10 chip, 128 GB memory — a single local box intended to run everything without cloud dependency.

Two trust zones on one machine:

### Host (minimal)
- **vLLM** — serves the local model on GPU.
- **OpenShell gateway** — a policy engine that mediates access to the model (likely enforcing what the sandboxed agent is allowed to ask/do).
- **`/srv/ledger`** — the only storage mounted from host into the sandbox.

### Sandbox (default-deny)
- **OpenClaw agent** — the actual agent process: watches the drop folder, generates memos, answers "Ask" queries.
- **Pipeline** — parse → extract → validate stages (the "Read → Verify" steps above).
- **Approval UI** — the human-in-the-loop web app.
- **SQLite + audit hash chain** — the register itself; tamper-evident (each entry presumably hashes the previous entry, blockchain-style, to make silent edits detectable).

### Network flows
- **Inference:** agent → gateway → local model (never leaves the box)
- **People:** browser → LAN (`:8443` or similar)
- **Outbound (internet):** explicitly **denied and logged** — this is a hard requirement, not just a default.

**[OPEN — architecture questions:]**
- Is "OpenShell" / "OpenClaw" real existing tooling you're using, or working names for components you're building? Worth clarifying naming/branding vs. actual library dependencies so the other agent doesn't go looking for nonexistent packages.
- What model is being served via vLLM (size, quantization)? 128GB unified memory suggests a mid-size local model (e.g. 70B-class) is the target.
- What enforces the sandbox boundary — containers, VM, gVisor, seccomp? "Default-deny" needs a concrete mechanism (e.g. network namespace with an explicit allowlist, or literally no network interface).
- Where does the PDF parsing happen — sandbox only? Any OCR requirement for scanned contracts?

## 9. Data model (inferred fields — needs formalization)

Per contract, at minimum:
- `id`, `entity_name`, `filename`, `page_count`, `uploaded_at`, `status` (proposed / committed / rejected)
- A set of extracted **fields**, each with:
  - `key` (e.g. term_end, notice_period, renewal_term, indemnity_cap)
  - `value`
  - `status` (verified / computed / failed)
  - `source_quote` (exact text span)
  - `source_page`
  - for computed fields: the formula/inputs used
- Derived **deadlines**: `date`, `description`, `contract_id`, `urgency`

**[OPEN]** — none of this is confirmed as an actual schema; it's reconstructed from the UI mockup and should be validated/replaced with the real schema once one exists.

## 10. Proof / demo criteria ("how we prove it")

The deck proposes two concrete demos meant to validate the trust claims, not just describe them:

1. **Ablation test** — run the same model with verification checks off vs. on. Off → a wrong date slips through uncaught. On → it's caught and blocks approval. Demonstrates the value is in the *engineering*, not a "better model."
2. **Unplug test** — physically disconnect the network (WAN ✕) and confirm the model, ledger, and approval flow all continue to function identically. Demonstrates genuine air-gap, not just an unused internet connection.
3. Additionally: verify the tamper-evident **hash chain** live on screen, showing every decision recorded in order.

These read like the actual acceptance criteria / demo script for a pitch or investor/customer walkthrough — useful as literal test cases for the other agent to build toward.

## 11. Key open questions to resolve before/while building

- [ ] Confirm hardware target and whether it's a hard constraint (must run fully offline) or an initial deployment target.
- [ ] Confirm or replace placeholder tool names (OpenShell, OpenClaw) with real chosen tech.
- [ ] Define exact verification algorithm for "value must quote the contract."
- [ ] Define the audit/hash-chain scheme (what's hashed, chained how, how a human verifies it).
- [ ] Decide file formats for calendar reminders and the one-page memo export.
- [ ] Decide behavior on Reject/Edit and how those states are tracked in the audit trail.
- [ ] Confirm whether "Ask" queries the structured register only, or also has fallback access to raw contract text.
- [ ] Define supported input types beyond PDF (scanned/OCR contracts? Word docs?).
- [ ] Define what "~⅔ accurate alone" was measured against — useful as a benchmark to beat/track over time.

---

*This document was reconstructed from a visual pitch-deck mockup (HTML/CSS slideshow), not from existing code or a written spec — treat structural/technical details as a strong starting draft, and confirm against whatever your other agent already knows or has built.*
