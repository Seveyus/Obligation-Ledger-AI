"""In-process adapter matching the `retriever.py` contract of the main backend.

The pipeline imports three functions from here and nothing else:

    index(contract_id, doc)   -> int                 # chunk count
    retrieve(query, k, cid)   -> list[Passage]       # never raises
    extract(contract_id, doc) -> ExtractionResult    # never raises

`retrieve` covers "show me where this came from" and the Ask tab. `extract` is
the one that carries the product's trust guarantees across the boundary:
per-field status, the verified quote with its offsets, the deadline computed in
code, and `can_approve`. A `Passage` cannot express any of that.

Coordinate system: this module speaks the caller's coordinates. Every offset it
returns — in `Passage` and in `Evidence` — indexes `ParsedDoc.text`, the same
normalised string the approval UI displays, so a span can be highlighted
directly. The document is never re-parsed or re-normalised here; the caller's
text is the single source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import storage
from .chunking import chunk_document
from .config import get_settings
from .embeddings import get_embedding_backend
from .extraction import run_extraction
from .llm_client import get_llm_client
from .pdf_parser import ParsedDocument, ParsedPage
from .retrieval import (
    build_index,
    cache_index,
    get_corpus_index,
    get_document_index,
    invalidate_index,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared types.
#
# The backend owns these definitions in its own `ingest.py`; when this module
# is dropped into that repository the import below wins and the local mirrors
# are never used. They exist so this package stays importable and testable on
# its own. The fields are identical either way.
# --------------------------------------------------------------------------

try:  # pragma: no cover - depends on which repository this runs in
    from ingest import ParsedDoc, Passage  # type: ignore[import-not-found]
except ImportError:

    @dataclass
    class Page:
        number: int
        text: str
        char_start: int

    @dataclass
    class ParsedDoc:
        text: str
        pages: list[Page]
        fmt: str
        converted_via: str | None = None

    @dataclass
    class Passage:
        text: str
        contract_id: int
        page: int
        char_start: int
        char_end: int
        score: float


@dataclass
class Evidence:
    """The span of `ParsedDoc.text` that backs a value."""

    quote: str
    page: int
    char_start: int | None
    char_end: int | None


@dataclass
class Obligation:
    field: str  # 'contract_end_date', 'termination_notice_period', ...
    value: str | None  # normalised: '2026-03-31', 'P60D', 'USD 120000.00'
    status: str  # 'verified' | 'computed' | 'failed'
    evidence: Evidence | None  # None for computed fields
    reason: str | None = None  # why it failed, for the reviewer
    formula: str | None = None  # computed fields only
    inputs: dict | None = None  # what the formula was evaluated on


@dataclass
class ExtractionResult:
    obligations: list[Obligation] = field(default_factory=list)
    #: False if any field failed verification. The Approve button follows this.
    can_approve: bool = False
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _document_id(contract_id: int) -> str:
    return str(int(contract_id))


def _contract_id(document_id: str) -> int | None:
    """Reverse of `_document_id`; None for documents ingested via the HTTP API."""
    return int(document_id) if document_id.isdigit() else None


def _page_spans(doc: ParsedDoc) -> list[tuple[int, int, int]]:
    """`(page_number, char_start, char_end)` spans into `doc.text`.

    Page text is taken by slicing `doc.text` rather than reading `Page.text`,
    so the indexed text and the offsets can never disagree with the string the
    reviewer is looking at.
    """
    pages = list(doc.pages)
    if not pages:
        return [(1, 0, len(doc.text))]

    spans: list[tuple[int, int, int]] = []
    for position, page in enumerate(pages):
        start = int(page.char_start)
        end = int(pages[position + 1].char_start) if position + 1 < len(pages) else len(doc.text)
        spans.append((int(page.number), start, max(start, end)))
    return spans


def _as_parsed_document(contract_id: int, doc: ParsedDoc) -> tuple[ParsedDocument, dict[int, int]]:
    spans = _page_spans(doc)
    parsed = ParsedDocument(
        filename=f"contract_{contract_id}",
        pages=[ParsedPage(page=number, text=doc.text[start:end]) for number, start, end in spans],
    )
    return parsed, {number: start for number, start, _ in spans}


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def index(contract_id: int, doc: ParsedDoc) -> int:
    """Index one parsed document. Returns chunk count.

    Called after a successful parse, before extraction. Idempotent per
    `contract_id`: re-indexing replaces the previous pages, chunks and vectors
    rather than layering on top of them.
    """
    settings = get_settings()
    settings.ensure_directories()
    storage.init_db(settings)

    document_id = _document_id(contract_id)
    parsed, page_offsets = _as_parsed_document(contract_id, doc)

    chunks = chunk_document(
        document_id,
        parsed,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )
    for chunk in chunks:
        base = page_offsets.get(chunk.page, 0)
        chunk.doc_start_offset = base + chunk.start_offset
        chunk.doc_end_offset = base + chunk.end_offset

    # Replacing the document row cascades the previous pages/chunks away.
    storage.insert_document(
        settings,
        document_id=document_id,
        filename=f"contract_{contract_id}.{doc.fmt}",
        stored_path=settings.uploads_dir / f"{document_id}.{doc.fmt}",
        page_count=parsed.page_count,
        chunk_count=len(chunks),
    )
    storage.insert_pages(settings, document_id, parsed.page_map())
    storage.insert_chunks(settings, chunks)

    backend = get_embedding_backend(settings)
    embeddings = None
    if backend is not None and chunks:
        try:
            embeddings = backend.encode_documents([chunk.text for chunk in chunks])
            storage.save_embeddings(settings, document_id, embeddings)
        except Exception as error:  # noqa: BLE001 - the dense path is optional
            logger.warning("embedding failed (%s); contract %s stays BM25-only", error, contract_id)
            embeddings = None

    invalidate_index(document_id)
    cache_index(
        build_index(
            document_id,
            chunks,
            settings=settings,
            embeddings=embeddings,
            backend=backend if embeddings is not None else None,
        )
    )
    return len(chunks)


def retrieve(query: str, k: int = 8, contract_id: int | None = None) -> list[Passage]:
    """Return top-k passages. `contract_id=None` searches the whole corpus.

    Never raises: retrieval failing is a degraded answer, not a broken request.
    """
    try:
        settings = get_settings()
        if contract_id is None:
            document_index = get_corpus_index(settings)
        else:
            document_index = get_document_index(settings, _document_id(contract_id))
        if document_index is None:
            return []

        by_key = {(chunk.document_id, chunk.chunk_id): chunk for chunk in document_index.chunks}

        passages: list[Passage] = []
        for hit in document_index.search(query, top_k=k):
            resolved = _contract_id(hit.document_id)
            if resolved is None:
                continue  # ingested through the HTTP API, not part of this corpus
            chunk = by_key.get((hit.document_id, hit.id))
            start = chunk.doc_start_offset if chunk and chunk.doc_start_offset is not None else 0
            end = (
                chunk.doc_end_offset
                if chunk and chunk.doc_end_offset is not None
                else len(hit.text)
            )
            passages.append(
                Passage(
                    text=hit.text,
                    contract_id=resolved,
                    page=hit.page,
                    char_start=start,
                    char_end=end,
                    score=hit.fused_score,
                )
            )
        return passages
    except Exception:  # noqa: BLE001 - contract says: never raise
        logger.exception("retrieve failed for query=%r contract_id=%s", query, contract_id)
        return []


def extract(contract_id: int, doc: ParsedDoc) -> ExtractionResult:
    """Extract obligations with verified evidence and computed deadlines.

    Called after `index()`. Every returned value has either a quote that was
    checked against `doc.text` in Python, or a formula that was evaluated in
    Python — never the model's word for it.

    Never raises: on failure returns an empty result with `can_approve=False`,
    which the UI should surface as "processing failed", not as "nothing to do".
    """
    try:
        settings = get_settings()
        settings.ensure_directories()
        storage.init_db(settings)
        document_id = _document_id(contract_id)

        document_index = get_document_index(settings, document_id)
        if document_index is None:  # defensive: pipeline should have indexed already
            index(contract_id, doc)
            document_index = get_document_index(settings, document_id)
        if document_index is None:
            return ExtractionResult(can_approve=False, failures=["index_unavailable"])

        spans = _page_spans(doc)
        pages = {number: doc.text[start:end] for number, start, end in spans}
        page_offsets = {number: start for number, start, _ in spans}

        result = run_extraction(
            settings,
            document_id,
            document_index,
            pages,
            client=get_llm_client(settings),
        )
        storage.save_obligations(settings, document_id, result.obligations)

        obligations: list[Obligation] = []
        for candidate in result.obligations:
            evidence = None
            if candidate.source_evidence is not None:
                source = candidate.source_evidence
                base = page_offsets.get(source.page, 0)
                evidence = Evidence(
                    quote=source.quote,
                    page=source.page,
                    char_start=None if source.start_offset is None else base + source.start_offset,
                    char_end=None if source.end_offset is None else base + source.end_offset,
                )
            obligations.append(
                Obligation(
                    field=candidate.obligation_type.value,
                    value=candidate.normalized_value,
                    status=candidate.status.value,
                    evidence=evidence,
                    reason=candidate.verification_reason,
                    formula=candidate.computation_formula,
                    inputs=candidate.computation_inputs,
                )
            )

        return ExtractionResult(
            obligations=obligations,
            can_approve=result.can_approve,
            failures=[
                f"{failure.reason}: {failure.detail or ''}".strip() for failure in result.failures
            ],
        )
    except Exception:  # noqa: BLE001 - same rule as retrieve
        logger.exception("extract failed for contract_id=%s", contract_id)
        return ExtractionResult(can_approve=False, failures=["extraction_crashed"])
