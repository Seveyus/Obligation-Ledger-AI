"""FastAPI surface of the RAG component.

This is the only thing the rest of the team integrates against: Aditya's
backend calls it over HTTP, and Saravanan can expose the same endpoints to
OpenClaw as tools. No UI, no ledger, no approval logic lives here — this
service proposes and proves, a human disposes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, storage
from .config import Settings, get_settings
from .date_math import (
    compute_notice_deadline,
    compute_renewal_date,
    compute_renewal_option_deadline,
)
from .embeddings import get_embedding_backend
from .extraction import DEFAULT_OBLIGATION_TYPES, run_extraction
from .ingestion import UnsupportedDocumentError, ingest_bytes
from .llm_client import get_llm_client
from .pdf_parser import DocumentParseError
from .retrieval import get_document_index, invalidate_index
from .schemas import (
    APPROVABLE_STATUSES,
    DeadlineComputeRequest,
    DeadlineComputeResponse,
    DeadlineOperation,
    DocumentPagesResponse,
    DocumentRecord,
    ExtractionResult,
    ExtractRequest,
    HealthResponse,
    IngestResponse,
    ObligationStatus,
    PageText,
    SearchRequest,
    SearchResponse,
    VerificationMethod,
    VerifyRequest,
    VerifyResponse,
    coerce_obligation_type,
)
from .verification import verify_evidence

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".text"}

SettingsDep = Annotated[Settings, Depends(get_settings)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    storage.init_db(settings)
    logger.info(
        "obligation-rag ready | data=%s | llm=%s | model=%s",
        settings.rag_data_dir,
        "fake" if settings.use_fake_llm else settings.llm_base_url,
        settings.llm_model,
    )
    yield


app = FastAPI(
    title="Obligation Ledger — RAG service",
    version=__version__,
    summary=(
        "Local contract RAG: PDF ingestion, hybrid retrieval, structured obligation "
        "extraction, deterministic evidence verification and deadline computation."
    ),
    lifespan=lifespan,
)

# The approval UI is served from another origin on the same LAN box.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(settings: SettingsDep) -> HealthResponse:
    backend = get_embedding_backend(settings)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_mode="fake" if settings.use_fake_llm else "openai_compatible",
        llm_model="fake" if settings.use_fake_llm else settings.llm_model,
        embedding_backend=backend.name if backend else "none",
        retrieval_mode="hybrid_bm25_vector_rrf" if backend else "bm25_only",
        document_count=storage.count_documents(settings),
    )


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


@app.post("/v1/documents/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest_document(
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Contract PDF (or .txt fixture)")],
    document_id: Annotated[str | None, Form(description="Optional caller-supplied id")] = None,
) -> IngestResponse:
    payload = await file.read()

    try:
        outcome = ingest_bytes(
            settings, payload, file.filename or "contract.pdf", document_id=document_id
        )
    except UnsupportedDocumentError as error:
        status = 400 if "empty_file" in str(error) else 415
        raise HTTPException(status_code=status, detail=str(error)) from error
    except DocumentParseError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return IngestResponse(
        document_id=outcome.document_id,
        filename=outcome.filename,
        page_count=outcome.page_count,
        chunk_count=outcome.chunk_count,
    )


@app.get("/v1/documents", response_model=list[DocumentRecord], tags=["documents"])
def list_documents(settings: SettingsDep) -> list[DocumentRecord]:
    return storage.list_documents(settings)


@app.get("/v1/documents/{document_id}", response_model=DocumentRecord, tags=["documents"])
def get_document(settings: SettingsDep, document_id: str) -> DocumentRecord:
    record = storage.get_document(settings, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown_document: {document_id}")
    return record


@app.get(
    "/v1/documents/{document_id}/pages", response_model=DocumentPagesResponse, tags=["documents"]
)
def get_document_pages(settings: SettingsDep, document_id: str) -> DocumentPagesResponse:
    """Page text as stored — the exact string evidence offsets refer to."""
    pages = storage.get_pages(settings, document_id)
    if not pages:
        raise HTTPException(status_code=404, detail=f"unknown_document: {document_id}")
    return DocumentPagesResponse(
        document_id=document_id,
        page_count=len(pages),
        pages=[PageText(page=page, text=text) for page, text in sorted(pages.items())],
    )


@app.delete("/v1/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(settings: SettingsDep, document_id: str) -> None:
    if storage.get_document(settings, document_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown_document: {document_id}")
    storage.delete_document(settings, document_id)
    invalidate_index(document_id)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@app.post("/v1/documents/{document_id}/search", response_model=SearchResponse, tags=["retrieval"])
def search_document(
    settings: SettingsDep, document_id: str, request: SearchRequest
) -> SearchResponse:
    index = get_document_index(settings, document_id)
    if index is None:
        raise HTTPException(status_code=404, detail=f"unknown_document: {document_id}")
    return SearchResponse(
        document_id=document_id,
        query=request.query,
        retrieval_mode=index.retrieval_mode,
        chunks=index.search(request.query, top_k=request.top_k),
    )


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


@app.post(
    "/v1/documents/{document_id}/extract", response_model=ExtractionResult, tags=["extraction"]
)
def extract_obligations(
    settings: SettingsDep, document_id: str, request: ExtractRequest | None = None
) -> ExtractionResult:
    request = request or ExtractRequest()
    index = get_document_index(settings, document_id)
    pages = storage.get_pages(settings, document_id)
    if index is None or not pages:
        raise HTTPException(status_code=404, detail=f"unknown_document: {document_id}")

    if request.obligation_types:
        try:
            obligation_types = [coerce_obligation_type(value) for value in request.obligation_types]
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail=f"unknown_obligation_type: {error}"
            ) from error
    else:
        obligation_types = DEFAULT_OBLIGATION_TYPES

    result = run_extraction(
        settings,
        document_id,
        index,
        pages,
        obligation_types=obligation_types,
        top_k=request.top_k,
        client=get_llm_client(settings),
    )
    storage.save_obligations(settings, document_id, result.obligations)
    return result


@app.get(
    "/v1/documents/{document_id}/obligations",
    response_model=ExtractionResult,
    tags=["extraction"],
)
def get_last_extraction(settings: SettingsDep, document_id: str) -> ExtractionResult:
    """The most recent extraction, without re-running the model."""
    if storage.get_document(settings, document_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown_document: {document_id}")
    obligations = storage.get_obligations(settings, document_id)
    return ExtractionResult(
        document_id=document_id,
        obligations=obligations,
        can_approve=bool(obligations)
        and all(obligation.status in APPROVABLE_STATUSES for obligation in obligations),
        llm_mode="cached",
        retrieval_mode="cached",
    )


# --------------------------------------------------------------------------
# Evidence and deadlines (deterministic, no model involved)
# --------------------------------------------------------------------------


@app.post("/v1/evidence/verify", response_model=VerifyResponse, tags=["verification"])
def verify_evidence_endpoint(settings: SettingsDep, request: VerifyRequest) -> VerifyResponse:
    """Re-verify a quote. Used by the approval UI when a human edits a value."""
    pages = storage.get_pages(settings, request.document_id)
    if not pages:
        raise HTTPException(status_code=404, detail=f"unknown_document: {request.document_id}")

    outcome = verify_evidence(
        request.quote,
        pages,
        claimed_page=request.page,
        fuzzy_threshold=settings.fuzzy_match_threshold,
        fuzzy_max_length_ratio=settings.fuzzy_max_length_ratio,
        min_quote_chars=settings.min_quote_chars,
        reject_unchecked_options=settings.reject_unchecked_options,
    )
    return VerifyResponse(
        document_id=request.document_id,
        verified=outcome.verified,
        page=outcome.page,
        verification_method=outcome.method or VerificationMethod.NONE,
        reason=outcome.reason,
        matched_text=outcome.matched_text,
        start_offset=outcome.start_offset,
        end_offset=outcome.end_offset,
        similarity=outcome.similarity,
    )


@app.post("/v1/deadlines/compute", response_model=DeadlineComputeResponse, tags=["verification"])
def compute_deadline(request: DeadlineComputeRequest) -> DeadlineComputeResponse:
    """Date arithmetic in code. The model never gets a vote here."""
    if request.operation is DeadlineOperation.RENEWAL_DATE:
        computation = compute_renewal_date(request.anchor_date, request.duration)
    elif request.operation is DeadlineOperation.RENEWAL_OPTION_DEADLINE:
        computation = compute_renewal_option_deadline(request.anchor_date, request.duration)
    else:
        computation = compute_notice_deadline(request.anchor_date, request.duration)

    return DeadlineComputeResponse(
        operation=request.operation,
        result_date=computation.result_iso,
        status=ObligationStatus.COMPUTED if computation.ok else ObligationStatus.FAILED,
        computation_formula=computation.formula,
        computation_inputs=computation.inputs,
        error=computation.error,
    )


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "obligation_rag.api:app",
        host=settings.rag_host,
        port=settings.rag_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
