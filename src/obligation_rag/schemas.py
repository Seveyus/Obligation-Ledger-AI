"""Pydantic contract shared with Aditya's backend.

These schemas are the public API of the RAG component. Anything the model
produces is parsed through them before it is allowed any further into the
pipeline — an unparseable model answer is a failure, never a silent pass.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ObligationType(StrEnum):
    """Obligation kinds the local model is asked to extract."""

    CONTRACT_START_DATE = "contract_start_date"
    CONTRACT_END_DATE = "contract_end_date"
    AUTOMATIC_RENEWAL = "automatic_renewal"
    RENEWAL_DURATION = "renewal_duration"
    TERMINATION_NOTICE_PERIOD = "termination_notice_period"
    NOTICE_DEADLINE = "notice_deadline"
    #: Notice a party must give to *exercise* a renewal option. The mirror image
    #: of a termination notice: miss a termination notice and you are locked in
    #: for another term; miss this one and the option "shall lapse and expire",
    #: so you lose the renewal. Both are deadlines, with opposite consequences.
    RENEWAL_OPTION_NOTICE = "renewal_option_notice"
    RENEWAL_OPTION_DEADLINE = "renewal_option_deadline"
    PAYMENT_OBLIGATION = "payment_obligation"
    FEE_ESCALATION = "fee_escalation"
    INDEMNIFICATION = "indemnification"
    LIABILITY_CAP = "liability_cap"
    GOVERNING_LAW = "governing_law"


#: Shorthand names accepted on the wire (the pitch deck and the UI mockup use
#: shorter labels than the canonical enum).
OBLIGATION_TYPE_ALIASES: dict[str, ObligationType] = {
    "term_start": ObligationType.CONTRACT_START_DATE,
    "start_date": ObligationType.CONTRACT_START_DATE,
    "term_end": ObligationType.CONTRACT_END_DATE,
    "end_date": ObligationType.CONTRACT_END_DATE,
    "termination_notice": ObligationType.TERMINATION_NOTICE_PERIOD,
    "notice": ObligationType.TERMINATION_NOTICE_PERIOD,
    "notice_period": ObligationType.TERMINATION_NOTICE_PERIOD,
    "deadline": ObligationType.NOTICE_DEADLINE,
    "renewal_notice": ObligationType.RENEWAL_OPTION_NOTICE,
    "notice_of_renewal": ObligationType.RENEWAL_OPTION_NOTICE,
    "option_to_renew_notice": ObligationType.RENEWAL_OPTION_NOTICE,
    "renewal_deadline": ObligationType.RENEWAL_OPTION_DEADLINE,
    "renewal_term": ObligationType.RENEWAL_DURATION,
    "auto_renewal": ObligationType.AUTOMATIC_RENEWAL,
    "payment": ObligationType.PAYMENT_OBLIGATION,
    "fee_increase": ObligationType.FEE_ESCALATION,
    "indemnity": ObligationType.INDEMNIFICATION,
    "indemnity_cap": ObligationType.LIABILITY_CAP,
    "liability": ObligationType.LIABILITY_CAP,
    "law": ObligationType.GOVERNING_LAW,
}


def coerce_obligation_type(value: str | ObligationType) -> ObligationType:
    """Accept canonical names, aliases and loose spacing/case."""
    if isinstance(value, ObligationType):
        return value
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if key in OBLIGATION_TYPE_ALIASES:
        return OBLIGATION_TYPE_ALIASES[key]
    return ObligationType(key)


class ObligationStatus(StrEnum):
    """Lifecycle of a single extracted field.

    ``proposed``       the model said it, nothing has been checked yet
    ``verified``       the supporting quote was found in the source page
    ``computed``       produced by deterministic Python code from verified inputs
    ``failed``         evidence missing, wrong page, or value unparseable
    ``human_verified`` a person confirmed/edited it (set by the approval UI)
    """

    PROPOSED = "proposed"
    VERIFIED = "verified"
    COMPUTED = "computed"
    FAILED = "failed"
    HUMAN_VERIFIED = "human_verified"


APPROVABLE_STATUSES: frozenset[ObligationStatus] = frozenset(
    {ObligationStatus.VERIFIED, ObligationStatus.COMPUTED, ObligationStatus.HUMAN_VERIFIED}
)


class VerificationMethod(StrEnum):
    """How a value earned its status. Never a model self-assessment."""

    NORMALIZED_EXACT_MATCH = "normalized_exact_match"
    FUZZY_MATCH = "fuzzy_match"
    DETERMINISTIC_COMPUTATION = "deterministic_computation"
    NONE = "none"


class DocumentRecord(BaseModel):
    id: str
    filename: str
    page_count: int
    uploaded_at: datetime
    chunk_count: int = 0
    sha256: str | None = None


class RetrievedChunk(BaseModel):
    id: str
    document_id: str
    page: int
    text: str
    lexical_score: float = 0.0
    vector_score: float = 0.0
    fused_score: float = 0.0


class SourceEvidence(BaseModel):
    """An exact span of the source document backing a value.

    ``start_offset`` / ``end_offset`` are character offsets into the stored
    text of ``page`` (the text returned by ``GET /v1/documents/{id}/pages``),
    so the approval UI can highlight the span directly.
    """

    quote: str
    page: int
    chunk_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class CandidateObligation(BaseModel):
    """A proposed ledger entry. Never authoritative until a human approves."""

    id: str
    document_id: str
    obligation_type: ObligationType
    raw_value: str
    normalized_value: str | None = None
    source_evidence: SourceEvidence | None = None
    status: ObligationStatus = ObligationStatus.PROPOSED
    verification_method: VerificationMethod = VerificationMethod.NONE
    verification_reason: str | None = None
    computation_formula: str | None = None
    computation_inputs: dict[str, Any] | None = None


class ExtractionFailure(BaseModel):
    obligation_type: ObligationType | None = None
    reason: str
    detail: str | None = None


class ExtractionResult(BaseModel):
    document_id: str
    obligations: list[CandidateObligation] = Field(default_factory=list)
    can_approve: bool = False
    failures: list[ExtractionFailure] = Field(default_factory=list)
    llm_mode: str = "unknown"
    retrieval_mode: str = "unknown"
    elapsed_seconds: float | None = None


# --------------------------------------------------------------------------
# Model-facing schema: what gpt-oss-120b is asked to return, nothing more.
# --------------------------------------------------------------------------


class LLMObligation(BaseModel):
    """One obligation as claimed by the model, before any verification."""

    model_config = ConfigDict(extra="ignore")

    obligation_type: str
    raw_value: str
    normalized_value: str | None = None
    quote: str
    page: int


class LLMExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    obligations: list[LLMObligation] = Field(default_factory=list)


class LLMContextChunk(BaseModel):
    """A retrieved chunk handed to the model as grounding context."""

    chunk_id: str
    page: int
    text: str


class LLMExtractionRequest(BaseModel):
    document_id: str
    obligation_types: list[ObligationType]
    context: list[LLMContextChunk]


# --------------------------------------------------------------------------
# HTTP request / response bodies
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    llm_mode: str
    llm_model: str
    embedding_backend: str
    retrieval_mode: str
    document_count: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    document_id: str
    query: str
    retrieval_mode: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    obligation_types: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class VerifyRequest(BaseModel):
    """Standalone re-verification of a quote (used by the approval UI on edit)."""

    document_id: str
    quote: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class VerifyResponse(BaseModel):
    document_id: str
    verified: bool
    page: int | None = None
    verification_method: VerificationMethod = VerificationMethod.NONE
    reason: str | None = None
    matched_text: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    similarity: float | None = None


class DeadlineOperation(StrEnum):
    NOTICE_DEADLINE = "notice_deadline"
    RENEWAL_OPTION_DEADLINE = "renewal_option_deadline"
    RENEWAL_DATE = "renewal_date"


class DeadlineComputeRequest(BaseModel):
    operation: DeadlineOperation = DeadlineOperation.NOTICE_DEADLINE
    #: ISO date or contract wording, e.g. "2026-03-31" or "March 31, 2026".
    anchor_date: str = Field(min_length=1)
    #: ISO-8601 duration or contract wording, e.g. "P60D" or "sixty (60) days".
    duration: str = Field(min_length=1)


class DeadlineComputeResponse(BaseModel):
    operation: DeadlineOperation
    result_date: str | None = None
    status: ObligationStatus
    computation_formula: str
    computation_inputs: dict[str, Any]
    error: str | None = None


class PageText(BaseModel):
    page: int
    text: str


class DocumentPagesResponse(BaseModel):
    document_id: str
    page_count: int
    pages: list[PageText]
