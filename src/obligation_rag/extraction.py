"""The pipeline: retrieve -> propose -> verify -> normalize -> compute.

Order matters and is not negotiable:

1. retrieval decides what the model is allowed to see
2. the model proposes values *with quotes*
3. Python verifies every quote against the page it claims (verification.py)
4. Python normalizes values into machine-usable form
5. Python computes derived deadlines from verified inputs (date_math.py)
6. anything unverifiable is marked failed and blocks approval

A model-proposed ``notice_deadline`` is discarded on principle: derived dates
come from code or they do not come at all.
"""

from __future__ import annotations

import logging
import re
import time

from .config import Settings
from .date_math import (
    compute_notice_deadline,
    compute_renewal_option_deadline,
    parse_date,
    parse_duration,
)
from .llm_client import LLMClient, LLMClientError, get_llm_client
from .retrieval import DocumentIndex
from .schemas import (
    APPROVABLE_STATUSES,
    CandidateObligation,
    ExtractionFailure,
    ExtractionResult,
    LLMContextChunk,
    LLMExtractionRequest,
    LLMObligation,
    ObligationStatus,
    ObligationType,
    RetrievedChunk,
    SourceEvidence,
    VerificationMethod,
    coerce_obligation_type,
)
from .verification import verify_evidence

logger = logging.getLogger(__name__)

#: Lexical queries, written the way contracts are written — BM25 rewards that.
QUERY_TEMPLATES: dict[ObligationType, str] = {
    ObligationType.CONTRACT_START_DATE: (
        "effective date commencement date this agreement commences begins term start"
    ),
    ObligationType.CONTRACT_END_DATE: (
        "initial term expires expiration end of term shall continue until termination date"
    ),
    ObligationType.AUTOMATIC_RENEWAL: (
        "automatically renew automatic renewal successive renewal terms unless terminated"
    ),
    ObligationType.RENEWAL_DURATION: (
        "renewal term successive periods of twelve months renew for additional term"
    ),
    ObligationType.TERMINATION_NOTICE_PERIOD: (
        "written notice of termination prior written notice days notice to terminate non-renewal"
    ),
    ObligationType.NOTICE_DEADLINE: (
        "notice must be given no later than deadline to provide notice of non-renewal"
    ),
    ObligationType.RENEWAL_OPTION_NOTICE: (
        "option to renew shall be exercised by providing written notice given not less than "
        "days prior to the termination date lapse and expire"
    ),
    ObligationType.RENEWAL_OPTION_DEADLINE: (
        "last day to exercise the option to renew before it lapses"
    ),
    ObligationType.PAYMENT_OBLIGATION: (
        "fees payable payment terms invoice net days annual fee amount due customer shall pay"
    ),
    ObligationType.FEE_ESCALATION: (
        "fees increase annually escalation percent uplift price adjustment CPI"
    ),
    ObligationType.INDEMNIFICATION: (
        "indemnify defend hold harmless third party claims indemnification"
    ),
    ObligationType.LIABILITY_CAP: (
        "limitation of liability aggregate liability shall not exceed cap damages"
    ),
    ObligationType.GOVERNING_LAW: (
        "governed by the laws of jurisdiction venue exclusive courts governing law"
    ),
}

DEFAULT_OBLIGATION_TYPES: list[ObligationType] = list(ObligationType)

#: Never asked of the model — computed from verified inputs instead.
COMPUTED_ONLY_TYPES: frozenset[ObligationType] = frozenset(
    {ObligationType.NOTICE_DEADLINE, ObligationType.RENEWAL_OPTION_DEADLINE}
)

DATE_TYPES: frozenset[ObligationType] = frozenset(
    {ObligationType.CONTRACT_START_DATE, ObligationType.CONTRACT_END_DATE}
)
DURATION_TYPES: frozenset[ObligationType] = frozenset(
    {
        ObligationType.TERMINATION_NOTICE_PERIOD,
        ObligationType.RENEWAL_DURATION,
        ObligationType.RENEWAL_OPTION_NOTICE,
    }
)
MONEY_TYPES: frozenset[ObligationType] = frozenset(
    {ObligationType.PAYMENT_OBLIGATION, ObligationType.LIABILITY_CAP}
)

_MONEY = re.compile(
    r"(?P<symbol>USD|EUR|GBP|CHF|CAD|AUD|\$|€|£)?\s?(?P<amount>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<scale>million|billion|thousand|k|m)?",
    re.IGNORECASE,
)
_SYMBOL_TO_CODE = {"$": "USD", "€": "EUR", "£": "GBP"}
_SCALE = {
    "thousand": 1_000,
    "k": 1_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
}
_PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
#: A renewal that a party must *choose* is not an automatic renewal. Filled
#: lease forms say "This Lease may be renewed" and mean the opposite of a
#: hands-off evergreen clause, so these phrasings normalise to false.
_NEGATIVE_RENEWAL = (
    "not automatically renew",
    "shall not renew",
    "no automatic renewal",
    "may not be renewed",
    "may be renewed",
    "option to renew",
    "shall have the option",
    "at tenant's option",
    "at its option",
)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def retrieve_context(
    index: DocumentIndex,
    obligation_types: list[ObligationType],
    *,
    top_k: int,
    max_chunks: int = 14,
) -> list[RetrievedChunk]:
    """Union of the per-type retrievals, best-scoring chunks first."""
    best: dict[str, RetrievedChunk] = {}
    for obligation_type in obligation_types:
        query = QUERY_TEMPLATES.get(obligation_type, obligation_type.value.replace("_", " "))
        for chunk in index.search(query, top_k=top_k):
            existing = best.get(chunk.id)
            if existing is None or chunk.fused_score > existing.fused_score:
                best[chunk.id] = chunk

    ordered = sorted(best.values(), key=lambda chunk: (-chunk.fused_score, chunk.page, chunk.id))
    selected = ordered[:max_chunks]
    # Reading order makes the excerpts easier for the model to reason over.
    return sorted(selected, key=lambda chunk: (chunk.page, chunk.id))


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _normalize_money(text: str) -> str | None:
    match = _MONEY.search(text)
    if not match:
        return None
    amount = float(match.group("amount").replace(",", ""))
    scale = match.group("scale")
    if scale:
        amount *= _SCALE[scale.lower()]
    symbol = (match.group("symbol") or "USD").upper()
    currency = _SYMBOL_TO_CODE.get(symbol, symbol)
    return f"{currency} {amount:,.2f}".replace(",", "")


def normalize_value(obligation_type: ObligationType, raw_value: str, quote: str) -> str | None:
    """Canonical machine-usable form of a value, or ``None`` if unparseable.

    The quote is used as a fallback source: a model that writes "sixty days"
    in ``raw_value`` while the page says "sixty (60) days" still gets P60D.
    """
    candidates = [raw_value, quote]

    if obligation_type in DATE_TYPES:
        for candidate in candidates:
            parsed = parse_date(candidate)
            if parsed:
                return parsed.isoformat()
        return None

    if obligation_type in DURATION_TYPES:
        for candidate in candidates:
            parsed = parse_duration(candidate)
            if parsed:
                return parsed.to_iso()
        return None

    if obligation_type is ObligationType.AUTOMATIC_RENEWAL:
        haystack = f"{raw_value} {quote}".lower()
        if any(marker in haystack for marker in _NEGATIVE_RENEWAL):
            return "false"
        if "false" in raw_value.lower() or raw_value.strip().lower() == "no":
            return "false"
        if "renew" in haystack or "true" in raw_value.lower():
            return "true"
        return None

    if obligation_type in MONEY_TYPES:
        for candidate in candidates:
            normalized = _normalize_money(candidate)
            if normalized:
                return normalized
        return None

    if obligation_type is ObligationType.FEE_ESCALATION:
        for candidate in candidates:
            match = _PERCENT.search(candidate)
            if match:
                return f"{float(match.group(1)):g}%"
        return None

    if obligation_type is ObligationType.GOVERNING_LAW:
        cleaned = re.sub(r"\s+", " ", raw_value).strip(" .,;")
        return cleaned or None

    cleaned = re.sub(r"\s+", " ", raw_value).strip()
    return cleaned or None


# --------------------------------------------------------------------------
# Verification of one proposed obligation
# --------------------------------------------------------------------------


def _locate_chunk(
    index: DocumentIndex, page: int, start: int | None, end: int | None
) -> str | None:
    for chunk in index.chunks:
        if chunk.page != page:
            continue
        if start is None or end is None:
            return chunk.chunk_id
        if chunk.start_offset <= start and end <= chunk.end_offset:
            return chunk.chunk_id
    return None


def _build_candidate(
    settings: Settings,
    document_id: str,
    index: DocumentIndex,
    pages: dict[int, str],
    proposal: LLMObligation,
    obligation_type: ObligationType,
) -> CandidateObligation:
    outcome = verify_evidence(
        proposal.quote,
        pages,
        claimed_page=proposal.page,
        fuzzy_threshold=settings.fuzzy_match_threshold,
        fuzzy_max_length_ratio=settings.fuzzy_max_length_ratio,
        min_quote_chars=settings.min_quote_chars,
        reject_unchecked_options=settings.reject_unchecked_options,
        # Anchors the checkbox test on the value, not on wherever the quote
        # happens to begin — quotes routinely cover a whole option block.
        value=proposal.raw_value,
    )

    evidence = SourceEvidence(
        quote=outcome.matched_text or proposal.quote,
        page=outcome.page if outcome.verified and outcome.page else proposal.page,
        chunk_id=(
            _locate_chunk(index, proposal.page, outcome.start_offset, outcome.end_offset)
            if outcome.verified
            else None
        ),
        start_offset=outcome.start_offset,
        end_offset=outcome.end_offset,
    )

    candidate = CandidateObligation(
        id=f"{document_id}:{obligation_type.value}",
        document_id=document_id,
        obligation_type=obligation_type,
        raw_value=proposal.raw_value,
        source_evidence=evidence,
        status=ObligationStatus.PROPOSED,
        verification_method=outcome.method,
        verification_reason=outcome.reason,
    )

    if not outcome.verified:
        candidate.status = ObligationStatus.FAILED
        candidate.verification_reason = outcome.reason or "quote_not_verified"
        return candidate

    normalized = normalize_value(obligation_type, proposal.raw_value, evidence.quote)
    candidate.normalized_value = normalized
    if normalized is None:
        candidate.status = ObligationStatus.FAILED
        candidate.verification_reason = (
            f"value_not_normalizable: quote verified but {proposal.raw_value!r} "
            f"could not be parsed as {obligation_type.value}"
        )
        return candidate

    candidate.status = ObligationStatus.VERIFIED
    return candidate


# --------------------------------------------------------------------------
# Derived values
# --------------------------------------------------------------------------


#: Derived deadline -> (notice input type, computation). Both are
#: ``end_date - notice`` but they mean opposite things to the reviewer, so they
#: are separate rows in the ledger with separate labels.
DERIVED_DEADLINES: tuple[tuple[ObligationType, ObligationType, str], ...] = (
    (
        ObligationType.NOTICE_DEADLINE,
        ObligationType.TERMINATION_NOTICE_PERIOD,
        "notice_deadline",
    ),
    (
        ObligationType.RENEWAL_OPTION_DEADLINE,
        ObligationType.RENEWAL_OPTION_NOTICE,
        "renewal_option_deadline",
    ),
)


def compute_derived_obligations(
    document_id: str,
    obligations: list[CandidateObligation],
    requested: list[ObligationType] | None = None,
) -> tuple[list[CandidateObligation], list[ExtractionFailure]]:
    """Compute deadlines in code from verified inputs only."""
    by_type = {
        obligation.obligation_type: obligation
        for obligation in obligations
        if obligation.status in APPROVABLE_STATUSES
    }
    end_date = by_type.get(ObligationType.CONTRACT_END_DATE)

    derived: list[CandidateObligation] = []
    failures: list[ExtractionFailure] = []

    for deadline_type, notice_type, label in DERIVED_DEADLINES:
        if requested is not None and deadline_type not in requested:
            continue
        notice = by_type.get(notice_type)

        if end_date is None or notice is None:
            missing = [
                name
                for name, value in (
                    ("contract_end_date", end_date),
                    (notice_type.value, notice),
                )
                if value is None
            ]
            # A contract has either a termination notice or a renewal option,
            # rarely both; only report the one whose notice was actually found.
            if notice is not None and missing:
                failures.append(
                    ExtractionFailure(
                        obligation_type=deadline_type,
                        reason="missing_verified_inputs",
                        detail=(
                            f"{label} was not computed because these verified inputs are "
                            f"missing: {', '.join(missing)}"
                        ),
                    )
                )
            continue

        if not end_date.normalized_value or not notice.normalized_value:
            continue

        computation = (
            compute_notice_deadline(end_date.normalized_value, notice.normalized_value)
            if deadline_type is ObligationType.NOTICE_DEADLINE
            else compute_renewal_option_deadline(end_date.normalized_value, notice.normalized_value)
        )
        if not computation.ok:
            failures.append(
                ExtractionFailure(
                    obligation_type=deadline_type,
                    reason="computation_failed",
                    detail=computation.error,
                )
            )
            continue

        inputs = dict(computation.inputs)
        inputs["source_obligation_ids"] = [end_date.id, notice.id]
        derived.append(
            CandidateObligation(
                id=f"{document_id}:{deadline_type.value}",
                document_id=document_id,
                obligation_type=deadline_type,
                raw_value=computation.result_iso or "",
                normalized_value=computation.result_iso,
                source_evidence=None,  # derived, not quoted: its evidence is its inputs
                status=ObligationStatus.COMPUTED,
                verification_method=VerificationMethod.DETERMINISTIC_COMPUTATION,
                verification_reason="calculated in code, not model output",
                computation_formula=computation.formula,
                computation_inputs=inputs,
            )
        )

    # Nothing derived and nothing reported: say once what was missing, rather
    # than letting a contract with no deadline at all pass unremarked.
    if not derived and not failures:
        deadline_type, notice_type, label = DERIVED_DEADLINES[0]
        wanted = requested if requested is not None else [pair[0] for pair in DERIVED_DEADLINES]
        missing = [
            name
            for name, value in (
                ("contract_end_date", end_date),
                (notice_type.value, by_type.get(notice_type)),
            )
            if value is None
        ]
        if deadline_type in wanted and missing:
            failures.append(
                ExtractionFailure(
                    obligation_type=deadline_type,
                    reason="missing_verified_inputs",
                    detail=(
                        f"{label} was not computed because these verified inputs are "
                        f"missing: {', '.join(missing)}"
                    ),
                )
            )
    return derived, failures


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_extraction(
    settings: Settings,
    document_id: str,
    index: DocumentIndex,
    pages: dict[int, str],
    *,
    obligation_types: list[ObligationType] | None = None,
    top_k: int | None = None,
    client: LLMClient | None = None,
) -> ExtractionResult:
    started = time.perf_counter()
    requested = obligation_types or DEFAULT_OBLIGATION_TYPES
    client = client or get_llm_client(settings)
    failures: list[ExtractionFailure] = []

    askable = [
        obligation_type
        for obligation_type in requested
        if obligation_type not in COMPUTED_ONLY_TYPES
    ]
    context = retrieve_context(index, askable or requested, top_k=top_k or settings.default_top_k)
    if not context:
        return ExtractionResult(
            document_id=document_id,
            can_approve=False,
            failures=[
                ExtractionFailure(
                    reason="no_context_retrieved", detail="retrieval returned nothing"
                )
            ],
            llm_mode=client.mode,
            retrieval_mode=index.retrieval_mode,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )

    request = LLMExtractionRequest(
        document_id=document_id,
        obligation_types=askable,
        context=[
            LLMContextChunk(chunk_id=chunk.id, page=chunk.page, text=chunk.text)
            for chunk in context
        ],
    )

    try:
        response = client.propose_obligations(request)
    except LLMClientError as error:
        return ExtractionResult(
            document_id=document_id,
            can_approve=False,
            failures=[ExtractionFailure(reason="llm_unavailable", detail=str(error))],
            llm_mode=client.mode,
            retrieval_mode=index.retrieval_mode,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )

    obligations: list[CandidateObligation] = []
    seen: set[ObligationType] = set()

    for proposal in response.obligations:
        try:
            obligation_type = coerce_obligation_type(proposal.obligation_type)
        except ValueError:
            failures.append(
                ExtractionFailure(
                    reason="unknown_obligation_type", detail=str(proposal.obligation_type)
                )
            )
            continue

        if obligation_type in COMPUTED_ONLY_TYPES:
            # Trust rule #2: a derived date from the model is never authoritative.
            failures.append(
                ExtractionFailure(
                    obligation_type=obligation_type,
                    reason="model_proposed_computed_field_ignored",
                    detail=(
                        f"{obligation_type.value} is computed by deterministic code; "
                        "the model's proposal was discarded"
                    ),
                )
            )
            continue

        if obligation_type in seen or obligation_type not in requested:
            continue
        seen.add(obligation_type)

        obligations.append(
            _build_candidate(settings, document_id, index, pages, proposal, obligation_type)
        )

    derived, derived_failures = compute_derived_obligations(document_id, obligations, requested)
    obligations.extend(derived)
    failures.extend(derived_failures)

    can_approve = bool(obligations) and all(
        obligation.status in APPROVABLE_STATUSES for obligation in obligations
    )

    return ExtractionResult(
        document_id=document_id,
        obligations=obligations,
        can_approve=can_approve,
        failures=failures,
        llm_mode=client.mode,
        retrieval_mode=index.retrieval_mode,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
