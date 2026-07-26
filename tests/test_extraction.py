"""Pipeline tests with the fake LLM, including the spec's ablation demo (§10)."""

from __future__ import annotations

import pytest

from obligation_rag.extraction import DEFAULT_OBLIGATION_TYPES, normalize_value, run_extraction
from obligation_rag.llm_client import (
    FakeLLMClient,
    HallucinatingLLMClient,
    LLMClient,
    LLMClientError,
    _extract_json_object,
)
from obligation_rag.schemas import (
    LLMExtractionResponse,
    LLMObligation,
    ObligationStatus,
    ObligationType,
    VerificationMethod,
)

DOCUMENT_ID = "contract_test"


@pytest.fixture
def result(settings, bm25_index, contract_pages):
    return run_extraction(settings, DOCUMENT_ID, bm25_index, contract_pages, client=FakeLLMClient())


def test_pipeline_extracts_the_whole_obligation_set(result):
    found = {obligation.obligation_type for obligation in result.obligations}

    assert ObligationType.CONTRACT_START_DATE in found
    assert ObligationType.CONTRACT_END_DATE in found
    assert ObligationType.TERMINATION_NOTICE_PERIOD in found
    assert ObligationType.NOTICE_DEADLINE in found
    assert result.llm_mode == "fake"


def test_every_extracted_value_is_normalized(result):
    by_type = {obligation.obligation_type: obligation for obligation in result.obligations}

    assert by_type[ObligationType.CONTRACT_START_DATE].normalized_value == "2024-03-01"
    assert by_type[ObligationType.CONTRACT_END_DATE].normalized_value == "2026-03-31"
    assert by_type[ObligationType.TERMINATION_NOTICE_PERIOD].normalized_value == "P60D"
    assert by_type[ObligationType.RENEWAL_DURATION].normalized_value == "P12M"
    assert by_type[ObligationType.AUTOMATIC_RENEWAL].normalized_value == "true"
    assert by_type[ObligationType.PAYMENT_OBLIGATION].normalized_value == "USD 120000.00"
    assert by_type[ObligationType.LIABILITY_CAP].normalized_value == "USD 250000.00"
    assert by_type[ObligationType.FEE_ESCALATION].normalized_value == "4%"
    assert by_type[ObligationType.GOVERNING_LAW].normalized_value == "Delaware"


def test_every_extracted_value_carries_verified_evidence(result, contract_pages):
    extracted = [
        obligation
        for obligation in result.obligations
        if obligation.obligation_type is not ObligationType.NOTICE_DEADLINE
    ]

    assert extracted
    for obligation in extracted:
        assert obligation.status is ObligationStatus.VERIFIED
        evidence = obligation.source_evidence
        assert evidence is not None
        assert evidence.chunk_id
        page_text = contract_pages[evidence.page]
        assert page_text[evidence.start_offset : evidence.end_offset] == evidence.quote


def test_the_deadline_is_computed_in_code_with_its_audit_trail(result):
    deadline = next(
        obligation
        for obligation in result.obligations
        if obligation.obligation_type is ObligationType.NOTICE_DEADLINE
    )

    assert deadline.status is ObligationStatus.COMPUTED
    assert deadline.verification_method is VerificationMethod.DETERMINISTIC_COMPUTATION
    assert deadline.normalized_value == "2026-01-30"  # 2026-03-31 minus 60 days
    assert deadline.computation_formula == (
        "notice_deadline = contract_end_date - termination_notice_period"
    )
    assert deadline.computation_inputs["source_obligation_ids"] == [
        f"{DOCUMENT_ID}:contract_end_date",
        f"{DOCUMENT_ID}:termination_notice_period",
    ]
    assert deadline.source_evidence is None, "a derived date is not quoted, it is computed"


def test_a_clean_run_can_be_approved(result):
    assert result.can_approve is True
    assert result.failures == []


# --------------------------------------------------------------------------
# Ablation (spec §10.1): the same pipeline, a model that invents a date.
# --------------------------------------------------------------------------


def test_hallucinated_quote_is_caught_and_blocks_approval(settings, bm25_index, contract_pages):
    result = run_extraction(
        settings, DOCUMENT_ID, bm25_index, contract_pages, client=HallucinatingLLMClient()
    )

    end_date = next(
        obligation
        for obligation in result.obligations
        if obligation.obligation_type is ObligationType.CONTRACT_END_DATE
    )
    assert end_date.raw_value == "December 31, 2029"
    assert end_date.status is ObligationStatus.FAILED
    assert "quote_not_found_on_page" in end_date.verification_reason
    assert result.can_approve is False


def test_no_deadline_is_computed_from_unverified_inputs(settings, bm25_index, contract_pages):
    result = run_extraction(
        settings, DOCUMENT_ID, bm25_index, contract_pages, client=HallucinatingLLMClient()
    )

    assert all(
        obligation.obligation_type is not ObligationType.NOTICE_DEADLINE
        for obligation in result.obligations
    )
    assert any(failure.reason == "missing_verified_inputs" for failure in result.failures)


# --------------------------------------------------------------------------
# Adversarial model behaviour
# --------------------------------------------------------------------------


class _ScriptedClient(LLMClient):
    mode = "scripted"

    def __init__(self, obligations: list[LLMObligation]) -> None:
        self._obligations = obligations

    def propose_obligations(self, request) -> LLMExtractionResponse:
        return LLMExtractionResponse(obligations=self._obligations)


def test_a_model_supplied_deadline_is_discarded(settings, bm25_index, contract_pages):
    """Trust rule #2: derived dates come from code or they do not come at all."""
    client = _ScriptedClient(
        [
            LLMObligation(
                obligation_type="notice_deadline",
                raw_value="2026-02-15",
                quote="Either party may elect not to renew this Agreement",
                page=3,
            )
        ]
    )

    result = run_extraction(settings, DOCUMENT_ID, bm25_index, contract_pages, client=client)

    assert result.obligations == []
    assert any(
        failure.reason == "model_proposed_computed_field_ignored" for failure in result.failures
    )
    assert result.can_approve is False


def test_a_real_quote_on_the_wrong_page_still_fails(settings, bm25_index, contract_pages):
    client = _ScriptedClient(
        [
            LLMObligation(
                obligation_type="term_end",
                raw_value="March 31, 2026",
                quote="The initial term of this Agreement expires on March 31, 2026",
                page=1,  # it is on page 2
            )
        ]
    )

    result = run_extraction(settings, DOCUMENT_ID, bm25_index, contract_pages, client=client)

    assert result.obligations[0].status is ObligationStatus.FAILED
    assert "wrong_page" in result.obligations[0].verification_reason
    assert "page 2" in result.obligations[0].verification_reason
    assert result.can_approve is False


def test_a_verified_quote_with_an_unusable_value_fails(settings, bm25_index, contract_pages):
    client = _ScriptedClient(
        [
            LLMObligation(
                obligation_type="contract_end_date",
                raw_value="whenever the parties agree",
                quote="3.4 Survival.",
                page=3,
            )
        ]
    )

    result = run_extraction(settings, DOCUMENT_ID, bm25_index, contract_pages, client=client)

    assert result.obligations[0].status is ObligationStatus.FAILED
    assert result.can_approve is False


def test_unknown_obligation_types_are_reported_not_crashed(settings, bm25_index, contract_pages):
    client = _ScriptedClient(
        [
            LLMObligation(
                obligation_type="favourite_colour",
                raw_value="blue",
                quote="3.4 Survival. Sections 4, 7, 8 and 11 survive any termination",
                page=3,
            )
        ]
    )

    result = run_extraction(settings, DOCUMENT_ID, bm25_index, contract_pages, client=client)

    assert result.obligations == []
    assert result.failures[0].reason == "unknown_obligation_type"


def test_an_unreachable_model_degrades_to_a_failure_not_an_exception(
    settings, bm25_index, contract_pages
):
    class _DeadClient(LLMClient):
        mode = "dead"

        def propose_obligations(self, request):
            raise LLMClientError("connection refused")

    result = run_extraction(settings, DOCUMENT_ID, bm25_index, contract_pages, client=_DeadClient())

    assert result.can_approve is False
    assert result.failures[0].reason == "llm_unavailable"


def test_requested_subset_is_respected(settings, bm25_index, contract_pages):
    result = run_extraction(
        settings,
        DOCUMENT_ID,
        bm25_index,
        contract_pages,
        obligation_types=[ObligationType.GOVERNING_LAW],
        client=FakeLLMClient(),
    )

    assert [obligation.obligation_type for obligation in result.obligations] == [
        ObligationType.GOVERNING_LAW
    ]


def test_default_type_list_covers_the_spec():
    assert len(DEFAULT_OBLIGATION_TYPES) == 11


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obligation_type", "raw", "quote", "expected"),
    [
        (ObligationType.CONTRACT_END_DATE, "March 31, 2026", "", "2026-03-31"),
        (
            ObligationType.CONTRACT_END_DATE,
            "the last day",
            "expires on March 31, 2026",
            "2026-03-31",
        ),
        (ObligationType.TERMINATION_NOTICE_PERIOD, "sixty days", "", "P60D"),
        (ObligationType.PAYMENT_OBLIGATION, "$120,000.00", "", "USD 120000.00"),
        (ObligationType.PAYMENT_OBLIGATION, "EUR 1.5 million", "", "EUR 1500000.00"),
        (ObligationType.LIABILITY_CAP, "£250,000", "", "GBP 250000.00"),
        (ObligationType.FEE_ESCALATION, "four percent (4%)", "", "4%"),
        (ObligationType.AUTOMATIC_RENEWAL, "yes", "shall automatically renew", "true"),
        (ObligationType.AUTOMATIC_RENEWAL, "no", "shall not renew automatically", "false"),
        (ObligationType.GOVERNING_LAW, " Delaware ", "", "Delaware"),
    ],
)
def test_normalize_value(obligation_type, raw, quote, expected):
    assert normalize_value(obligation_type, raw, quote) == expected


@pytest.mark.parametrize(
    ("obligation_type", "raw"),
    [
        (ObligationType.CONTRACT_START_DATE, "at some point"),
        (ObligationType.TERMINATION_NOTICE_PERIOD, "reasonable notice"),
        (ObligationType.LIABILITY_CAP, "unlimited"),
    ],
)
def test_normalize_value_refuses_to_invent(obligation_type, raw):
    assert normalize_value(obligation_type, raw, "") is None


def test_json_repair_handles_fenced_and_chatty_model_output():
    assert _extract_json_object('```json\n{"obligations": []}\n```') == '{"obligations": []}'
    assert _extract_json_object('Sure!\n{"a": {"b": 1}} hope that helps') == '{"a": {"b": 1}}'
    assert (
        _extract_json_object('{"quote": "he said \\"hi\\" }"}') == '{"quote": "he said \\"hi\\" }"}'
    )

    with pytest.raises(LLMClientError, match="model_returned_no_json"):
        _extract_json_object("I could not find anything.")
    with pytest.raises(LLMClientError, match="truncated"):
        _extract_json_object('{"obligations": [')
