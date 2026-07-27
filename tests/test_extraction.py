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
    """The eleven types from the spec, plus the renewal-option pair the lease
    fixture forced us to add (a notice to *keep* a contract, not to leave it)."""
    spec = {
        "contract_start_date",
        "contract_end_date",
        "automatic_renewal",
        "renewal_duration",
        "termination_notice_period",
        "notice_deadline",
        "payment_obligation",
        "fee_escalation",
        "indemnification",
        "liability_cap",
        "governing_law",
    }
    values = {obligation_type.value for obligation_type in DEFAULT_OBLIGATION_TYPES}

    assert spec <= values
    assert values - spec == {"renewal_option_notice", "renewal_option_deadline"}


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


# --------------------------------------------------------------------------
# Regressions found by running the real sample dataset
# --------------------------------------------------------------------------


def test_a_clause_stating_two_durations_assigns_each_to_its_own_field():
    """From 07-ridgeline-supply-long.pdf: 60-month term, 270-day notice, one
    sentence. Taking the first duration for both made the notice 60 months and
    the computed deadline wrong by four years — while looking entirely plausible."""
    from obligation_rag.llm_client import FakeLLMClient
    from obligation_rag.schemas import LLMContextChunk, LLMExtractionRequest

    clause = (
        "RENEWAL This Agreement shall renew automatically for successive terms of "
        "sixty (60) months unless either party gives written notice of non-renewal "
        "at least two hundred and seventy (270) days before the end of the "
        "then-current term."
    )
    response = FakeLLMClient().propose_obligations(
        LLMExtractionRequest(
            document_id="doc",
            obligation_types=[
                ObligationType.RENEWAL_DURATION,
                ObligationType.TERMINATION_NOTICE_PERIOD,
            ],
            context=[LLMContextChunk(chunk_id="chunk_0", page=13, text=clause)],
        )
    )

    by_type = {item.obligation_type: item.raw_value for item in response.obligations}
    assert by_type["renewal_duration"] == "P60M"
    assert by_type["termination_notice_period"] == "P270D"


def test_non_renewal_does_not_anchor_the_renewal_term():
    """ "non-renewal" carries "renew" but belongs to the notice side."""
    from obligation_rag.llm_client import _anchor

    lowered = "gives written notice of non-renewal at least two hundred and seventy (270) days"

    assert _anchor(lowered, "renew", skip_after=("non-", "non ")) is None
    assert _anchor(lowered, "notice") == lowered.index("notice")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 06-tri-party: a cap with no figure is still a cap.
        (
            "Escrow Agent's liability shall not exceed the fees it has received",
            "Escrow Agent's liability shall not exceed the fees it has received",
        ),
        (
            "liability is limited to the amounts paid in the preceding 12 months",
            "liability is limited to the amounts paid in the preceding 12 months",
        ),
    ],
)
def test_a_cap_without_a_figure_is_kept_not_failed(raw, expected):
    assert normalize_value(ObligationType.LIABILITY_CAP, raw, "") == expected


def test_a_figure_still_wins_over_the_clause_text():
    assert (
        normalize_value(ObligationType.LIABILITY_CAP, "shall not exceed $250,000", "")
        == "USD 250000.00"
    )


def test_a_cap_expressed_as_a_multiple_of_fees_is_not_read_as_an_amount():
    """From 01-harborview-msa-clean.pdf. The old money regex matched the "12"
    of "twelve (12) months" and the "m" of "months" as the million multiplier,
    recording a USD 12,000,000 cap on a contract billing USD 180,000 a year —
    verified, approvable, and invented."""
    clause = (
        "Neither party's aggregate liability under this Agreement shall exceed the "
        "total fees paid in the twelve (12) months preceding the claim."
    )

    value = normalize_value(ObligationType.LIABILITY_CAP, clause, "")

    assert value is not None
    assert "12000000" not in value
    assert "exceed the" in value


@pytest.mark.parametrize(
    "text", ["payable within thirty (30) days", "cure the breach within 15 days"]
)
def test_a_bare_number_is_never_money(text):
    from obligation_rag.extraction import _normalize_money

    assert _normalize_money(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("USD 180,000 per annum", "USD 180000.00"),
        ("$120,000.00", "USD 120000.00"),
        ("EUR 1.5 million", "EUR 1500000.00"),
        ("2,400,000 dollars", "USD 2400000.00"),
    ],
)
def test_money_still_parses_when_a_currency_says_so(text, expected):
    from obligation_rag.extraction import _normalize_money

    assert _normalize_money(text) == expected
