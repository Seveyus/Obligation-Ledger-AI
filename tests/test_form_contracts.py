"""Filled-template contracts: checkboxes, running headers, renewal options.

These are the failure modes the prose fixture cannot exercise. They were found
by running the pipeline against a real filled lease form from the dataset, and
each one silently produced a *wrong but approvable* answer before the fix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from obligation_rag import storage
from obligation_rag.config import Settings
from obligation_rag.date_math import compute_renewal_option_deadline
from obligation_rag.extraction import DEFAULT_OBLIGATION_TYPES, retrieve_context, run_extraction
from obligation_rag.ingestion import ingest_path
from obligation_rag.llm_client import FakeLLMClient
from obligation_rag.pdf_parser import parse_document, strip_page_furniture
from obligation_rag.retrieval import clear_index_cache, get_document_index
from obligation_rag.schemas import ObligationStatus, ObligationType
from obligation_rag.verification import (
    preceding_checkbox,
    unchecked_option_reason,
    verify_evidence,
    verify_quote_on_page,
)

LEASE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_lease_form.txt"

PAGE = (
    "Rent Increase (Check one)\n"
    "☐ Rent will NOT be increased. All of the terms and conditions shall apply\n"
    "during each renewal term.\n"
    "☒ Rent will be increased, except that the Base Rent shall be increased by:\n"
    "☒ 3% each renewal term.\n"
    "☐ $___________ each renewal term.\n"
)


# --------------------------------------------------------------------------
# 1. Checkbox options
# --------------------------------------------------------------------------


def test_a_quote_from_a_ticked_option_verifies():
    outcome = verify_quote_on_page("☒ 3% each renewal term.", PAGE, page=3)

    assert outcome.verified


def test_a_quote_from_an_unticked_option_is_rejected():
    """The text is really on the page — it is just not what was agreed."""
    outcome = verify_quote_on_page("☐ Rent will NOT be increased.", PAGE, page=3)

    assert not outcome.verified
    assert "unchecked_option" in outcome.reason


def test_the_tick_is_found_even_when_the_quote_omits_it():
    """Models quote the sentence, not the glyph in front of it."""
    ticked = verify_quote_on_page("Rent will be increased, except that the Base Rent", PAGE, page=3)
    unticked = verify_quote_on_page(
        "Rent will NOT be increased. All of the terms and conditions", PAGE, page=3
    )

    assert ticked.verified
    assert not unticked.verified
    assert "unchecked_option" in unticked.reason


def test_a_checkbox_on_a_previous_line_does_not_govern_the_quote():
    outcome = verify_quote_on_page("during each renewal term.\n☒ Rent will be", PAGE, page=3)

    assert outcome.verified, "the ☐ two lines up governs a different option"


def test_prose_without_any_checkbox_is_unaffected():
    prose = "Either party may terminate this Agreement on sixty (60) days written notice."

    assert verify_quote_on_page(
        "terminate this Agreement on sixty (60) days", prose, page=1
    ).verified


def test_inline_options_on_one_line_are_resolved_individually():
    line = "Tenant shall pay for: ☒ Water ☐ Gas ☒ Power ☐ Sewage Disposal charges"

    assert preceding_checkbox(line, line.index("Water")) == "☒"
    assert preceding_checkbox(line, line.index("Gas")) == "☐"
    assert unchecked_option_reason(line, line.index("Power")) is None
    assert unchecked_option_reason(line, line.index("Sewage")) is not None


def test_a_quote_covering_both_options_is_judged_on_the_value():
    """Found on the real PDF: the quote opens on the empty box but the value
    it supports is the ticked one. Judging the span start condemned it."""
    page = (
        "such insurance to have: (Check one)\n"
        "☐ A minimum aggregate policy in the amount of no less than $__________\n"
        "☒ Limits of liability of not less than $1,000,000 per occurrence\n"
    )
    quote = (
        "☐ A minimum aggregate policy in the amount of no less than $__________\n"
        "☒ Limits of liability of not less than $1,000,000 per occurrence"
    )

    assert verify_quote_on_page(quote, page, page=11, value="$1,000,000").verified
    assert not verify_quote_on_page(quote, page, page=11, value="minimum aggregate").verified
    assert not verify_quote_on_page(quote, page, page=11).verified  # no value: span start rules


def test_the_rule_can_be_switched_off():
    outcome = verify_quote_on_page(
        "☐ Rent will NOT be increased.", PAGE, page=3, reject_unchecked_options=False
    )

    assert outcome.verified


def test_unchecked_options_block_approval_end_to_end(tmp_path: Path):
    pages = {3: PAGE}

    outcome = verify_evidence("Rent will NOT be increased. All of the terms", pages, claimed_page=3)

    assert not outcome.verified


# --------------------------------------------------------------------------
# 2. Running headers and footers
# --------------------------------------------------------------------------


def test_repeated_headers_are_stripped():
    pages = [f"HEADER LINE\nclause {n} body text\nFOOTER LINE" for n in range(5)]

    cleaned = strip_page_furniture(pages)

    assert all("HEADER LINE" not in page for page in cleaned)
    assert all("FOOTER LINE" not in page for page in cleaned)
    assert "clause 3 body text" in cleaned[3]


def test_a_line_that_only_repeats_twice_is_kept():
    pages = ["shared line\nbody a", "shared line\nbody b", "other\nbody c", "other2\nbody d"]

    assert all("shared line" in page for page in strip_page_furniture(pages[:2]) if page)
    assert "shared line" in strip_page_furniture(pages)[0]


def test_a_repeated_line_in_the_middle_of_a_page_is_kept():
    """Only head/tail zones are furniture; a repeated clause body is content."""
    pages = [f"head {n}\nx\nx\nSHARED CLAUSE BODY\ny\ny\ntail {n}" for n in range(5)]

    cleaned = strip_page_furniture(pages)

    assert all("SHARED CLAUSE BODY" in page for page in cleaned)


def test_short_documents_are_left_alone():
    pages = ["HEADER\na", "HEADER\nb"]

    assert strip_page_furniture(pages) == pages


def test_the_lease_fixture_loses_its_running_header():
    parsed = parse_document(LEASE)

    assert parsed.page_count == 19
    assert all("INITIAL ________ DATE" not in page.text for page in parsed.pages)
    assert "Commencement" in parsed.pages[2].text  # content survives

    kept = parse_document(LEASE, strip_furniture=False)
    assert "INITIAL ________ DATE" in kept.pages[2].text


# --------------------------------------------------------------------------
# 3. Renewal-option notice (the inverse of a termination notice)
# --------------------------------------------------------------------------


def test_renewal_option_deadline_is_computed_with_its_consequence():
    computation = compute_renewal_option_deadline("2031-01-31", "not less than 90 days")

    assert computation.result_iso == "2030-11-02"
    assert computation.formula == (
        "renewal_option_deadline = contract_end_date - renewal_option_notice"
    )
    assert computation.inputs["consequence"] == "the renewal option lapses after this date"


@pytest.fixture
def lease_extraction():
    with tempfile.TemporaryDirectory() as scratch:
        settings = Settings(rag_data_dir=Path(scratch), use_fake_llm=True)
        clear_index_cache()
        outcome = ingest_path(settings, LEASE)
        index = get_document_index(settings, outcome.document_id)
        pages = storage.get_pages(settings, outcome.document_id)
        yield run_extraction(settings, outcome.document_id, index, pages, client=FakeLLMClient())
        clear_index_cache()


def test_the_lease_notice_is_not_labelled_a_termination_notice(lease_extraction):
    """The 90 days keep the lease alive; filing them under "termination
    notice" would tell the reviewer the opposite of what the clause says."""
    by_type = {o.obligation_type: o for o in lease_extraction.obligations}

    assert by_type[ObligationType.RENEWAL_OPTION_NOTICE].normalized_value == "P90D"

    termination = by_type.get(ObligationType.TERMINATION_NOTICE_PERIOD)
    assert termination is None or termination.normalized_value != "P90D"


def test_the_lease_deadline_is_the_option_deadline(lease_extraction):
    by_type = {o.obligation_type: o for o in lease_extraction.obligations}

    deadline = by_type[ObligationType.RENEWAL_OPTION_DEADLINE]
    assert deadline.status is ObligationStatus.COMPUTED
    assert deadline.normalized_value == "2030-11-02"  # 2031-01-31 minus 90 days
    assert deadline.computation_inputs["consequence"] == (
        "the renewal option lapses after this date"
    )


def test_an_optional_renewal_is_not_an_automatic_renewal(lease_extraction):
    by_type = {o.obligation_type: o for o in lease_extraction.obligations}
    renewal = by_type.get(ObligationType.AUTOMATIC_RENEWAL)

    if renewal is not None:
        assert renewal.normalized_value == "false"


def test_the_material_lease_clauses_reach_the_model(tmp_path: Path):
    """Retrieval coverage on a 19-page form, BM25 only."""
    settings = Settings(rag_data_dir=tmp_path / "data", use_fake_llm=True)
    clear_index_cache()
    outcome = ingest_path(settings, LEASE)
    index = get_document_index(settings, outcome.document_id)

    context = retrieve_context(index, DEFAULT_OBLIGATION_TYPES, top_k=6)
    joined = "\n".join(chunk.text for chunk in context)

    for needle in (
        "commence on February 1, 2026",
        "January 31, 2031",
        "not less than 90 days prior",
        "$8,500, payable",
        "3% each renewal term",
        "$1,000,000 per occurrence",
        "laws of the State of Texas",
    ):
        assert needle in joined, f"{needle!r} never reached the model"
    clear_index_cache()
