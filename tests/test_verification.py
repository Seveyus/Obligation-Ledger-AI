"""The trust boundary: a quote is real, or the value does not ship."""

from __future__ import annotations

from obligation_rag.schemas import VerificationMethod
from obligation_rag.verification import (
    normalize_text,
    normalize_with_map,
    verify_evidence,
    verify_quote_on_page,
)

PAGE = (
    "3.2 Notice of Non-Renewal. Either party may elect not to renew this Agreement by\n"
    "delivering written notice of termination to the other party not less than sixty (60)\n"
    "days prior to the end of the then-current term."
)


def test_normalization_folds_pdf_punctuation():
    assert normalize_text("“sixty (60) days’ notice”") == '"sixty (60) days\' notice"'
    assert normalize_text("A B") == "a b"
    assert normalize_text("soft­hyphen") == "softhyphen"
    assert normalize_text("reﬂect") == "reflect"  # fi/fl ligature
    assert normalize_text("  spaced   out \n text ") == "spaced out text"


def test_offset_map_points_back_at_the_original_string():
    original = "The  “notice”\nperiod"
    normalized, offsets = normalize_with_map(original)

    assert len(normalized) == len(offsets)
    index = normalized.index("notice")
    assert original[offsets[index] : offsets[index] + len("notice")] == "notice"


def test_exact_quote_verifies_and_reports_the_span():
    outcome = verify_quote_on_page("written notice of termination", PAGE, page=3)

    assert outcome.verified
    assert outcome.method is VerificationMethod.NORMALIZED_EXACT_MATCH
    assert PAGE[outcome.start_offset : outcome.end_offset] == "written notice of termination"


def test_quote_spanning_a_line_break_still_verifies():
    """The model quotes prose; the PDF has hard wraps. Whitespace must not matter."""
    outcome = verify_quote_on_page(
        "not less than sixty (60) days prior to the end of the then-current term", PAGE, page=3
    )

    assert outcome.verified
    assert outcome.method is VerificationMethod.NORMALIZED_EXACT_MATCH


def test_smart_quotes_and_casing_do_not_break_verification():
    page = "The Agreement is “governed by the laws of the State of Delaware”."

    outcome = verify_quote_on_page("governed by the laws of the State of DELAWARE", page, page=6)

    assert outcome.verified


def test_invented_quote_fails():
    outcome = verify_quote_on_page(
        "This Agreement shall remain in force until December 31, 2029.", PAGE, page=3
    )

    assert not outcome.verified
    assert outcome.method is VerificationMethod.NONE
    assert "quote_not_found_on_page" in outcome.reason


def test_bounded_fuzzy_fallback_absorbs_extraction_artifacts():
    """One mangled character is a PDF artifact; a different sentence is not."""
    outcome = verify_quote_on_page(
        "delivering written notlce of termination to the other party", PAGE, page=3
    )

    assert outcome.verified
    assert outcome.method is VerificationMethod.FUZZY_MATCH
    assert outcome.similarity >= 0.92


def test_fuzzy_fallback_does_not_rescue_a_rewritten_clause():
    outcome = verify_quote_on_page(
        "either party may cancel at any time by giving thirty days notice", PAGE, page=3
    )

    assert not outcome.verified


def test_quote_too_short_is_refused():
    outcome = verify_quote_on_page("notice", PAGE, page=3)

    assert not outcome.verified
    assert "quote_too_short" in outcome.reason


def test_right_quote_on_the_wrong_page_fails_but_names_the_real_page():
    pages = {1: "Cover page.", 3: PAGE}

    outcome = verify_evidence("written notice of termination", pages, claimed_page=1)

    assert not outcome.verified
    assert "wrong_page" in outcome.reason
    assert "page 3" in outcome.reason


def test_page_out_of_range_is_reported_as_such():
    outcome = verify_evidence("written notice of termination", {1: PAGE}, claimed_page=9)

    assert not outcome.verified
    assert "page_out_of_range" in outcome.reason


def test_document_wide_search_when_no_page_is_claimed(contract_pages):
    outcome = verify_evidence("automatically\nrenew for successive renewal terms", contract_pages)

    assert outcome.verified
    assert outcome.page == 2


def test_verification_against_the_real_fixture(contract_pages):
    outcome = verify_evidence(
        "the annual subscription fee\nshall increase by four percent (4%)",
        contract_pages,
        claimed_page=4,
    )

    assert outcome.verified
    assert outcome.method is VerificationMethod.NORMALIZED_EXACT_MATCH
    page_text = contract_pages[4]
    assert "four percent (4%)" in page_text[outcome.start_offset : outcome.end_offset]
