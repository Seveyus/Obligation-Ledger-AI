"""Page numbers are load-bearing: a citation is worthless if the page is wrong."""

from __future__ import annotations

from pathlib import Path

import pytest

from obligation_rag.pdf_parser import (
    DocumentParseError,
    clean_page_text,
    parse_document,
    parse_pdf,
    parse_text_file,
)

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


def _write_pdf(path: Path, pages: list[str]) -> Path:
    document = fitz.open()
    for body in pages:
        page = document.new_page()
        page.insert_text((72, 100), body, fontsize=11)
    document.save(path)
    document.close()
    return path


def test_text_fixture_keeps_one_based_page_numbers(sample_contract_path: Path):
    parsed = parse_text_file(sample_contract_path)

    assert parsed.page_count == 6
    assert [page.page for page in parsed.pages] == [1, 2, 3, 4, 5, 6]
    assert parsed.pages[0].page == 1, "pages are 1-based, not indexes"


def test_clauses_land_on_the_page_a_human_would_turn_to(sample_contract_path: Path):
    pages = parse_text_file(sample_contract_path).page_map()

    assert "Notice of Non-Renewal" in pages[3]
    assert "sixty (60) days" in pages[3]
    assert "sixty (60) days" not in pages[1]
    assert "Governing Law" in pages[6]


def test_pdf_extraction_preserves_page_boundaries(tmp_path: Path):
    path = _write_pdf(
        tmp_path / "contract.pdf",
        ["Page one: the Effective Date is March 1, 2024.", "Page two: sixty (60) days notice."],
    )

    parsed = parse_pdf(path)

    assert parsed.page_count == 2
    assert "Effective Date" in parsed.pages[0].text
    assert "sixty (60) days" in parsed.pages[1].text
    assert "sixty" not in parsed.pages[0].text


def test_dispatch_by_extension(tmp_path: Path, sample_contract_path: Path):
    pdf = _write_pdf(tmp_path / "c.pdf", ["Hello contract."])

    assert parse_document(pdf).page_count == 1
    assert parse_document(sample_contract_path).page_count == 6


def test_hyphenated_line_breaks_are_rejoined():
    cleaned = clean_page_text("the party may termi-\nnate this Agreement")

    assert "terminate this Agreement" in cleaned
    assert "termi-" not in cleaned


def test_excess_blank_lines_collapse_but_paragraphs_survive():
    cleaned = clean_page_text("clause one\n\n\n\nclause two")

    assert cleaned == "clause one\n\nclause two"


def test_unsupported_file_type_is_refused(tmp_path: Path):
    path = tmp_path / "contract.docx"
    path.write_bytes(b"not a pdf")

    with pytest.raises(DocumentParseError, match="unsupported_file_type"):
        parse_document(path)


def test_scanned_pdf_without_text_layer_is_refused(tmp_path: Path):
    """OCR is out of scope; failing loudly beats silently indexing nothing."""
    path = _write_pdf(tmp_path / "scan.pdf", ["", ""])

    with pytest.raises(DocumentParseError, match="no_extractable_text"):
        parse_pdf(path)
