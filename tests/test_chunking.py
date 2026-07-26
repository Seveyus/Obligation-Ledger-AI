"""Chunks carry the page number a citation will be checked against."""

from __future__ import annotations

import pytest

from obligation_rag.chunking import chunk_document, chunk_page
from obligation_rag.pdf_parser import ParsedDocument, ParsedPage


def test_every_chunk_belongs_to_exactly_one_page(settings, parsed_contract):
    chunks = chunk_document("doc", parsed_contract, chunk_size=400, overlap=80)

    pages = {page.page for page in parsed_contract.pages}
    assert {chunk.page for chunk in chunks} <= pages
    for chunk in chunks:
        page_text = parsed_contract.page_map()[chunk.page]
        assert chunk.text in page_text, "a chunk must be verbatim page content"


def test_offsets_address_the_original_page_text(parsed_contract):
    chunks = chunk_document("doc", parsed_contract, chunk_size=500, overlap=100)
    pages = parsed_contract.page_map()

    for chunk in chunks:
        assert pages[chunk.page][chunk.start_offset : chunk.end_offset].strip() == chunk.text


def test_chunk_ids_are_unique_and_document_wide(parsed_contract):
    chunks = chunk_document("doc", parsed_contract, chunk_size=350, overlap=50)

    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "chunk_0"


def test_a_long_page_is_split_with_overlap():
    clause = "The Provider shall deliver the Services in accordance with the Service Levels. "
    page = ParsedPage(page=1, text="\n\n".join(clause * 3 for _ in range(8)))

    chunks = chunk_page("doc", 1, page.text, chunk_size=400, overlap=120, start_index=0)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 600 for chunk in chunks)
    assert chunks[1].start_offset < chunks[0].end_offset, "consecutive chunks must overlap"


def test_a_single_oversized_sentence_is_not_dropped():
    page_text = "x" * 3000

    chunks = chunk_page("doc", 1, page_text, chunk_size=1000, overlap=100, start_index=0)

    assert chunks
    assert sum(len(chunk.text) for chunk in chunks) >= 3000


def test_empty_page_yields_no_chunks():
    assert chunk_page("doc", 1, "   \n\n  ", chunk_size=400, overlap=50, start_index=0) == []


def test_chunk_numbering_continues_across_pages():
    document = ParsedDocument(
        filename="c.txt",
        pages=[ParsedPage(page=1, text="First page clause."), ParsedPage(page=2, text="Second.")],
    )

    chunks = chunk_document("doc", document, chunk_size=200, overlap=20)

    assert [chunk.chunk_id for chunk in chunks] == ["chunk_0", "chunk_1"]
    assert [chunk.page for chunk in chunks] == [1, 2]


def test_overlap_must_be_smaller_than_the_chunk(parsed_contract):
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("doc", parsed_contract, chunk_size=100, overlap=100)
