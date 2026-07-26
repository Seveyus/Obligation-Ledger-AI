"""Page-aware chunking.

A chunk never spans two pages. That is the whole point: every chunk carries an
unambiguous page number, so an obligation extracted from a chunk can cite a
page a human can turn to.

Chunks are packed out of paragraph/sentence segments so a clause is not cut in
the middle, with a character overlap between consecutive chunks so a clause
sitting on a boundary is still retrievable in one piece.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_parser import ParsedDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z0-9(\"'])")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    page: int
    index_in_page: int
    text: str
    start_offset: int
    end_offset: int


def _segments(page_text: str) -> list[str]:
    """Split a page into paragraph-sized pieces, then sentence-sized if needed."""
    segments: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(page_text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        segments.append(paragraph)
    return segments


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    if len(segment) <= max_chars:
        return [segment]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(segment):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    # A single sentence longer than the budget (tables, address blocks) is cut
    # on the character grid rather than dropped.
    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final.append(piece)
            continue
        for start in range(0, len(piece), max_chars):
            final.append(piece[start : start + max_chars])
    return final


def chunk_page(
    document_id: str,
    page: int,
    page_text: str,
    *,
    chunk_size: int,
    overlap: int,
    start_index: int,
) -> list[Chunk]:
    """Chunk a single page, tracking offsets into ``page_text``."""
    chunks: list[Chunk] = []
    if not page_text.strip():
        return chunks

    pieces: list[str] = []
    for segment in _segments(page_text):
        pieces.extend(_split_long_segment(segment, chunk_size))

    # Offsets of each piece inside the original page text.
    cursor = 0
    located: list[tuple[str, int]] = []
    for piece in pieces:
        position = page_text.find(piece, cursor)
        if position == -1:  # defensive: whitespace normalisation edge case
            position = cursor
        located.append((piece, position))
        cursor = position + len(piece)

    # Pack pieces into non-overlapping spans first...
    spans: list[tuple[int, int]] = []
    span_start: int | None = None
    span_end = 0
    for piece, position in located:
        if span_start is not None and (position + len(piece)) - span_start > chunk_size:
            spans.append((span_start, span_end))
            span_start = None
        if span_start is None:
            span_start = position
        span_end = position + len(piece)
    if span_start is not None:
        spans.append((span_start, span_end))

    # ...then widen each span backwards for the overlap, so a clause sitting on
    # a boundary stays retrievable in one piece.
    for index_in_page, (start, end) in enumerate(spans):
        if index_in_page and overlap:
            floor = spans[index_in_page - 1][0] + 1
            widened = max(floor, start - overlap)
            boundary = page_text.find(" ", widened, start)
            start = boundary + 1 if boundary != -1 else widened

        while start < end and page_text[start].isspace():
            start += 1
        while end > start and page_text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"chunk_{start_index + len(chunks)}",
                document_id=document_id,
                page=page,
                index_in_page=index_in_page,
                text=page_text[start:end],
                start_offset=start,
                end_offset=end,
            )
        )

    return chunks


def chunk_document(
    document_id: str,
    document: ParsedDocument,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk every page of ``document``, numbering chunks across the document."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in document.pages:
        chunks.extend(
            chunk_page(
                document_id,
                page.page,
                page.text,
                chunk_size=chunk_size,
                overlap=overlap,
                start_index=len(chunks),
            )
        )
    return chunks
