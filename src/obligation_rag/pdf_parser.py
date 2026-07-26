"""PDF (and plain-text fixture) parsing with page numbers preserved.

Page numbers are 1-based and follow the document, not the array index: a quote
reported on page 9 must be findable on the page a human sees as page 9.

The cleaned text produced here is the canonical text of the document. It is
what gets chunked, what gets indexed, and what evidence verification compares
against — so the same string is used everywhere and offsets stay meaningful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:  # PyMuPDF >= 1.24 ships the `pymupdf` name; older builds only have `fitz`.
    import pymupdf as fitz
except ImportError:  # pragma: no cover - depends on the installed wheel
    import fitz

#: Plain-text fixtures separate pages with a form feed, like `pdftotext` does.
PAGE_SEPARATOR = "\f"

_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class DocumentParseError(RuntimeError):
    """Raised when a document yields no usable text (e.g. a scanned PDF)."""


@dataclass(slots=True)
class ParsedPage:
    page: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class ParsedDocument:
    filename: str
    pages: list[ParsedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(page.char_count for page in self.pages)

    def page_map(self) -> dict[int, str]:
        return {page.page: page.text for page in self.pages}


def clean_page_text(raw: str) -> str:
    """Repair the usual PDF extraction artifacts without moving content.

    Line-wrapped hyphenated words are re-joined ("termi-\\nnation" -> "termination")
    because neither the model nor a human quotes the hyphen.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _TRAILING_SPACES.sub("\n", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


#: How much of a page's head/tail is considered running furniture.
_FURNITURE_ZONE_LINES = 3
#: Share of pages a line must appear on before it counts as furniture.
_FURNITURE_PAGE_RATIO = 0.6
_FURNITURE_MAX_LINE_CHARS = 120


def strip_page_furniture(pages: list[str]) -> list[str]:
    """Remove running headers and footers repeated across the document.

    Filled contract templates repeat a header on every page ("INITIAL ____ DATE
    ____ / Commercial Lease Agreement (Rev. 1343D17)"). It lands at the top of
    every chunk, wastes model context and turns up inside quotes.

    Only identical lines, only in the head/tail zone of a page, and only when
    they appear on most pages — a clause is never removed, because a clause
    does not repeat verbatim across two thirds of a contract.
    """
    if len(pages) < 3:
        return pages

    counts: dict[str, int] = {}
    for page in pages:
        lines = page.splitlines()
        zone = lines[:_FURNITURE_ZONE_LINES] + lines[-_FURNITURE_ZONE_LINES:]
        for candidate in {line.strip() for line in zone}:
            if candidate and len(candidate) <= _FURNITURE_MAX_LINE_CHARS:
                counts[candidate] = counts.get(candidate, 0) + 1

    threshold = max(3, int(len(pages) * _FURNITURE_PAGE_RATIO))
    furniture = {line for line, count in counts.items() if count >= threshold}
    if not furniture:
        return pages

    cleaned: list[str] = []
    for page in pages:
        lines = page.splitlines()
        head_end = min(_FURNITURE_ZONE_LINES, len(lines))
        tail_start = max(0, len(lines) - _FURNITURE_ZONE_LINES)
        kept = [
            line
            for position, line in enumerate(lines)
            if not ((position < head_end or position >= tail_start) and line.strip() in furniture)
        ]
        cleaned.append("\n".join(kept).strip())
    return cleaned


def _build(path: Path, blocks: list[str], *, strip_furniture: bool) -> ParsedDocument:
    texts = [clean_page_text(block) for block in blocks]
    if strip_furniture:
        texts = strip_page_furniture(texts)
    pages = [ParsedPage(page=index, text=text) for index, text in enumerate(texts, start=1)]
    _reject_empty(path, pages)
    return ParsedDocument(filename=path.name, pages=pages)


def parse_pdf(path: str | Path, *, strip_furniture: bool = True) -> ParsedDocument:
    """Extract text page by page from a PDF."""
    path = Path(path)
    with fitz.open(path) as document:
        blocks = [page.get_text("text") for page in document]
    return _build(path, blocks, strip_furniture=strip_furniture)


def parse_text_file(path: str | Path, *, strip_furniture: bool = True) -> ParsedDocument:
    """Parse a plain-text fixture, splitting pages on form feeds."""
    path = Path(path)
    blocks = path.read_text(encoding="utf-8").split(PAGE_SEPARATOR)
    return _build(path, blocks, strip_furniture=strip_furniture)


def parse_document(path: str | Path, *, strip_furniture: bool = True) -> ParsedDocument:
    """Dispatch on file extension. PDFs are the product; text files are fixtures."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, strip_furniture=strip_furniture)
    if suffix in {".txt", ".md", ".text"}:
        return parse_text_file(path, strip_furniture=strip_furniture)
    raise DocumentParseError(f"unsupported_file_type: {suffix or '(none)'}")


def _reject_empty(path: Path, pages: list[ParsedPage]) -> None:
    """A document with no extractable text needs OCR, which is out of scope."""
    if not pages or all(not page.text.strip() for page in pages):
        raise DocumentParseError(
            f"no_extractable_text: {path.name} contains no selectable text "
            "(a scanned document would need OCR, which this service does not do)"
        )
