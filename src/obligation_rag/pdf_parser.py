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


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Extract text page by page from a PDF."""
    path = Path(path)
    pages: list[ParsedPage] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            pages.append(ParsedPage(page=index, text=clean_page_text(page.get_text("text"))))
    _reject_empty(path, pages)
    return ParsedDocument(filename=path.name, pages=pages)


def parse_text_file(path: str | Path) -> ParsedDocument:
    """Parse a plain-text fixture, splitting pages on form feeds."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    pages = [
        ParsedPage(page=index, text=clean_page_text(block))
        for index, block in enumerate(raw.split(PAGE_SEPARATOR), start=1)
    ]
    _reject_empty(path, pages)
    return ParsedDocument(filename=path.name, pages=pages)


def parse_document(path: str | Path) -> ParsedDocument:
    """Dispatch on file extension. PDFs are the product; text files are fixtures."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".txt", ".md", ".text"}:
        return parse_text_file(path)
    raise DocumentParseError(f"unsupported_file_type: {suffix or '(none)'}")


def _reject_empty(path: Path, pages: list[ParsedPage]) -> None:
    """A document with no extractable text needs OCR, which is out of scope."""
    if not pages or all(not page.text.strip() for page in pages):
        raise DocumentParseError(
            f"no_extractable_text: {path.name} contains no selectable text "
            "(a scanned document would need OCR, which this service does not do)"
        )
