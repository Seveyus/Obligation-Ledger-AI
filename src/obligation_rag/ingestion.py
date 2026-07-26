"""Ingestion pipeline: parse -> chunk -> persist -> index.

Shared by the HTTP endpoint and the CLI scripts so both produce byte-identical
state — a document ingested from the terminal is immediately servable by the
API, and vice versa.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from . import storage
from .chunking import chunk_document
from .config import Settings
from .embeddings import get_embedding_backend
from .pdf_parser import DocumentParseError, parse_document
from .retrieval import build_index, cache_index, invalidate_index

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".text"}


class UnsupportedDocumentError(DocumentParseError):
    """The file is not something this service can read."""


@dataclass(slots=True)
class IngestionOutcome:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    retrieval_mode: str
    stored_path: Path


def ingest_bytes(
    settings: Settings,
    payload: bytes,
    filename: str,
    *,
    document_id: str | None = None,
) -> IngestionOutcome:
    """Ingest an in-memory document. Raises ``DocumentParseError`` on bad input."""
    filename = Path(filename or "contract.pdf").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocumentError(
            f"unsupported_file_type: {suffix or '(none)'}; expected one of "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )
    if not payload:
        raise UnsupportedDocumentError("empty_file")

    digest = hashlib.sha256(payload).hexdigest()
    # Content-addressed by default: re-uploading the same contract is idempotent.
    resolved_id = document_id or f"contract_{digest[:12]}"

    settings.ensure_directories()
    storage.init_db(settings)
    stored_path = settings.uploads_dir / f"{resolved_id}{suffix}"
    stored_path.write_bytes(payload)

    try:
        parsed = parse_document(stored_path, strip_furniture=settings.strip_page_furniture)
    except DocumentParseError:
        stored_path.unlink(missing_ok=True)
        raise

    chunks = chunk_document(
        resolved_id,
        parsed,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )

    storage.insert_document(
        settings,
        document_id=resolved_id,
        filename=filename,
        stored_path=stored_path,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        sha256=digest,
    )
    storage.insert_pages(settings, resolved_id, parsed.page_map())
    storage.insert_chunks(settings, chunks)

    backend = get_embedding_backend(settings)
    embeddings = None
    if backend is not None and chunks:
        try:
            embeddings = backend.encode([chunk.text for chunk in chunks])
            storage.save_embeddings(settings, resolved_id, embeddings)
        except Exception as error:  # noqa: BLE001 - the dense path is optional
            logger.warning("embedding failed (%s); document stays BM25-only", error)
            embeddings = None

    invalidate_index(resolved_id)
    index = build_index(
        resolved_id,
        chunks,
        settings=settings,
        embeddings=embeddings,
        backend=backend if embeddings is not None else None,
    )
    cache_index(index)

    return IngestionOutcome(
        document_id=resolved_id,
        filename=filename,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        retrieval_mode=index.retrieval_mode,
        stored_path=stored_path,
    )


def ingest_path(
    settings: Settings, path: str | Path, *, document_id: str | None = None
) -> IngestionOutcome:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return ingest_bytes(settings, path.read_bytes(), path.name, document_id=document_id)
