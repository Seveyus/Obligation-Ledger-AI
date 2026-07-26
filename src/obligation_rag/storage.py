"""SQLite persistence for documents, pages, chunks and candidate obligations.

Deliberately plain ``sqlite3``: this component ships inside a default-deny
sandbox, so every dependency has to earn its place. Connections are opened per
call because FastAPI runs sync endpoints in a thread pool.

Note the split: text lives in SQLite, vectors live next to it as ``.npy``
files under ``RAG_DATA_DIR/indexes``. Neither is the ledger — the committed,
hash-chained register is Aditya's side of the wall.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .config import Settings
from .schemas import (
    CandidateObligation,
    DocumentRecord,
    ObligationStatus,
    ObligationType,
    SourceEvidence,
    VerificationMethod,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    page_count   INTEGER NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    uploaded_at  TEXT NOT NULL,
    sha256       TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    document_id  TEXT NOT NULL,
    page         INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (document_id, page),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT NOT NULL,
    document_id   TEXT NOT NULL,
    page          INTEGER NOT NULL,
    index_in_page INTEGER NOT NULL,
    text          TEXT NOT NULL,
    start_offset  INTEGER NOT NULL,
    end_offset    INTEGER NOT NULL,
    doc_start_offset INTEGER,
    doc_end_offset   INTEGER,
    PRIMARY KEY (document_id, chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS obligations (
    id                  TEXT PRIMARY KEY,
    document_id         TEXT NOT NULL,
    obligation_type     TEXT NOT NULL,
    raw_value           TEXT NOT NULL,
    normalized_value    TEXT,
    quote               TEXT,
    page                INTEGER,
    chunk_id            TEXT,
    start_offset        INTEGER,
    end_offset          INTEGER,
    status              TEXT NOT NULL,
    verification_method TEXT NOT NULL,
    verification_reason TEXT,
    computation_formula TEXT,
    computation_inputs  TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_obligations_document ON obligations(document_id);
"""


@contextmanager
def connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    settings.ensure_directories()
    connection = sqlite3.connect(settings.db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


#: Columns added after the first release. `CREATE TABLE IF NOT EXISTS` cannot
#: add them to a database that already exists, so they are applied by hand.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("chunks", "doc_start_offset", "INTEGER"),
    ("chunks", "doc_end_offset", "INTEGER"),
)


def init_db(settings: Settings) -> None:
    with connect(settings) as connection:
        connection.executescript(SCHEMA)
        for table, column, column_type in _MIGRATIONS:
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def insert_document(
    settings: Settings,
    *,
    document_id: str,
    filename: str,
    stored_path: Path,
    page_count: int,
    chunk_count: int,
    sha256: str | None = None,
    uploaded_at: datetime | None = None,
) -> DocumentRecord:
    timestamp = uploaded_at or datetime.now().astimezone()
    with connect(settings) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO documents
                (id, filename, stored_path, page_count, chunk_count, uploaded_at, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                str(stored_path),
                page_count,
                chunk_count,
                timestamp.isoformat(),
                sha256,
            ),
        )
    return DocumentRecord(
        id=document_id,
        filename=filename,
        page_count=page_count,
        chunk_count=chunk_count,
        uploaded_at=timestamp,
        sha256=sha256,
    )


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        filename=row["filename"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        sha256=row["sha256"],
    )


def get_document(settings: Settings, document_id: str) -> DocumentRecord | None:
    with connect(settings) as connection:
        row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _row_to_document(row) if row else None


def get_document_path(settings: Settings, document_id: str) -> Path | None:
    with connect(settings) as connection:
        row = connection.execute(
            "SELECT stored_path FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return Path(row["stored_path"]) if row else None


def list_documents(settings: Settings) -> list[DocumentRecord]:
    with connect(settings) as connection:
        rows = connection.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return [_row_to_document(row) for row in rows]


def count_documents(settings: Settings) -> int:
    with connect(settings) as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM documents").fetchone()
    return int(row["total"])


def delete_document(settings: Settings, document_id: str) -> None:
    with connect(settings) as connection:
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    embeddings_path = settings.indexes_dir / f"{document_id}.npy"
    embeddings_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Pages and chunks
# --------------------------------------------------------------------------


def insert_pages(settings: Settings, document_id: str, pages: dict[int, str]) -> None:
    with connect(settings) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO pages (document_id, page, text) VALUES (?, ?, ?)",
            [(document_id, page, text) for page, text in sorted(pages.items())],
        )


def get_pages(settings: Settings, document_id: str) -> dict[int, str]:
    with connect(settings) as connection:
        rows = connection.execute(
            "SELECT page, text FROM pages WHERE document_id = ? ORDER BY page", (document_id,)
        ).fetchall()
    return {int(row["page"]): row["text"] for row in rows}


def insert_chunks(settings: Settings, chunks: Iterable[Chunk]) -> None:
    with connect(settings) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, page, index_in_page, text, start_offset,
                 end_offset, doc_start_offset, doc_end_offset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.page,
                    chunk.index_in_page,
                    chunk.text,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.doc_start_offset,
                    chunk.doc_end_offset,
                )
                for chunk in chunks
            ],
        )


def get_chunks(settings: Settings, document_id: str) -> list[Chunk]:
    with connect(settings) as connection:
        rows = connection.execute(
            """
            SELECT * FROM chunks WHERE document_id = ?
            ORDER BY page, index_in_page
            """,
            (document_id,),
        ).fetchall()
    return [
        Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            page=int(row["page"]),
            index_in_page=int(row["index_in_page"]),
            text=row["text"],
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            doc_start_offset=row["doc_start_offset"],
            doc_end_offset=row["doc_end_offset"],
        )
        for row in rows
    ]


def get_all_chunks(settings: Settings) -> list[Chunk]:
    """Every chunk of every document, for corpus-wide retrieval."""
    with connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM chunks ORDER BY document_id, page, index_in_page"
        ).fetchall()
    return [
        Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            page=int(row["page"]),
            index_in_page=int(row["index_in_page"]),
            text=row["text"],
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            doc_start_offset=row["doc_start_offset"],
            doc_end_offset=row["doc_end_offset"],
        )
        for row in rows
    ]


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------


def embeddings_path(settings: Settings, document_id: str) -> Path:
    return settings.indexes_dir / f"{document_id}.npy"


def save_embeddings(settings: Settings, document_id: str, matrix: np.ndarray) -> None:
    settings.ensure_directories()
    np.save(embeddings_path(settings, document_id), matrix.astype(np.float32))


def load_embeddings(settings: Settings, document_id: str) -> np.ndarray | None:
    path = embeddings_path(settings, document_id)
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (ValueError, OSError):  # corrupted index: fall back to BM25-only
        return None


# --------------------------------------------------------------------------
# Candidate obligations (latest extraction run per document)
# --------------------------------------------------------------------------


def save_obligations(
    settings: Settings, document_id: str, obligations: Iterable[CandidateObligation]
) -> None:
    now = datetime.now().astimezone().isoformat()
    with connect(settings) as connection:
        connection.execute("DELETE FROM obligations WHERE document_id = ?", (document_id,))
        connection.executemany(
            """
            INSERT INTO obligations (
                id, document_id, obligation_type, raw_value, normalized_value,
                quote, page, chunk_id, start_offset, end_offset,
                status, verification_method, verification_reason,
                computation_formula, computation_inputs, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    obligation.id,
                    obligation.document_id,
                    obligation.obligation_type.value,
                    obligation.raw_value,
                    obligation.normalized_value,
                    obligation.source_evidence.quote if obligation.source_evidence else None,
                    obligation.source_evidence.page if obligation.source_evidence else None,
                    obligation.source_evidence.chunk_id if obligation.source_evidence else None,
                    (
                        obligation.source_evidence.start_offset
                        if obligation.source_evidence
                        else None
                    ),
                    obligation.source_evidence.end_offset if obligation.source_evidence else None,
                    obligation.status.value,
                    obligation.verification_method.value,
                    obligation.verification_reason,
                    obligation.computation_formula,
                    (
                        json.dumps(obligation.computation_inputs)
                        if obligation.computation_inputs
                        else None
                    ),
                    now,
                )
                for obligation in obligations
            ],
        )


def get_obligations(settings: Settings, document_id: str) -> list[CandidateObligation]:
    with connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM obligations WHERE document_id = ? ORDER BY rowid", (document_id,)
        ).fetchall()

    obligations: list[CandidateObligation] = []
    for row in rows:
        evidence = None
        if row["quote"]:
            evidence = SourceEvidence(
                quote=row["quote"],
                page=int(row["page"]) if row["page"] is not None else 0,
                chunk_id=row["chunk_id"],
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
            )
        obligations.append(
            CandidateObligation(
                id=row["id"],
                document_id=row["document_id"],
                obligation_type=ObligationType(row["obligation_type"]),
                raw_value=row["raw_value"],
                normalized_value=row["normalized_value"],
                source_evidence=evidence,
                status=ObligationStatus(row["status"]),
                verification_method=VerificationMethod(row["verification_method"]),
                verification_reason=row["verification_reason"],
                computation_formula=row["computation_formula"],
                computation_inputs=(
                    json.loads(row["computation_inputs"]) if row["computation_inputs"] else None
                ),
            )
        )
    return obligations
