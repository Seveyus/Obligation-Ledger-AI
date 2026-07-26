#!/usr/bin/env python3
"""Ingest a contract into the local RAG store, without going through HTTP.

    python scripts/ingest_pdf.py data/samples/sample_contract.txt
    python scripts/ingest_pdf.py /path/to/contract.pdf --document-id contract_123

Writes to the same `RAG_DATA_DIR` the API uses, so anything ingested here is
immediately servable by `GET /v1/documents`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # run without installing the package
    sys.path.insert(0, str(REPO_ROOT / "src"))

from obligation_rag.config import get_settings  # noqa: E402
from obligation_rag.ingestion import ingest_path  # noqa: E402
from obligation_rag.pdf_parser import DocumentParseError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="PDF (or .txt fixture) to ingest")
    parser.add_argument("--document-id", default=None, help="override the generated document id")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()

    settings = get_settings()
    try:
        outcome = ingest_path(settings, args.path, document_id=args.document_id)
    except FileNotFoundError:
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 2
    except DocumentParseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "document_id": outcome.document_id,
                    "filename": outcome.filename,
                    "page_count": outcome.page_count,
                    "chunk_count": outcome.chunk_count,
                    "retrieval_mode": outcome.retrieval_mode,
                },
                indent=2,
            )
        )
    else:
        print(f"document_id     {outcome.document_id}")
        print(f"filename        {outcome.filename}")
        print(f"pages           {outcome.page_count}")
        print(f"chunks          {outcome.chunk_count}")
        print(f"retrieval       {outcome.retrieval_mode}")
        print(f"data dir        {settings.rag_data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
