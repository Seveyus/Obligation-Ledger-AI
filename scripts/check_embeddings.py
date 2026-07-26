#!/usr/bin/env python3
"""Check that dense retrieval is actually loaded, and that it actually helps.

    python scripts/check_embeddings.py

Two things go wrong silently when enabling embeddings, and neither raises:

1. the model fails to load and the service falls back to BM25-only
2. the model loads but the instruction prefix is wrong, so retrieval quality
   quietly drops instead of improving

So this measures rather than reports. It runs the same natural-language
questions twice — BM25-only, then with whatever backend is configured — and
prints both scores. If the second is not better, the setup is not right.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from obligation_rag.config import Settings, get_settings  # noqa: E402
from obligation_rag.embeddings import get_embedding_backend  # noqa: E402
from obligation_rag.ingestion import ingest_path  # noqa: E402
from obligation_rag.retrieval import clear_index_cache, get_document_index  # noqa: E402

LEASE = REPO_ROOT / "data" / "samples" / "sample_lease_form.txt"

#: Questions phrased the way a user would, with the page that answers them.
#: Lexical search fails several of these — that is the whole point of the
#: dense side, and the only reason to pay for a model.
ASK_BENCHMARK: list[tuple[str, int]] = [
    ("how much do I pay every month?", 3),
    ("when do I have to tell the landlord I want to stay?", 3),
    ("what happens if I pay late?", 5),
    ("can I sublet the office?", 14),
    ("who is responsible for fixing the roof?", 10),
    ("what is my deposit?", 6),
]


def _score(settings: Settings, document: Path, label: str) -> int:
    with tempfile.TemporaryDirectory(prefix="obligation-rag-emb-") as scratch:
        scoped = settings.model_copy(update={"rag_data_dir": Path(scratch)})
        clear_index_cache()
        outcome = ingest_path(scoped, document)
        index = get_document_index(scoped, outcome.document_id)
        if index is None:
            print(f"  {label}: ingestion produced no index")
            return 0

        print(f"\n{label}  ({index.retrieval_mode})")
        hits = 0
        for question, expected in ASK_BENCHMARK:
            pages = [chunk.page for chunk in index.search(question, top_k=3)]
            found = expected in pages
            hits += found
            print(
                f"  {'HIT ' if found else 'MISS'} p.{expected:<3} got={str(pages):<14} {question}"
            )
        clear_index_cache()
        return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=LEASE)
    args = parser.parse_args()

    settings = get_settings()
    backend = get_embedding_backend(settings)

    print("configured EMBEDDING_MODEL_PATH :", settings.embedding_model_path or "(empty)")
    print("query prefix                    :", repr(settings.embedding_query_prefix))
    print("document prefix                 :", repr(settings.embedding_document_prefix))

    if backend is None:
        print("\nbackend: NONE — retrieval is BM25-only.")
        print("Either EMBEDDING_MODEL_PATH is unset, or the model failed to load;")
        print("the service logs a warning and falls back rather than failing.")
        return 1

    print(f"\nbackend: {backend.name}, {backend.dimension} dimensions")

    if backend.name == "hashing":
        print(
            "\nThis is the deterministic test backend (USE_FAKE_EMBEDDINGS=true).\n"
            "It hashes character n-grams and understands nothing, so it exercises the\n"
            "hybrid code path but will usually score WORSE than BM25 alone. It is not\n"
            "a way to skip downloading a real encoder."
        )

    baseline = _score(
        settings.model_copy(update={"embedding_model_path": None, "use_fake_embeddings": False}),
        args.path,
        "BM25 only (baseline)",
    )
    hybrid = _score(settings, args.path, "hybrid (BM25 + vectors, RRF)")

    if args.path != LEASE:
        print("\n(scores are only meaningful on the bundled lease fixture)")
        return 0

    print(
        f"\nAsk benchmark: BM25 {baseline}/{len(ASK_BENCHMARK)}  ->  "
        f"hybrid {hybrid}/{len(ASK_BENCHMARK)}"
    )
    if hybrid > baseline:
        print("Dense retrieval is helping. Setup looks right.")
        return 0
    if backend.name == "hashing":
        print("Expected with the test backend — see the note above.")
        return 0
    print(
        "No improvement. Check the instruction prefix on the model card — a wrong\n"
        "or missing prefix costs most of what these encoders are worth."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
