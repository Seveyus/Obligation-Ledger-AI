#!/usr/bin/env python3
"""Run the whole pipeline on one document and print a review-style table.

    python scripts/test_pipeline.py                      # fixture + fake LLM
    python scripts/test_pipeline.py --ablation           # spec §10.1 demo
    python scripts/test_pipeline.py contract.pdf --real  # real gpt-oss-120b

The `--ablation` run swaps in a model that invents a plausible date. With
verification on, it is caught and approval is blocked — which is the whole
argument of the product.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # run without installing the package
    sys.path.insert(0, str(REPO_ROOT / "src"))

from obligation_rag import storage  # noqa: E402
from obligation_rag.config import Settings, get_settings  # noqa: E402
from obligation_rag.extraction import run_extraction  # noqa: E402
from obligation_rag.ingestion import ingest_path  # noqa: E402
from obligation_rag.llm_client import (  # noqa: E402
    FakeLLMClient,
    HallucinatingLLMClient,
    OpenAICompatibleClient,
)
from obligation_rag.retrieval import get_document_index  # noqa: E402
from obligation_rag.schemas import ExtractionResult, ObligationStatus  # noqa: E402

DEFAULT_DOCUMENT = REPO_ROOT / "data" / "samples" / "sample_contract.txt"

SAMPLE_QUERIES = [
    "termination notice period",
    "automatic renewal of the agreement",
    "annual fee increase",
    "limitation of liability cap",
]


def print_result(result: ExtractionResult) -> None:
    print(f"\nretrieval : {result.retrieval_mode}")
    print(f"llm       : {result.llm_mode}")
    print(f"elapsed   : {result.elapsed_seconds}s\n")

    header = f"{'FIELD':28} {'VALUE':22} {'STATUS':9} {'PAGE':5} EVIDENCE"
    print(header)
    print("-" * len(header))
    for obligation in result.obligations:
        evidence = obligation.source_evidence
        page = f"p.{evidence.page}" if evidence else "—"
        if evidence:
            quote = " ".join(evidence.quote.split())
            detail = f'"{quote[:60]}…"' if len(quote) > 60 else f'"{quote}"'
        else:
            detail = f"{obligation.computation_formula} (calculated in code)"
        print(
            f"{obligation.obligation_type.value:28} "
            f"{str(obligation.normalized_value)[:22]:22} "
            f"{obligation.status.value:9} {page:5} {detail}"
        )
        if obligation.status is ObligationStatus.FAILED:
            print(f"{'':28} └─ {obligation.verification_reason}")

    for failure in result.failures:
        label = failure.obligation_type.value if failure.obligation_type else "-"
        print(f"\n! {label}: {failure.reason} — {failure.detail or ''}")

    verdict = "APPROVAL UNLOCKED" if result.can_approve else "APPROVAL BLOCKED"
    print(f"\ncan_approve = {result.can_approve}  ->  {verdict}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument(
        "--real", action="store_true", help="call the configured gpt-oss-120b endpoint"
    )
    parser.add_argument(
        "--ablation", action="store_true", help="run with a hallucinating model instead"
    )
    parser.add_argument(
        "--keep", action="store_true", help="use RAG_DATA_DIR instead of a temp directory"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="obligation-rag-") as scratch:
        base = get_settings()
        settings = (
            base
            if args.keep
            else Settings(
                rag_data_dir=Path(scratch),
                use_fake_llm=base.use_fake_llm,
                use_fake_embeddings=base.use_fake_embeddings,
                embedding_model_path=base.embedding_model_path,
                llm_base_url=base.llm_base_url,
                llm_api_key=base.llm_api_key,
                llm_model=base.llm_model,
            )
        )

        print(f"→ ingesting {args.path}")
        outcome = ingest_path(settings, args.path)
        print(
            f"  {outcome.document_id}: {outcome.page_count} pages, "
            f"{outcome.chunk_count} chunks, {outcome.retrieval_mode}"
        )

        index = get_document_index(settings, outcome.document_id)
        assert index is not None

        print("\n→ retrieval check")
        for query in SAMPLE_QUERIES:
            hits = index.search(query, top_k=1)
            if hits:
                snippet = " ".join(hits[0].text.split())[:70]
                print(f"  {query:38} -> p.{hits[0].page}  {snippet}…")
            else:
                print(f"  {query:38} -> (no hit)")

        if args.ablation:
            client = HallucinatingLLMClient()
            print("\n→ extraction with a HALLUCINATING model (ablation demo)")
        elif args.real:
            client = OpenAICompatibleClient(settings)
            print(f"\n→ extraction with {settings.llm_model} at {settings.llm_base_url}")
        else:
            client = FakeLLMClient()
            print("\n→ extraction with the deterministic fake model")

        pages = storage.get_pages(settings, outcome.document_id)
        result = run_extraction(settings, outcome.document_id, index, pages, client=client)
        print_result(result)

        return 0 if (result.can_approve or args.ablation) else 1


if __name__ == "__main__":
    raise SystemExit(main())
