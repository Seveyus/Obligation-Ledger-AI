"""Shared fixtures. Every test runs against a throwaway data directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from obligation_rag.chunking import chunk_document
from obligation_rag.config import Settings
from obligation_rag.embeddings import HashingEmbeddingBackend
from obligation_rag.pdf_parser import ParsedDocument, parse_document
from obligation_rag.retrieval import DocumentIndex, build_index, clear_index_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONTRACT = REPO_ROOT / "data" / "samples" / "sample_contract.txt"

DOCUMENT_ID = "contract_test"


@pytest.fixture(autouse=True)
def _clean_index_cache():
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Hermetic settings: explicit kwargs beat both the environment and .env."""
    return Settings(
        rag_data_dir=tmp_path / "runtime-data",
        use_fake_llm=True,
        use_fake_embeddings=False,
        embedding_model_path=None,
    )


@pytest.fixture
def sample_contract_path() -> Path:
    assert SAMPLE_CONTRACT.exists(), "synthetic fixture is missing"
    return SAMPLE_CONTRACT


@pytest.fixture
def parsed_contract(sample_contract_path: Path) -> ParsedDocument:
    return parse_document(sample_contract_path)


@pytest.fixture
def contract_pages(parsed_contract: ParsedDocument) -> dict[int, str]:
    return parsed_contract.page_map()


@pytest.fixture
def bm25_index(settings: Settings, parsed_contract: ParsedDocument) -> DocumentIndex:
    chunks = chunk_document(
        DOCUMENT_ID,
        parsed_contract,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )
    return build_index(DOCUMENT_ID, chunks, settings=settings)


@pytest.fixture
def hybrid_index(settings: Settings, parsed_contract: ParsedDocument) -> DocumentIndex:
    chunks = chunk_document(
        DOCUMENT_ID,
        parsed_contract,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )
    backend = HashingEmbeddingBackend()
    embeddings = backend.encode([chunk.text for chunk in chunks])
    return build_index(
        DOCUMENT_ID, chunks, settings=settings, embeddings=embeddings, backend=backend
    )
