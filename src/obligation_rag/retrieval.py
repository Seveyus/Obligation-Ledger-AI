"""Hybrid retrieval: BM25 + local embeddings, fused with Reciprocal Rank Fusion.

Why RRF rather than a weighted score blend: BM25 scores and cosine similarities
live on different, document-dependent scales, so any fixed weighting is a
guess. RRF only consumes *ranks*, which is exactly what is comparable between
the two systems, and it degrades gracefully to "whatever ranker is available"
when no embedding backend is loaded.

Contract clauses are lexically distinctive ("sixty (60) days' written notice"),
so BM25 alone already retrieves well — the dense side is there for paraphrases
("cancel the agreement" vs "terminate this Agreement").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .config import Settings
from .embeddings import EmbeddingBackend, get_embedding_backend
from .schemas import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")

RETRIEVAL_MODE_HYBRID = "hybrid_bm25_vector_rrf"
RETRIEVAL_MODE_BM25 = "bm25_only"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Numbers matter here ("60", "2026")."""
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """Fuse ranked id lists. ``score(d) = sum_i w_i / (k + rank_i(d))``, rank 1-based."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + position)
    return fused


@dataclass(slots=True)
class DocumentIndex:
    """In-memory retrieval index for a single document."""

    document_id: str
    chunks: list[Chunk]
    bm25: BM25Okapi
    embeddings: np.ndarray | None = None
    backend: EmbeddingBackend | None = None
    rrf_k: int = 60
    candidate_pool: int = 25

    @property
    def retrieval_mode(self) -> str:
        has_vectors = self.embeddings is not None and self.backend is not None
        return RETRIEVAL_MODE_HYBRID if has_vectors else RETRIEVAL_MODE_BM25

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]

        lexical_raw = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        highest = float(lexical_raw.max()) if lexical_raw.size else 0.0
        lexical_scores = lexical_raw / highest if highest > 0 else lexical_raw
        lexical_ranking = [
            chunk_ids[i]
            for i in np.argsort(-lexical_raw)[: self.candidate_pool]
            if lexical_raw[i] > 0
        ]

        vector_scores = np.zeros(len(self.chunks), dtype=np.float32)
        rankings = [lexical_ranking]
        if self.embeddings is not None and self.backend is not None:
            query_vector = self.backend.encode([query])[0]
            vector_scores = self.embeddings @ query_vector  # both L2-normalized -> cosine
            vector_ranking = [
                chunk_ids[i] for i in np.argsort(-vector_scores)[: self.candidate_pool]
            ]
            rankings.append(vector_ranking)

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        if not fused:  # no lexical hit and no vectors: nothing relevant
            return []

        by_id = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]

        results: list[RetrievedChunk] = []
        for chunk_id, fused_score in ordered:
            index = by_id[chunk_id]
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page=chunk.page,
                    text=chunk.text,
                    lexical_score=round(float(lexical_scores[index]), 6),
                    vector_score=round(float(vector_scores[index]), 6),
                    fused_score=round(float(fused_score), 6),
                )
            )
        return results


def build_index(
    document_id: str,
    chunks: list[Chunk],
    *,
    settings: Settings,
    embeddings: np.ndarray | None = None,
    backend: EmbeddingBackend | None = None,
) -> DocumentIndex:
    corpus = [tokenize(chunk.text) for chunk in chunks] or [[""]]
    if embeddings is not None and len(embeddings) != len(chunks):
        embeddings = None  # stale index on disk; BM25-only until re-ingested
    return DocumentIndex(
        document_id=document_id,
        chunks=chunks,
        bm25=BM25Okapi(corpus),
        embeddings=embeddings,
        backend=backend,
        rrf_k=settings.rrf_k,
        candidate_pool=settings.candidate_pool,
    )


# --------------------------------------------------------------------------
# Process-local index cache. Rebuilding BM25 for a contract is cheap, but not
# free on every request.
# --------------------------------------------------------------------------

_INDEX_CACHE: dict[str, DocumentIndex] = {}


def get_document_index(settings: Settings, document_id: str) -> DocumentIndex | None:
    """Load (and cache) the index for ``document_id``, or ``None`` if unknown."""
    from . import storage  # local import keeps storage <-> retrieval decoupled

    cached = _INDEX_CACHE.get(document_id)
    if cached is not None:
        return cached

    chunks = storage.get_chunks(settings, document_id)
    if not chunks:
        return None

    embeddings = storage.load_embeddings(settings, document_id)
    backend = get_embedding_backend(settings) if embeddings is not None else None
    index = build_index(
        document_id, chunks, settings=settings, embeddings=embeddings, backend=backend
    )
    _INDEX_CACHE[document_id] = index
    return index


def cache_index(index: DocumentIndex) -> None:
    _INDEX_CACHE[index.document_id] = index


def invalidate_index(document_id: str) -> None:
    _INDEX_CACHE.pop(document_id, None)


def clear_index_cache() -> None:
    _INDEX_CACHE.clear()
