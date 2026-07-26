"""Embedding backends behind a single abstraction.

The service must run on a box that may not have model weights at all, so dense
retrieval is optional by design:

* ``SentenceTransformerBackend`` — a locally downloaded sentence-transformers
  model (``EMBEDDING_MODEL_PATH``). Nothing is ever downloaded at runtime.
* ``HashingEmbeddingBackend`` — deterministic hashed bag-of-features. No
  weights, no network, reproducible; used to exercise the hybrid retrieval path
  in tests (``USE_FAKE_EMBEDDINGS=true``).
* ``None`` — no backend available, retrieval degrades to BM25-only.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingBackend(ABC):
    """Turns text into L2-normalized float32 vectors.

    Retrieval encoders are asymmetric: several families (e5, bge, nomic) were
    trained with a short instruction glued to the front of queries, and a
    different one — often nothing — on passages. Encoding both sides the same
    way costs a large slice of their accuracy, silently. The prefixes are
    configuration, so any model can be dropped in without touching code.
    """

    name: str = "abstract"
    query_prefix: str = ""
    document_prefix: str = ""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode search queries, applying the model's query instruction."""
        if not self.query_prefix:
            return self.encode(texts)
        return self.encode([f"{self.query_prefix}{text}" for text in texts])

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode chunks for indexing, applying the model's passage prefix."""
        if not self.document_prefix:
            return self.encode(texts)
        return self.encode([f"{self.document_prefix}{text}" for text in texts])

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic hashed embeddings: word unigrams + character trigrams.

    Not competitive with a real encoder, but it is stable across processes
    (blake2b, not Python's randomized ``hash``) and needs no weights, which
    makes the hybrid retrieval path testable offline.
    """

    name = "hashing"

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            for token in tokens:
                matrix[row, self._bucket(f"w:{token}")] += 1.0
                for start in range(len(token) - 2):
                    matrix[row, self._bucket(f"c:{token[start : start + 3]}")] += 0.5
        # Sublinear scaling keeps long chunks from dominating.
        matrix = np.log1p(matrix)
        return self._l2_normalize(matrix)


class SentenceTransformerBackend(EmbeddingBackend):
    """Local sentence-transformers model. Loaded lazily, never downloaded."""

    name = "sentence-transformers"

    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import

            self._model = SentenceTransformer(self.model_path, local_files_only=True)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return self._l2_normalize(np.asarray(vectors, dtype=np.float32))


#: Loading a sentence-transformers model costs seconds; do it once per process.
_BACKEND_CACHE: dict[tuple[bool, str], EmbeddingBackend | None] = {}


def get_embedding_backend(settings: Settings) -> EmbeddingBackend | None:
    """Pick a backend, or ``None`` to run BM25-only."""
    cache_key = (settings.use_fake_embeddings, (settings.embedding_model_path or "").strip())
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]
    backend = _build_embedding_backend(settings)
    _BACKEND_CACHE[cache_key] = backend
    return backend


def clear_backend_cache() -> None:
    _BACKEND_CACHE.clear()


def _build_embedding_backend(settings: Settings) -> EmbeddingBackend | None:
    if settings.use_fake_embeddings:
        return HashingEmbeddingBackend()

    model_path = (settings.embedding_model_path or "").strip()
    if not model_path:
        return None

    if not Path(model_path).exists():
        logger.warning(
            "EMBEDDING_MODEL_PATH=%s does not exist; falling back to BM25-only retrieval",
            model_path,
        )
        return None

    try:
        backend = SentenceTransformerBackend(
            model_path,
            batch_size=settings.embedding_batch_size,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
        backend.encode(["warmup"])  # fail fast at startup, not mid-ingestion
        return backend
    except Exception as error:  # pragma: no cover - depends on local install
        logger.warning(
            "could not load embedding model at %s (%s); falling back to BM25-only retrieval",
            model_path,
            error,
        )
        return None
