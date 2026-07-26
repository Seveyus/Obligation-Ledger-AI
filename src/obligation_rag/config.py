"""Runtime configuration.

Every field can be overridden by an environment variable of the same name in
upper case (``LLM_BASE_URL``, ``RAG_DATA_DIR``, ...) or by a ``.env`` file at
the repository root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Local reasoning model (served outside this repository) ---
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "local"
    llm_model: str = "gpt-oss-120b"
    llm_timeout_seconds: float = 180.0
    llm_max_output_tokens: int = 4096
    use_fake_llm: bool = False

    # --- Embeddings (optional: the service degrades to BM25-only) ---
    embedding_model_path: str | None = None
    use_fake_embeddings: bool = False
    embedding_batch_size: int = 16

    # --- Storage / service ---
    rag_data_dir: Path = Path("./runtime-data")
    rag_host: str = "127.0.0.1"
    rag_port: int = 8001

    # --- Chunking ---
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    # --- Retrieval ---
    default_top_k: int = 6
    rrf_k: int = 60
    candidate_pool: int = 25

    # --- Verification ---
    fuzzy_match_threshold: float = 0.92
    fuzzy_max_length_ratio: float = 1.35
    min_quote_chars: int = 12
    #: Reject a quote lifted from an unticked form option (☐).
    reject_unchecked_options: bool = True

    # --- Parsing ---
    #: Drop running headers/footers repeated across pages.
    strip_page_furniture: bool = True

    @property
    def uploads_dir(self) -> Path:
        return self.rag_data_dir / "uploads"

    @property
    def indexes_dir(self) -> Path:
        return self.rag_data_dir / "indexes"

    @property
    def db_path(self) -> Path:
        return self.rag_data_dir / "ledger_rag.sqlite3"

    def ensure_directories(self) -> None:
        for directory in (self.rag_data_dir, self.uploads_dir, self.indexes_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests after patching the environment)."""
    get_settings.cache_clear()
