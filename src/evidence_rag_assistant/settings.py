"""Service configuration, read from environment variables.

Every setting has a default matching the evaluated local configuration and can be
overridden by an environment variable prefixed EVIDENCERAG_ (or a .env file), for
example EVIDENCERAG_K=5 or EVIDENCERAG_DATA_DIR=/mnt/data. The same code therefore
runs locally and in a container; only the environment differs.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVIDENCERAG_", env_file=".env", extra="ignore")

    # Data: chunks and index live under data_dir (on GCP, a mounted bucket).
    data_dir: Path = Path("data")

    # Retrieval, as evaluated on LitQA2 (see notebooks/03_experiments.ipynb).
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    k: int = Field(default=8, ge=1, le=20)
    min_retrieval_score: float = Field(default=0.6, ge=0.0, le=1.0)

    # Language model.
    llm_backend: Literal["ollama"] = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_timeout_seconds: float = Field(default=300, gt=0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    llm_auth: Literal["none", "google_id_token"] = "none"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "processed" / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    """Read settings once per process."""
    return Settings()
