"""Application configuration module.

This module centralises runtime configuration using pydantic settings. It
collects environment variables for external services such as OpenRouter, Milvus,
Tavily and application level flags.  The `Settings` instance is imported
throughout the codebase to ensure a single source of truth for configuration
values.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Pydantic settings for the EduPolicy Agent service."""

    # --- Core paths -----------------------------------------------------
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")

    # --- LLM configuration ----------------------------------------------
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        env="OPENROUTER_BASE_URL",
    )
    openrouter_model: str = Field(
        default="openrouter/auto",
        description=(
            "Identifier for the OpenRouter hosted model. Users can override this"
            " value in the environment to target a preferred free model."
        ),
    )

    # --- Embeddings -----------------------------------------------------
    embedding_model: str = Field(default="intfloat/e5-large-v2")
    chunk_size: int = Field(default=750)
    chunk_overlap: int = Field(default=150)
    top_k: int = Field(default=4, description="Default number of RAG results to return.")

    # --- Milvus settings -------------------------------------------------
    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_collection: str = Field(default="regulations_collection")
    milvus_dim: int = Field(default=1024, description="Embedding dimension for e5-large-v2.")

    # --- SQLite ---------------------------------------------------------
    sqlite_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "student_records.db")

    # --- External search ------------------------------------------------
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    tavily_max_results: int = Field(default=4)

    # --- Application ----------------------------------------------------
    session_memory_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "session_memory.json")
    enable_debug_logging: bool = Field(default=False)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.enable_debug_logging:
        import logging

        logging.basicConfig(level=logging.DEBUG)
    return settings


settings = get_settings()
