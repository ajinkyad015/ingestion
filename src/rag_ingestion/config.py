from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    All configuration must be provided through .env or environment
    variables. No configuration values are hardcoded outside of
    documented defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    #
    # Application
    #
    app_name: str = Field(
        default="rag-ingestion",
        alias="APP_NAME",
    )

    app_env: Literal[
        "development",
        "test",
        "production",
    ] = Field(
        default="development",
        alias="APP_ENV",
    )

    app_debug: bool = Field(
        default=False,
        alias="APP_DEBUG",
    )

    #
    # Logging
    #
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    log_format: Literal[
        "json",
        "console",
    ] = Field(
        default="json",
        alias="LOG_FORMAT",
    )

    #
    # Loader
    #
    supported_extensions: str = Field(
        default=".pdf,.docx,.txt,.md,.markdown,.html,.htm",
        alias="SUPPORTED_EXTENSIONS",
    )

    #
    # Chunking
    #
    chunk_size: int = Field(
        default=512,
        alias="CHUNK_SIZE",
        ge=64,
    )

    chunk_overlap: int = Field(
        default=64,
        alias="CHUNK_OVERLAP",
        ge=0,
    )

    chunking_strategy: Literal["recursive"] = Field(
        default="recursive",
        alias="CHUNKING_STRATEGY",
    )

    #
    # Embeddings
    #
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    embedding_batch_size: int = Field(
        default=32,
        alias="EMBEDDING_BATCH_SIZE",
        ge=1,
    )

    embedding_device: str = Field(
        default="cpu",
        alias="EMBEDDING_DEVICE",
    )

    embedding_normalize: bool = Field(
        default=True,
        alias="EMBEDDING_NORMALIZE",
    )

    #
    # ChromaDB
    #
    chroma_persist_directory: Path = Field(
        default=Path("./chroma"),
        alias="CHROMA_PERSIST_DIRECTORY",
    )

    chroma_collection: str = Field(
        default="documents",
        alias="CHROMA_COLLECTION",
    )

    #
    # Metadata
    #
    hash_algorithm: Literal["sha256"] = Field(
        default="sha256",
        alias="HASH_ALGORITHM",
    )

    #
    # Parser
    #
    docling_enable_ocr: bool = Field(
        default=False,
        alias="DOCLING_ENABLE_OCR",
    )

    #
    # Performance
    #
    max_workers: int = Field(
        default=4,
        alias="MAX_WORKERS",
        ge=1,
    )

    #
    # CLI
    #
    default_input_directory: Path = Field(
        default=Path("./documents"),
        alias="DEFAULT_INPUT_DIRECTORY",
    )

    @property
    def supported_extensions_list(self) -> tuple[str, ...]:
        """
        Returns the configured file extensions as a normalized tuple.
        """
        return tuple(
            extension.strip().lower()
            for extension in self.supported_extensions.split(",")
            if extension.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings.

    The settings object is instantiated once per process to avoid
    repeatedly parsing environment variables.
    """
    return Settings()