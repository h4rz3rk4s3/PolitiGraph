"""
config/settings.py
==================
Centralised, type-safe configuration using Pydantic Settings v2.
All values are loaded from environment variables (or a .env file).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    user: str = Field("neo4j", alias="NEO4J_USER")
    password: str = Field("politigraph", alias="NEO4J_PASSWORD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class QdrantSettings(BaseSettings):
    host: str = Field("localhost", alias="QDRANT_HOST")
    port: int = Field(6333, alias="QDRANT_PORT")
    collection: str = Field("politigraph_speeches", alias="QDRANT_COLLECTION")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class LLMSettings(BaseSettings):
    base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    model: str = Field("mistral:7b-instruct-q4_K_M", alias="LLM_MODEL")
    temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(1024, alias="LLM_MAX_TOKENS")
    request_timeout: int = Field(120, alias="LLM_REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class EmbeddingSettings(BaseSettings):
    model: str = Field("intfloat/multilingual-e5-large", alias="EMBEDDING_MODEL")
    batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE")
    device: Literal["cpu", "cuda", "mps"] = Field("cpu", alias="EMBEDDING_DEVICE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class PipelineSettings(BaseSettings):
    nlp_batch_size: int = Field(50, alias="NLP_BATCH_SIZE")
    gliner_confidence_threshold: float = Field(0.5, alias="GLINER_CONFIDENCE_THRESHOLD")
    gliner_model: str = Field("urchade/gliner_mediumv2.1", alias="GLINER_MODEL")
    raw_data_dir: Path = Field(Path("data/raw"), alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(Path("data/processed"), alias="PROCESSED_DATA_DIR")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @field_validator("raw_data_dir", "processed_data_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class Settings(BaseSettings):
    """Root settings object — aggregate of all sub-configs."""

    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
