from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    App-wide configuration loaded from the .env file.
    Extra keys in .env are silently ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # FastAPI server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Milvus — vector store for unstructured document chunks
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection_name: str = Field(default="agentic_rag_chunks")
    milvus_username: str | None = Field(default=None)
    milvus_password: str | None = Field(default=None)

    embedding_dim: int = Field(default=768)          # must match the embedding model output
    chunk_size_tokens: int = Field(default=1000)
    chunk_overlap_tokens: int = Field(default=150)

    retrieval_top_k: int = Field(default=8)
    retrieval_similarity_threshold: float = Field(default=0.35)
    max_context_tokens: int = Field(default=3500)

    # PostgreSQL — structured data store
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432)
    pg_database: str = Field(default="agentic_rag")
    pg_user: str = Field(default="raguser")
    pg_password: str = Field(default="ragpass")

    sql_max_rows: int = Field(default=200)
    sql_timeout_seconds: int = Field(default=30)

    # Ollama — local LLM and embedding server
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    llm_model: str = Field(default="gemma3n:latest")
    embedding_model: str = Field(default="embeddinggemma:latest")

    request_timeout_seconds: int = Field(default=90)
    max_retries: int = Field(default=2)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance (parsed once per process)."""
    return Settings()