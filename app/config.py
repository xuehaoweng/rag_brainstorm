from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the local RAG service."""

    app_name: str = "RAG Learning"
    database_path: Path = Path("data/sqlite/rag_learning.db")
    faiss_index_path: Path = Path("data/faiss/rag_learning.index")
    default_markdown_root: Path | None = None
    embedding_provider: str = "bge-m3"
    embedding_model_path: Path = Path("models/BAAI/bge-m3")
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_api_key: str | None = None
    llm_model: str = "doubao-seed-2-0-code-preview-260215"
    llm_timeout_seconds: float = 60.0

    vector_backend: str = "faiss"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_learning_chunks"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_LEARNING_",
        extra="ignore",
    )


settings = Settings()
