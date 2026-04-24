from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the local RAG service."""

    app_name: str = "Self RAG"
    database_path: Path = Path("data/sqlite/self_rag.db")
    faiss_index_path: Path = Path("data/faiss/self_rag.index")
    default_markdown_root: Path | None = None
    embedding_provider: str = "bge-m3"
    embedding_model_path: Path = Path("models/BAAI/bge-m3")
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_api_key: str | None = None
    llm_model: str = "doubao-seed-2-0-code-preview-260215"
    llm_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SELF_RAG_",
        extra="ignore",
    )


settings = Settings()
