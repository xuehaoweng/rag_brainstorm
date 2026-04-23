from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the local RAG service."""

    app_name: str = "Self RAG"
    database_path: Path = Path("data/sqlite/self_rag.db")
    default_markdown_root: Path | None = None
    embedding_provider: str = "bge-m3"
    embedding_model_path: Path = Path("models/BAAI/bge-m3")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SELF_RAG_",
        extra="ignore",
    )


settings = Settings()
