from pathlib import Path

import pytest

pytest.importorskip("faiss")

from app.indexing.embeddings import HashEmbeddingProvider
from app.indexing.persistent_index import (
    build_persistent_index,
    get_persistent_index_status,
    search_persistent_vector_index,
)


def test_build_persistent_index_writes_sqlite_and_faiss(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "python.md").write_text(
        "# Python\n\nPytest verifies retrieval behavior with enough body text.",
        encoding="utf-8",
    )
    (notes / "garden.md").write_text(
        "# Garden\n\nTomatoes need soil and sunlight in a small garden.",
        encoding="utf-8",
    )
    database_path = tmp_path / "self_rag.db"
    faiss_index_path = tmp_path / "self_rag.index"

    result = build_persistent_index(
        root=notes,
        embedding_provider=HashEmbeddingProvider(dimensions=64),
        max_chars=300,
        overlap_chars=0,
        database_path=database_path,
        faiss_index_path=faiss_index_path,
    )
    status = get_persistent_index_status(
        database_path=database_path,
        faiss_index_path=faiss_index_path,
    )

    assert result.document_count == 2
    assert result.chunk_count == 2
    assert result.indexed_chunk_count == 2
    assert result.vector_dimensions == 64
    assert faiss_index_path.exists()
    assert status.exists is True
    assert status.provider == "hash"


def test_search_persistent_vector_index_returns_stored_chunks(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "python.md").write_text(
        "# Python\n\nPytest verifies retrieval behavior with enough body text.",
        encoding="utf-8",
    )
    database_path = tmp_path / "self_rag.db"
    faiss_index_path = tmp_path / "self_rag.index"
    provider = HashEmbeddingProvider(dimensions=64)
    build_persistent_index(
        root=notes,
        embedding_provider=provider,
        max_chars=300,
        overlap_chars=0,
        database_path=database_path,
        faiss_index_path=faiss_index_path,
    )

    result = search_persistent_vector_index(
        query="pytest retrieval",
        top_k=1,
        embedding_provider=provider,
        database_path=database_path,
        faiss_index_path=faiss_index_path,
    )

    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.debug.provider == "hash"
    assert result.results[0].document_path == "python.md"
