from pathlib import Path

from app.main import (
    IndexPreviewRequest,
    VectorSearchRequest,
    health,
    hybrid_search,
    keyword_search,
    preview_index,
    vector_search,
)


def test_health() -> None:
    assert health() == {"status": "ok"}


def test_preview_index_returns_chunks(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# RAG\n\nHybrid retrieval matters.", encoding="utf-8")

    response = preview_index(IndexPreviewRequest(root=tmp_path))

    assert response.document_count == 1
    assert response.chunk_count == 1
    assert response.chunks[0].document_path == "note.md"
    assert response.chunks[0].heading_path == "RAG"


def test_vector_search_returns_ranked_chunks(tmp_path: Path) -> None:
    (tmp_path / "python.md").write_text(
        "# Python\n\nPytest verifies retrieval behavior.",
        encoding="utf-8",
    )
    (tmp_path / "garden.md").write_text(
        "# Garden\n\nTomatoes need soil and sunlight.",
        encoding="utf-8",
    )

    response = vector_search(
        VectorSearchRequest(
            root=tmp_path,
            query="pytest retrieval",
            top_k=1,
            embedding_provider="hash",
        )
    )

    assert response.document_count == 2
    assert response.chunk_count == 2
    assert response.debug.provider == "hash"
    assert response.debug.indexed_chunk_count == 2
    assert response.results[0].rank == 1
    assert response.results[0].document_path == "python.md"


def test_keyword_search_matches_exact_terms(tmp_path: Path) -> None:
    (tmp_path / "docker.md").write_text(
        "# Docker 镜像优化\n\n镜像构建的体积优化可以使用多阶段构建。",
        encoding="utf-8",
    )
    (tmp_path / "python.md").write_text("# Python\n\npytest retrieval", encoding="utf-8")

    response = keyword_search(
        VectorSearchRequest(
            root=tmp_path,
            query="镜像构建的体积优化",
            top_k=1,
            embedding_provider="hash",
        )
    )

    assert response.results[0].document_path == "docker.md"


def test_hybrid_search_includes_keyword_results(tmp_path: Path) -> None:
    (tmp_path / "docker.md").write_text(
        "# Docker 镜像优化\n\n镜像构建的体积优化可以使用多阶段构建。",
        encoding="utf-8",
    )
    (tmp_path / "garden.md").write_text("# Garden\n\nTomatoes need sunlight.", encoding="utf-8")

    response = hybrid_search(
        VectorSearchRequest(
            root=tmp_path,
            query="镜像构建的体积优化",
            top_k=1,
            embedding_provider="hash",
        )
    )

    assert response.results[0].document_path == "docker.md"
    assert response.keyword_results[0].document_path == "docker.md"
