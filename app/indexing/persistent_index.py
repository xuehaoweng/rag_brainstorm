from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import settings
from app.db import connect, init_db
from app.documents.loader import scan_markdown_folder
from app.documents.models import MarkdownChunk
from app.indexing.chunker import chunk_markdown_document
from app.indexing.embeddings import EmbeddingProvider
from app.retrieval.schemas import RetrievedChunk, VectorSearchDebug
from app.retrieval.vector import embedding_text, is_searchable


@dataclass(frozen=True, slots=True)
class PersistentIndexBuildResult:
    document_count: int
    chunk_count: int
    indexed_chunk_count: int
    provider: str
    vector_dimensions: int
    root: str
    database_path: str
    faiss_index_path: str
    built_at: str


@dataclass(frozen=True, slots=True)
class PersistentIndexStatus:
    exists: bool
    document_count: int
    chunk_count: int
    indexed_chunk_count: int
    provider: str | None
    vector_dimensions: int | None
    root: str | None
    database_path: str
    faiss_index_path: str
    built_at: str | None


@dataclass(frozen=True, slots=True)
class PersistentVectorSearchResult:
    document_count: int
    chunk_count: int
    results: list[RetrievedChunk]
    debug: VectorSearchDebug


def build_persistent_index(
    *,
    root: Path,
    embedding_provider: EmbeddingProvider,
    max_chars: int,
    overlap_chars: int,
    database_path: Path | None = None,
    faiss_index_path: Path | None = None,
) -> PersistentIndexBuildResult:
    faiss = _load_faiss()
    init_db(database_path)
    documents = scan_markdown_folder(root)
    chunks_by_document = [
        (
            document,
            chunk_markdown_document(
                document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            ),
        )
        for document in documents
    ]
    searchable_chunks = [
        chunk
        for _, chunks in chunks_by_document
        for chunk in chunks
        if is_searchable(chunk)
    ]
    vectors = (
        embedding_provider.embed_texts([embedding_text(chunk) for chunk in searchable_chunks])
        if searchable_chunks
        else []
    )
    vector_dimensions = len(vectors[0]) if vectors else 0
    built_at = datetime.now(UTC).isoformat()
    index_path = faiss_index_path or settings.faiss_index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with connect(database_path) as connection:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")
        connection.execute("DELETE FROM index_metadata")

        chunk_ids_by_key: dict[tuple[str, int], int] = {}
        for document, chunks in chunks_by_document:
            cursor = connection.execute(
                """
                INSERT INTO documents(path, content_hash, modified_time, size, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document.relative_path,
                    document.content_hash,
                    document.modified_time,
                    document.size,
                    built_at,
                ),
            )
            document_id = int(cursor.lastrowid)
            for chunk in chunks:
                chunk_cursor = connection.execute(
                    """
                    INSERT INTO chunks(
                        document_id,
                        chunk_index,
                        heading_path,
                        start_line,
                        end_line,
                        text,
                        text_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk.chunk_index,
                        chunk.heading_path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        chunk.text_hash,
                    ),
                )
                chunk_ids_by_key[(chunk.document_path, chunk.chunk_index)] = int(chunk_cursor.lastrowid)

        vector_ids = np.array(
            [chunk_ids_by_key[(chunk.document_path, chunk.chunk_index)] for chunk in searchable_chunks],
            dtype=np.int64,
        )
        _write_metadata(
            connection,
            {
                "root": str(root),
                "provider": embedding_provider.name,
                "vector_dimensions": str(vector_dimensions),
                "document_count": str(len(documents)),
                "chunk_count": str(sum(len(chunks) for _, chunks in chunks_by_document)),
                "indexed_chunk_count": str(len(searchable_chunks)),
                "faiss_index_path": str(index_path),
                "built_at": built_at,
            },
        )

    if vectors:
        matrix = _vector_matrix(vectors)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vector_dimensions))
        index.add_with_ids(matrix, vector_ids)
        faiss.write_index(index, str(index_path))
    elif index_path.exists():
        index_path.unlink()

    return PersistentIndexBuildResult(
        document_count=len(documents),
        chunk_count=sum(len(chunks) for _, chunks in chunks_by_document),
        indexed_chunk_count=len(searchable_chunks),
        provider=embedding_provider.name,
        vector_dimensions=vector_dimensions,
        root=str(root),
        database_path=str(database_path or settings.database_path),
        faiss_index_path=str(index_path),
        built_at=built_at,
    )


def get_persistent_index_status(
    *,
    database_path: Path | None = None,
    faiss_index_path: Path | None = None,
) -> PersistentIndexStatus:
    init_db(database_path)
    index_path = faiss_index_path or settings.faiss_index_path
    with connect(database_path) as connection:
        metadata = _read_metadata(connection)
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    indexed_chunk_count = int(metadata["indexed_chunk_count"]) if metadata.get("indexed_chunk_count") else 0
    vector_dimensions = int(metadata["vector_dimensions"]) if metadata.get("vector_dimensions") else None
    return PersistentIndexStatus(
        exists=bool(metadata) and index_path.exists() and indexed_chunk_count > 0,
        document_count=int(document_count),
        chunk_count=int(chunk_count),
        indexed_chunk_count=indexed_chunk_count,
        provider=metadata.get("provider"),
        vector_dimensions=vector_dimensions,
        root=metadata.get("root"),
        database_path=str(database_path or settings.database_path),
        faiss_index_path=str(index_path),
        built_at=metadata.get("built_at"),
    )


def search_persistent_vector_index(
    *,
    query: str,
    top_k: int,
    embedding_provider: EmbeddingProvider,
    database_path: Path | None = None,
    faiss_index_path: Path | None = None,
) -> PersistentVectorSearchResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    faiss = _load_faiss()
    index_path = faiss_index_path or settings.faiss_index_path
    status = get_persistent_index_status(database_path=database_path, faiss_index_path=index_path)
    if not status.exists:
        raise ValueError("Persistent index does not exist. Build it first with /api/index/build.")
    if status.provider != embedding_provider.name:
        raise ValueError(
            f"Persistent index was built with provider '{status.provider}', "
            f"but query requested '{embedding_provider.name}'. Rebuild the index or change provider."
        )

    query_vector = embedding_provider.embed_texts([query])[0]
    index = faiss.read_index(str(index_path))
    scores, ids = index.search(_vector_matrix([query_vector]), top_k)
    chunk_ids = [int(item) for item in ids[0].tolist() if int(item) >= 0]
    score_by_id = {
        int(chunk_id): float(score)
        for chunk_id, score in zip(ids[0].tolist(), scores[0].tolist(), strict=True)
        if int(chunk_id) >= 0
    }
    chunks = _load_chunks_by_ids(chunk_ids, database_path=database_path)
    results = [
        RetrievedChunk(
            rank=rank,
            score=score_by_id[chunk_id],
            document_path=chunk.document_path,
            chunk_index=chunk.chunk_index,
            heading_path=chunk.heading_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=chunk.text,
            text_hash=chunk.text_hash,
        )
        for rank, (chunk_id, chunk) in enumerate(chunks, start=1)
    ]
    return PersistentVectorSearchResult(
        document_count=status.document_count,
        chunk_count=status.chunk_count,
        results=results,
        debug=VectorSearchDebug(
            provider=embedding_provider.name,
            indexed_chunk_count=status.indexed_chunk_count,
            query_vector_dimensions=len(query_vector),
        ),
    )


def load_persistent_chunks(*, database_path: Path | None = None) -> list[MarkdownChunk]:
    init_db(database_path)
    with connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                documents.path AS document_path,
                chunks.chunk_index,
                chunks.heading_path,
                chunks.start_line,
                chunks.end_line,
                chunks.text,
                chunks.text_hash
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY documents.path, chunks.chunk_index
            """
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def _load_chunks_by_ids(
    chunk_ids: list[int],
    *,
    database_path: Path | None = None,
) -> list[tuple[int, MarkdownChunk]]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    with connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                chunks.id AS chunk_id,
                documents.path AS document_path,
                chunks.chunk_index,
                chunks.heading_path,
                chunks.start_line,
                chunks.end_line,
                chunks.text,
                chunks.text_hash
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    chunks_by_id = {int(row["chunk_id"]): _chunk_from_row(row) for row in rows}
    return [(chunk_id, chunks_by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in chunks_by_id]


def _chunk_from_row(row) -> MarkdownChunk:
    return MarkdownChunk(
        document_path=row["document_path"],
        chunk_index=int(row["chunk_index"]),
        heading_path=row["heading_path"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        text=row["text"],
        text_hash=row["text_hash"],
    )


def _write_metadata(connection, metadata: dict[str, str]) -> None:
    for key, value in metadata.items():
        connection.execute(
            """
            INSERT INTO index_metadata(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )


def _read_metadata(connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
    return {row["key"]: row["value"] for row in rows}


def _vector_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Install project dependencies before using persistent indexing."
        ) from exc
    return faiss
