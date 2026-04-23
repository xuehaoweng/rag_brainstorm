from pathlib import Path

import pytest

from app.documents.models import MarkdownChunk, MarkdownDocument
from app.indexing.chunker import chunk_markdown_document
from app.indexing.embeddings import HashEmbeddingProvider
from app.indexing.vector_store import InMemoryVectorStore


def test_in_memory_vector_store_returns_ranked_results() -> None:
    document = MarkdownDocument(
        path=Path("/notes/search.md"),
        relative_path="search.md",
        content="# Search\n\nPython pytest testing\n\nTomato garden soil",
        content_hash="hash",
        modified_time=0,
        size=0,
    )
    chunks = chunk_markdown_document(document, max_chars=300, overlap_chars=0)
    provider = HashEmbeddingProvider(dimensions=64)
    vectors = provider.embed_texts([chunk.text for chunk in chunks])
    query_vector = provider.embed_texts(["python pytest testing"])[0]
    store = InMemoryVectorStore()

    for chunk, vector in zip(chunks, vectors, strict=True):
        store.add(chunk, vector)

    results = store.search(query_vector, top_k=1)

    assert len(results) == 1
    assert results[0].rank == 1
    assert "Python pytest testing" in results[0].chunk.text
    assert results[0].score > 0.5


def test_in_memory_vector_store_rejects_mismatched_dimensions() -> None:
    chunk = MarkdownChunk(
        document_path="note.md",
        chunk_index=0,
        heading_path="Note",
        start_line=1,
        end_line=1,
        text="text",
        text_hash="hash",
    )
    store = InMemoryVectorStore()
    store.add(chunk, [1.0, 0.0])

    with pytest.raises(ValueError, match="same dimensions"):
        store.search([1.0], top_k=1)
