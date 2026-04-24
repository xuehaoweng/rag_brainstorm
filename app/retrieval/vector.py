from app.documents.models import MarkdownChunk
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.vector_store import InMemoryVectorStore
from app.retrieval.schemas import RetrievedChunk, VectorSearchDebug


class VectorRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def search(
        self,
        chunks: list[MarkdownChunk],
        query: str,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], VectorSearchDebug]:
        searchable_chunks = [chunk for chunk in chunks if _is_searchable(chunk)]
        texts = [embedding_text(chunk) for chunk in searchable_chunks]
        chunk_vectors = self.embedding_provider.embed_texts(texts) if texts else []
        query_vector = self.embedding_provider.embed_texts([query])[0]

        store = InMemoryVectorStore()
        for chunk, vector in zip(searchable_chunks, chunk_vectors, strict=True):
            store.add(chunk, vector)

        results = [
            RetrievedChunk(
                rank=result.rank,
                score=result.score,
                document_path=result.chunk.document_path,
                chunk_index=result.chunk.chunk_index,
                heading_path=result.chunk.heading_path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                text=result.chunk.text,
                text_hash=result.chunk.text_hash,
            )
            for result in store.search(query_vector, top_k=top_k)
        ]
        return (
            results,
            VectorSearchDebug(
                provider=self.embedding_provider.name,
                indexed_chunk_count=len(store),
                query_vector_dimensions=len(query_vector),
            ),
        )


def embedding_text(chunk: MarkdownChunk) -> str:
    """Include source metadata so title-like queries can hit the right note."""

    return "\n".join(
        part
        for part in [
            f"文件名：{chunk.document_path}",
            f"标题路径：{chunk.heading_path}",
            chunk.text,
        ]
        if part
    )


def is_searchable(chunk: MarkdownChunk) -> bool:
    # Heading-only chunks create noisy high-similarity matches in small corpora.
    body = "\n".join(
        line
        for line in chunk.text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return len(body.strip()) >= 30


_embedding_text = embedding_text
_is_searchable = is_searchable
