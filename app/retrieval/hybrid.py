from app.documents.models import MarkdownChunk
from app.indexing.embeddings import EmbeddingProvider
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.schemas import RetrievedChunk
from app.retrieval.vector import VectorRetriever


class HybridRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.vector_retriever = VectorRetriever(embedding_provider)
        self.keyword_retriever = KeywordRetriever()

    def search(
        self,
        chunks: list[MarkdownChunk],
        query: str,
        top_k: int,
        vector_weight: float = 0.35,
        keyword_weight: float = 0.65,
        min_score: float = 0.4,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
        vector_results, _ = self.vector_retriever.search(chunks, query, top_k=max(top_k * 4, 20))
        keyword_results = self.keyword_retriever.search(chunks, query, top_k=max(top_k * 4, 20))

        return (
            merge_results(
                vector_results=vector_results,
                keyword_results=keyword_results,
                top_k=top_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                min_score=min_score,
            ),
            vector_results[:top_k],
            keyword_results[:top_k],
        )


def key(result: RetrievedChunk) -> str:
    return f"{result.document_path}:{result.chunk_index}"


def merge_results(
    *,
    vector_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    top_k: int,
    vector_weight: float = 0.35,
    keyword_weight: float = 0.65,
    min_score: float = 0.4,
) -> list[RetrievedChunk]:
    vector_scores = _normalize({key(result): result.score for result in vector_results})
    keyword_scores = _normalize({key(result): result.score for result in keyword_results})
    by_key = {key(result): result for result in [*vector_results, *keyword_results]}

    merged = []
    for item_key, result in by_key.items():
        score = vector_weight * vector_scores.get(item_key, 0.0)
        score += keyword_weight * keyword_scores.get(item_key, 0.0)
        if score >= min_score:
            merged.append((result, score))

    merged.sort(key=lambda item: item[1], reverse=True)
    return [
        result.model_copy(update={"rank": rank, "score": score})
        for rank, (result, score) in enumerate(merged[:top_k], start=1)
    ]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {item_key: 1.0 for item_key in scores}
    return {
        item_key: (score - minimum) / (maximum - minimum)
        for item_key, score in scores.items()
    }
