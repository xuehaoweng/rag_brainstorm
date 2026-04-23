from dataclasses import dataclass
from math import sqrt

from app.documents.models import MarkdownChunk


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk: MarkdownChunk
    vector: list[float]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    chunk: MarkdownChunk
    score: float
    rank: int


class InMemoryVectorStore:
    """Simple cosine-similarity vector store for the first retrieval milestone."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, chunk: MarkdownChunk, vector: list[float]) -> None:
        if not vector:
            raise ValueError("vector must not be empty")
        self._records.append(VectorRecord(chunk=chunk, vector=vector))

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query_vector:
            raise ValueError("query_vector must not be empty")

        scored = [
            VectorSearchResult(
                chunk=record.chunk,
                score=_cosine(query_vector, record.vector),
                rank=0,
            )
            for record in self._records
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return [
            VectorSearchResult(chunk=result.chunk, score=result.score, rank=index)
            for index, result in enumerate(scored[:top_k], start=1)
        ]

    def __len__(self) -> int:
        return len(self._records)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
