from collections import Counter
from math import log
import re

from app.documents.models import MarkdownChunk
from app.retrieval.schemas import RetrievedChunk


class KeywordRetriever:
    """Small lexical retriever for exact terms, filenames, and Chinese phrases."""

    def search(
        self,
        chunks: list[MarkdownChunk],
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = _significant_terms(query)
        if not query_terms:
            return []

        scored = []
        for chunk in chunks:
            text = _search_text(chunk)
            score = _score(text, query, query_terms)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievedChunk(
                rank=rank,
                score=score,
                document_path=chunk.document_path,
                chunk_index=chunk.chunk_index,
                heading_path=chunk.heading_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
                text_hash=chunk.text_hash,
            )
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]


def _search_text(chunk: MarkdownChunk) -> str:
    return "\n".join([chunk.document_path, chunk.heading_path, chunk.text]).lower()


def _score(text: str, query: str, query_terms: list[str]) -> float:
    normalized_query = query.lower().strip()
    score = 0.0
    if normalized_query and normalized_query in text:
        score += 8.0

    counts = Counter(_terms(text))
    matched_terms = 0
    for term in query_terms:
        if term in counts:
            matched_terms += 1
            score += (1.0 + log(1 + counts[term])) * _term_weight(term)
    required_matches = min(2, len(query_terms))
    return score if matched_terms >= required_matches else 0.0


def _terms(text: str) -> list[str]:
    lowered = text.lower()
    terms = re.findall(r"[a-z0-9_./:-]+", lowered)
    cjk = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for segment in cjk:
        terms.extend(_cjk_terms(segment))
    return terms


def _significant_terms(text: str) -> list[str]:
    return sorted(
        {
            term
            for term in _terms(text)
            if len(term) >= 2 and term not in {"的", "了", "和", "与", "及"}
        },
        key=len,
        reverse=True,
    )


def _cjk_terms(segment: str) -> list[str]:
    terms = list(segment)
    terms.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    terms.extend(segment[index : index + 3] for index in range(max(0, len(segment) - 2)))
    terms.append(segment)
    return terms


def _term_weight(term: str) -> float:
    if len(term) >= 3:
        return 1.6
    if len(term) == 2:
        return 1.2
    return 0.4
