"""Retrieval quality metrics: Recall@K, Precision@K, MRR."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation.dataset import EvalCase, ExpectedSource
from app.retrieval.schemas import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Metrics for a single eval case."""
    recall_at_k: float
    precision_at_k: float
    mrr: float  # Mean Reciprocal Rank (for this single case, it's just RR)
    hits: int
    expected_count: int
    retrieved_count: int


@dataclass(slots=True)
class AggregateMetrics:
    """Aggregated metrics across all eval cases."""
    mean_recall: float = 0.0
    mean_precision: float = 0.0
    mean_mrr: float = 0.0
    total_cases: int = 0
    per_case: list[RetrievalMetrics] = field(default_factory=list)


def compute_retrieval_metrics(
    case: EvalCase,
    results: list[RetrievedChunk],
) -> RetrievalMetrics:
    """Compute retrieval metrics for a single query."""
    if not case.expected_sources:
        return RetrievalMetrics(
            recall_at_k=1.0,
            precision_at_k=1.0 if not results else 0.0,
            mrr=0.0,
            hits=0,
            expected_count=0,
            retrieved_count=len(results),
        )

    hits = 0
    first_hit_rank = 0

    for rank, chunk in enumerate(results, start=1):
        if _matches_any_expected(chunk, case.expected_sources):
            hits += 1
            if first_hit_rank == 0:
                first_hit_rank = rank

    expected_count = len(case.expected_sources)
    retrieved_count = len(results)

    recall = hits / expected_count if expected_count > 0 else 0.0
    precision = hits / retrieved_count if retrieved_count > 0 else 0.0
    mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        hits=hits,
        expected_count=expected_count,
        retrieved_count=retrieved_count,
    )


def aggregate_metrics(per_case: list[RetrievalMetrics]) -> AggregateMetrics:
    """Compute mean metrics across multiple cases."""
    if not per_case:
        return AggregateMetrics()

    n = len(per_case)
    return AggregateMetrics(
        mean_recall=sum(m.recall_at_k for m in per_case) / n,
        mean_precision=sum(m.precision_at_k for m in per_case) / n,
        mean_mrr=sum(m.mrr for m in per_case) / n,
        total_cases=n,
        per_case=per_case,
    )


def _matches_any_expected(
    chunk: RetrievedChunk,
    expected: list[ExpectedSource],
) -> bool:
    """Check if a retrieved chunk matches any expected source.

    Matching logic:
    - document_path: the chunk's path must contain the expected path
      (supports both exact and partial match)
    - heading_path: if specified, the chunk's heading must contain it
    """
    for source in expected:
        path_match = source.document_path in chunk.document_path
        if not path_match:
            continue
        if source.heading_path:
            if source.heading_path in chunk.heading_path:
                return True
        else:
            return True
    return False
