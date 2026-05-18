"""Evaluation runner: execute an eval dataset against the retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvalCase, EvalDataset, load_eval_dataset
from app.evaluation.metrics import (
    AggregateMetrics,
    RetrievalMetrics,
    aggregate_metrics,
    compute_retrieval_metrics,
)
from app.retrieval.schemas import RetrievedChunk


class EvalCaseResult(BaseModel):
    query: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hits: int
    expected_count: int
    retrieved_count: int
    answer_keyword_hits: list[str] = Field(default_factory=list)
    answer_keyword_misses: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    dataset_name: str
    total_cases: int
    mean_recall: float
    mean_precision: float
    mean_mrr: float
    answer_keyword_accuracy: float
    cases: list[EvalCaseResult]


def run_eval(
    dataset_path: Path,
    retrieval_fn,
    *,
    answer_fn=None,
) -> EvalReport:
    """Run evaluation.

    Args:
        dataset_path: Path to eval JSON file.
        retrieval_fn: Callable(query: str) -> list[RetrievedChunk]
        answer_fn: Optional callable(query: str) -> str  (for answer quality check)
    """
    dataset = load_eval_dataset(dataset_path)
    per_case_metrics: list[RetrievalMetrics] = []
    case_results: list[EvalCaseResult] = []

    total_keywords = 0
    hit_keywords = 0

    for case in dataset.cases:
        results = retrieval_fn(case.query)
        metrics = compute_retrieval_metrics(case, results)
        per_case_metrics.append(metrics)

        # Answer quality check (optional)
        keyword_hits: list[str] = []
        keyword_misses: list[str] = []
        if answer_fn and case.expected_answer_contains:
            answer = answer_fn(case.query)
            for kw in case.expected_answer_contains:
                total_keywords += 1
                if kw in answer:
                    keyword_hits.append(kw)
                    hit_keywords += 1
                else:
                    keyword_misses.append(kw)

        case_results.append(
            EvalCaseResult(
                query=case.query,
                recall_at_k=metrics.recall_at_k,
                precision_at_k=metrics.precision_at_k,
                mrr=metrics.mrr,
                hits=metrics.hits,
                expected_count=metrics.expected_count,
                retrieved_count=metrics.retrieved_count,
                answer_keyword_hits=keyword_hits,
                answer_keyword_misses=keyword_misses,
            )
        )

    agg = aggregate_metrics(per_case_metrics)
    keyword_accuracy = hit_keywords / total_keywords if total_keywords > 0 else 0.0

    return EvalReport(
        dataset_name=dataset.name,
        total_cases=agg.total_cases,
        mean_recall=round(agg.mean_recall, 4),
        mean_precision=round(agg.mean_precision, 4),
        mean_mrr=round(agg.mean_mrr, 4),
        answer_keyword_accuracy=round(keyword_accuracy, 4),
        cases=case_results,
    )
