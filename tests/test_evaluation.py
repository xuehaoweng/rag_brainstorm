import json
from pathlib import Path

from app.evaluation.dataset import EvalDataset, load_eval_dataset
from app.evaluation.metrics import (
    aggregate_metrics,
    compute_retrieval_metrics,
)
from app.evaluation.dataset import EvalCase, ExpectedSource
from app.evaluation.runner import run_eval
from app.retrieval.schemas import RetrievedChunk


def _make_chunk(doc_path: str, heading: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=1.0 - rank * 0.1,
        document_path=doc_path,
        chunk_index=0,
        heading_path=heading,
        start_line=1,
        end_line=5,
        text="some text",
        text_hash="hash",
    )


# --- Dataset tests ---

def test_load_eval_dataset(tmp_path: Path) -> None:
    data = {
        "name": "test",
        "cases": [
            {
                "query": "what is RAG?",
                "expected_sources": [{"document_path": "rag.md", "heading_path": "RAG"}],
                "expected_answer_contains": ["retrieval"],
            }
        ],
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    dataset = load_eval_dataset(path)
    assert dataset.name == "test"
    assert len(dataset.cases) == 1
    assert dataset.cases[0].query == "what is RAG?"


# --- Metrics tests ---

def test_recall_at_k_perfect() -> None:
    case = EvalCase(
        query="test",
        expected_sources=[ExpectedSource(document_path="rag.md", heading_path="RAG")],
    )
    results = [_make_chunk("rag.md", "RAG", rank=1)]
    m = compute_retrieval_metrics(case, results)
    assert m.recall_at_k == 1.0
    assert m.precision_at_k == 1.0
    assert m.mrr == 1.0


def test_recall_at_k_partial() -> None:
    case = EvalCase(
        query="test",
        expected_sources=[
            ExpectedSource(document_path="rag.md", heading_path="RAG"),
            ExpectedSource(document_path="deploy.md", heading_path="部署"),
        ],
    )
    results = [
        _make_chunk("other.md", "Other", rank=1),
        _make_chunk("rag.md", "RAG", rank=2),
    ]
    m = compute_retrieval_metrics(case, results)
    assert m.recall_at_k == 0.5  # 1 of 2 found
    assert m.precision_at_k == 0.5  # 1 of 2 results relevant
    assert m.mrr == 0.5  # first hit at rank 2


def test_recall_at_k_zero() -> None:
    case = EvalCase(
        query="test",
        expected_sources=[ExpectedSource(document_path="rag.md", heading_path="RAG")],
    )
    results = [_make_chunk("other.md", "Other", rank=1)]
    m = compute_retrieval_metrics(case, results)
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0


def test_mrr_first_hit_at_rank_3() -> None:
    case = EvalCase(
        query="test",
        expected_sources=[ExpectedSource(document_path="rag.md", heading_path="")],
    )
    results = [
        _make_chunk("a.md", "A", rank=1),
        _make_chunk("b.md", "B", rank=2),
        _make_chunk("rag.md", "RAG", rank=3),
    ]
    m = compute_retrieval_metrics(case, results)
    assert abs(m.mrr - 1 / 3) < 1e-9


def test_aggregate_metrics() -> None:
    case1 = EvalCase(
        query="q1",
        expected_sources=[ExpectedSource(document_path="a.md", heading_path="")],
    )
    case2 = EvalCase(
        query="q2",
        expected_sources=[ExpectedSource(document_path="b.md", heading_path="")],
    )
    m1 = compute_retrieval_metrics(case1, [_make_chunk("a.md", "", 1)])
    m2 = compute_retrieval_metrics(case2, [_make_chunk("other.md", "", 1)])

    agg = aggregate_metrics([m1, m2])
    assert agg.mean_recall == 0.5  # (1.0 + 0.0) / 2
    assert agg.total_cases == 2


# --- Runner tests ---

def test_run_eval_integration(tmp_path: Path) -> None:
    data = {
        "name": "integration-test",
        "cases": [
            {
                "query": "RAG 是什么？",
                "expected_sources": [{"document_path": "rag.md", "heading_path": ""}],
                "expected_answer_contains": ["检索"],
            },
            {
                "query": "向量检索",
                "expected_sources": [{"document_path": "vector.md", "heading_path": ""}],
            },
        ],
    }
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(data), encoding="utf-8")

    # Simulate retrieval that always returns rag.md
    def fake_retrieval(query: str) -> list[RetrievedChunk]:
        return [_make_chunk("rag.md", "RAG", 1)]

    def fake_answer(query: str) -> str:
        return "RAG 是检索增强生成"

    report = run_eval(eval_path, fake_retrieval, answer_fn=fake_answer)

    assert report.dataset_name == "integration-test"
    assert report.total_cases == 2
    assert report.mean_recall == 0.5  # hits rag.md but misses vector.md
    assert report.cases[0].recall_at_k == 1.0
    assert report.cases[1].recall_at_k == 0.0
    assert report.answer_keyword_accuracy == 1.0  # "检索" found in answer


def test_partial_match_document_path() -> None:
    """Expected path 'rag.md' should match chunk path 'docs/notes/rag.md'."""
    case = EvalCase(
        query="test",
        expected_sources=[ExpectedSource(document_path="rag.md", heading_path="")],
    )
    results = [_make_chunk("docs/notes/rag.md", "Section", rank=1)]
    m = compute_retrieval_metrics(case, results)
    assert m.recall_at_k == 1.0
