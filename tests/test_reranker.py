from app.retrieval.reranker import rerank, _parse_scores
from app.retrieval.schemas import RetrievedChunk


def _make_chunk(index: int, text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        rank=index,
        score=score,
        document_path=f"doc{index}.md",
        chunk_index=index,
        heading_path=f"Section {index}",
        start_line=1,
        end_line=3,
        text=text,
        text_hash=f"hash{index}",
    )


class FakeRerankLLM:
    """Returns scores in bracket format: highest for chunk 3, lowest for chunk 1."""
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "[1] 3\n[2] 7\n[3] 9"


class NumberOnlyLLM:
    """Returns scores without bracket format — tests fallback parsing."""
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "3\n7\n9"


class FailingLLM:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM down")


class EmptyLLM:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "\n  \n"


def test_rerank_reorders_by_score() -> None:
    chunks = [
        _make_chunk(1, "部署相关但不精确的内容"),
        _make_chunk(2, "比较相关的部署文档"),
        _make_chunk(3, "非常精确的部署指南"),
    ]

    result = rerank("怎么部署？", chunks, FakeRerankLLM())

    assert len(result) == 3
    # Chunk 3 scored 9 → rank 1
    assert result[0].document_path == "doc3.md"
    assert result[0].rank == 1
    assert result[0].score == 9.0
    # Chunk 2 scored 7 → rank 2
    assert result[1].document_path == "doc2.md"
    assert result[1].rank == 2
    # Chunk 1 scored 3 → rank 3
    assert result[2].document_path == "doc1.md"
    assert result[2].rank == 3


def test_rerank_respects_top_k() -> None:
    chunks = [
        _make_chunk(1, "内容 1"),
        _make_chunk(2, "内容 2"),
        _make_chunk(3, "内容 3"),
    ]

    result = rerank("问题", chunks, FakeRerankLLM(), top_k=2)

    assert len(result) == 2
    assert result[0].document_path == "doc3.md"  # highest score
    assert result[1].document_path == "doc2.md"


def test_rerank_fallback_on_failure() -> None:
    chunks = [
        _make_chunk(1, "内容 1"),
        _make_chunk(2, "内容 2"),
    ]

    result = rerank("问题", chunks, FailingLLM())

    # Returns original order unchanged
    assert len(result) == 2
    assert result[0].document_path == "doc1.md"
    assert result[1].document_path == "doc2.md"


def test_rerank_fallback_on_empty_response() -> None:
    chunks = [
        _make_chunk(1, "内容 1"),
        _make_chunk(2, "内容 2"),
    ]

    result = rerank("问题", chunks, EmptyLLM())

    assert len(result) == 2
    assert result[0].document_path == "doc1.md"


def test_rerank_empty_chunks() -> None:
    result = rerank("问题", [], FakeRerankLLM())
    assert result == []


def test_rerank_fallback_number_only_format() -> None:
    chunks = [
        _make_chunk(1, "内容 1"),
        _make_chunk(2, "内容 2"),
        _make_chunk(3, "内容 3"),
    ]

    result = rerank("问题", chunks, NumberOnlyLLM())

    # Should still parse: line 1→3, line 2→7, line 3→9
    assert result[0].document_path == "doc3.md"
    assert result[0].score == 9.0


def test_parse_scores_bracket_format() -> None:
    raw = "[1] 8\n[2] 5\n[3] 9"
    scores = _parse_scores(raw, expected_count=3)
    assert scores == {1: 8, 2: 5, 3: 9}


def test_parse_scores_caps_at_10() -> None:
    raw = "[1] 15\n[2] 10"
    scores = _parse_scores(raw, expected_count=2)
    assert scores == {1: 10, 2: 10}


def test_parse_scores_ignores_out_of_range_index() -> None:
    raw = "[1] 8\n[99] 5"
    scores = _parse_scores(raw, expected_count=3)
    assert scores == {1: 8}


def test_parse_scores_fallback_line_numbers() -> None:
    raw = "8\n5\n9"
    scores = _parse_scores(raw, expected_count=3)
    assert scores == {1: 8, 2: 5, 3: 9}
