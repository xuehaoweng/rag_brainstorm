from pathlib import Path

from app.generation.prompt import build_answer_prompt
from app.main import AnswerRequest, answer_question
from app.retrieval.schemas import RetrievedChunk


class FakeLLMProvider:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        assert "严格受证据约束" in system_prompt
        assert "<证据>" in user_prompt
        assert "当前知识库证据不足" in system_prompt
        return "RAG 需要先检索证据再生成回答。[1]"


class NoCitationLLMProvider:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "RAG 需要先检索证据再生成回答。"


def _raise_if_called() -> None:
    raise AssertionError("LLM should not be called")


def test_build_answer_prompt_adds_source_citations() -> None:
    chunk = RetrievedChunk(
        rank=1,
        score=1.0,
        document_path="rag.md",
        chunk_index=0,
        heading_path="RAG",
        start_line=1,
        end_line=3,
        text="RAG 先检索，再生成。",
        text_hash="hash",
    )

    prompt = build_answer_prompt("RAG 是什么？", [chunk])

    assert "[来源 1] rag.md:1-3 > RAG" in prompt.context
    assert "<证据>" in prompt.user_prompt
    assert "问题：RAG 是什么？" in prompt.user_prompt
    assert prompt.citations[0].document_path == "rag.md"


def test_answer_question_returns_answer_with_citations(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "rag.md").write_text(
        "# RAG\n\nRAG 需要先检索证据再生成回答，避免没有依据的输出。",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.main.create_llm_provider", lambda: FakeLLMProvider())

    response = answer_question(
        AnswerRequest(
            root=tmp_path,
            query="RAG 怎么回答？",
            mode="keyword",
            top_k=1,
            embedding_provider="hash",
        )
    )

    assert response.answer == "RAG 需要先检索证据再生成回答。[1]"
    assert response.citations[0].document_path == "rag.md"
    assert response.retrieved_chunks[0].document_path == "rag.md"
    assert response.model == "fake-model"


def test_answer_question_returns_fallback_when_no_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.main.create_llm_provider", _raise_if_called)

    response = answer_question(
        AnswerRequest(
            root=tmp_path,
            query="RAG 怎么回答？",
            mode="keyword",
            top_k=1,
            embedding_provider="hash",
        )
    )

    assert response.answer == "当前知识库证据不足。"
    assert response.citations == []
    assert response.model == "not-called"
    assert response.provider == "not-called"


def test_answer_question_rejects_unquoted_llm_output(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "rag.md").write_text(
        "# RAG\n\nRAG 需要先检索证据再生成回答，避免没有依据的输出。",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.main.create_llm_provider", lambda: NoCitationLLMProvider())

    response = answer_question(
        AnswerRequest(
            root=tmp_path,
            query="RAG 怎么回答？",
            mode="keyword",
            top_k=1,
            embedding_provider="hash",
        )
    )

    assert response.answer == "当前知识库证据不足。"
    assert response.citations[0].document_path == "rag.md"
