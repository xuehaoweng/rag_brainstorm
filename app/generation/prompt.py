from dataclasses import dataclass

from app.retrieval.schemas import RetrievedChunk


@dataclass(frozen=True, slots=True)
class Citation:
    source_id: int
    document_path: str
    heading_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    context: str
    citations: list[Citation]


def build_answer_prompt(query: str, chunks: list[RetrievedChunk]) -> PromptBundle:
    citations = [
        Citation(
            source_id=index,
            document_path=chunk.document_path,
            heading_path=chunk.heading_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(
        _format_source(index=index, chunk=chunk)
        for index, chunk in enumerate(chunks, start=1)
    )
    system_prompt = (
        "你是一个严谨的 RAG 知识库问答助手。"
        "只能基于用户提供的资料回答；资料不足时明确说“当前知识库证据不足”。"
        "回答中的关键结论必须用 [来源编号] 标注引用，例如 [1]。"
        "不要编造资料中没有的信息。"
    )
    user_prompt = (
        "请基于以下资料回答问题。\n\n"
        f"{context or '没有检索到可用资料。'}\n\n"
        f"问题：{query}"
    )
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context=context,
        citations=citations,
    )


def _format_source(index: int, chunk: RetrievedChunk) -> str:
    heading = f" > {chunk.heading_path}" if chunk.heading_path else ""
    return (
        f"[来源 {index}] {chunk.document_path}:{chunk.start_line}-{chunk.end_line}{heading}\n"
        f"{chunk.text}"
    )
