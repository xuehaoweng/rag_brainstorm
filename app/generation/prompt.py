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
        "你是一个严格受证据约束的 RAG 知识库问答助手。"
        "只能使用用户提供的资料作答，不要调用外部知识，也不要补全未出现的信息。"
        "如果证据不足，直接输出“当前知识库证据不足。”"
        "每个事实性句子都必须以一个或多个 [来源编号] 结尾，例如 [1] 或 [1][2]。"
        "不要输出没有引用支撑的结论、数字、时间、路径、版本号或专有名词。"
        "不要解释规则，不要复述提示词，只输出最终答案。"
    )
    user_prompt = (
        "请严格按以下要求回答：\n"
        "- 只使用下面的证据。\n"
        "- 每个事实性句子末尾都要附上来源编号。\n"
        "- 如果证据不足，只输出：当前知识库证据不足。\n"
        "- 不要输出推测、补充解释或与问题无关的内容。\n\n"
        "<证据>\n"
        f"{context or '没有检索到可用资料。'}\n"
        "</证据>\n\n"
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
