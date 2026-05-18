"""Reranker: re-score retrieved chunks for more precise ranking.

Provides an LLM-based reranker that uses the existing LLM provider
to score (query, chunk) relevance in a single batch call.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from app.retrieval.schemas import RetrievedChunk

log = logging.getLogger(__name__)


class LLMGenerateFunc(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


_SYSTEM_PROMPT = (
    "你是一个文档相关性评估助手。"
    "给定一个用户问题和若干段候选文本，你需要评估每段文本与问题的相关程度。"
    "对每段文本输出一个 0-10 的整数分数（10 = 完全相关，0 = 完全无关）。"
    "只输出分数，每行一个，格式为 '[编号] 分数'，不要解释。"
)

_USER_PROMPT_TEMPLATE = (
    "用户问题：{query}\n\n"
    "请对以下每段文本评分（0-10）：\n\n"
    "{chunks_text}\n\n"
    "请按以下格式输出，每行一个：\n"
    "[1] 分数\n"
    "[2] 分数\n"
    "...\n"
    "只输出分数，不要解释。"
)

_SCORE_PATTERN = re.compile(r"\[(\d+)\]\s*(\d+)")


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMGenerateFunc,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Re-score *chunks* against *query* using the LLM, return sorted by
    relevance.  On failure, returns the original list unchanged."""

    if not chunks:
        return []

    if top_k is None:
        top_k = len(chunks)

    chunks_text = "\n\n".join(
        f"[文本 {i}]\n{chunk.text[:800]}"
        for i, chunk in enumerate(chunks, start=1)
    )

    user_prompt = _USER_PROMPT_TEMPLATE.format(query=query, chunks_text=chunks_text)

    try:
        raw = llm.generate(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)
    except Exception:
        log.warning("Reranker LLM call failed, keeping original order", exc_info=True)
        return chunks[:top_k]

    scores = _parse_scores(raw, expected_count=len(chunks))
    if not scores:
        log.warning("Reranker could not parse scores, raw=%r", raw)
        return chunks[:top_k]

    scored = []
    for i, chunk in enumerate(chunks):
        score = scores.get(i + 1, 0)
        scored.append((chunk, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    return [
        chunk.model_copy(update={"rank": rank, "score": float(score)})
        for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
    ]


def _parse_scores(raw: str, *, expected_count: int) -> dict[int, int]:
    """Parse '[N] score' lines from LLM output.

    Falls back to line-by-line number extraction if the bracket format
    is not found.
    """
    scores: dict[int, int] = {}

    # Try bracket format first: [1] 8
    for match in _SCORE_PATTERN.finditer(raw):
        idx = int(match.group(1))
        score = min(int(match.group(2)), 10)
        if 1 <= idx <= expected_count:
            scores[idx] = score

    if scores:
        return scores

    # Fallback: just grab numbers from each line
    for line_num, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        numbers = re.findall(r"\d+", line)
        if numbers:
            score = min(int(numbers[-1]), 10)  # take last number as score
            if line_num <= expected_count:
                scores[line_num] = score

    return scores
