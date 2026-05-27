"""Multi-query rewriting: expand a single user query into several
reformulations so the retriever can cast a wider net.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Protocol

log = logging.getLogger(__name__)

_NUM_REWRITES = 3

_SYSTEM_PROMPT = (
    "你是一个搜索查询改写助手。"
    "你的任务是将用户的问题改写为多个不同角度的检索查询，"
    "以便在知识库中召回更多相关文档。"
)

_USER_PROMPT_TEMPLATE = (
    "请将以下问题改写为 {n} 个不同角度的检索查询。\n"
    "要求：\n"
    "- 每行一个查询，不要编号，不要加前缀\n"
    "- 使用不同的关键词和表述方式\n"
    "- 覆盖同义词、上位概念或相关术语\n"
    "- 保持与原始问题相同的语言（中文问题用中文改写，英文用英文）\n"
    "- 只输出查询，不要解释\n\n"
    "原始问题：{query}"
)


class LLMGenerateFunc(Protocol):
    """Minimal interface — anything with a generate(system_prompt, user_prompt) method."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


def rewrite_query(
    query: str,
    llm: LLMGenerateFunc,
    *,
    num_rewrites: int = _NUM_REWRITES,
) -> list[str]:
    """Return *[original_query, rewrite_1, rewrite_2, ...]*.  On any
    failure the list degrades gracefully to just ``[query]``."""

    user_prompt = _USER_PROMPT_TEMPLATE.format(n=num_rewrites, query=query)

    log.info("[multi-query] original query: %r, requesting %d rewrites", query, num_rewrites)
    t0 = perf_counter()

    try:
        raw = llm.generate(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)
    except Exception:
        log.warning("Query rewrite failed, falling back to original query", exc_info=True)
        return [query]

    elapsed = perf_counter() - t0
    rewrites = _parse_rewrites(raw)
    if not rewrites:
        log.warning("Query rewrite returned no usable lines (%.2fs), raw=%r", elapsed, raw)
        return [query]

    result = [query] + rewrites[:num_rewrites]
    log.info(
        "[multi-query] LLM rewrite done in %.2fs, %d queries: %s",
        elapsed,
        len(result),
        " | ".join(result),
    )
    # Always keep the original query first.
    return result


def _parse_rewrites(raw: str) -> list[str]:
    """Extract non-empty lines, stripping optional numbering like '1.' or '1)'."""
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering: "1. ", "1) ", "- ", "* "
        line = re.sub(r"^[\d]+[.)\-]\s*", "", line)
        line = re.sub(r"^[-*]\s+", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines
