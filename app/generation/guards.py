from __future__ import annotations

import re

from app.generation.prompt import Citation

EMPTY_EVIDENCE_ANSWER = "当前知识库证据不足。"
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def validate_answer(answer: str, citations: list[Citation]) -> tuple[str, bool]:
    cleaned = answer.strip()
    if not cleaned:
        return EMPTY_EVIDENCE_ANSWER, False
    if not citations:
        return EMPTY_EVIDENCE_ANSWER, False

    citation_ids = {citation.source_id for citation in citations}
    cited_ids = {int(match) for match in _CITATION_PATTERN.findall(cleaned)}
    if not cited_ids:
        return EMPTY_EVIDENCE_ANSWER, False
    if not cited_ids.issubset(citation_ids):
        return EMPTY_EVIDENCE_ANSWER, False

    return cleaned, True
