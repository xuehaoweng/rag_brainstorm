"""Evaluation dataset schema and loader."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ExpectedSource(BaseModel):
    """A document chunk that should be retrieved for a given query."""
    document_path: str = Field(description="Relative path of the expected source file")
    heading_path: str = Field(default="", description="Optional heading path within the document")


class EvalCase(BaseModel):
    """A single evaluation case: a query with its ground truth."""
    query: str = Field(description="The user question")
    expected_sources: list[ExpectedSource] = Field(
        description="Document chunks that should appear in retrieval results"
    )
    expected_answer_contains: list[str] = Field(
        default_factory=list,
        description="Keywords the final answer should contain (for answer quality check)",
    )


class EvalDataset(BaseModel):
    """A collection of evaluation cases."""
    name: str = Field(default="default")
    cases: list[EvalCase]


def load_eval_dataset(path: Path) -> EvalDataset:
    """Load an evaluation dataset from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(data)
