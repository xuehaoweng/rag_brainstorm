from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    path: Path
    relative_path: str
    content: str
    content_hash: str
    modified_time: float
    size: int


@dataclass(frozen=True, slots=True)
class MarkdownChunk:
    document_path: str
    chunk_index: int
    heading_path: str
    start_line: int
    end_line: int
    text: str
    text_hash: str

