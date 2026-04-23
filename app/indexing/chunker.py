from hashlib import sha256
from itertools import count

from app.documents.models import MarkdownChunk, MarkdownDocument


MAX_CHARS = 1_800
OVERLAP_CHARS = 160


def chunk_markdown_document(
    document: MarkdownDocument,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[MarkdownChunk]:
    """Split a Markdown document into structure-aware chunks."""

    sections = _split_sections(document.content)
    chunks: list[MarkdownChunk] = []
    indexes = count()

    for section in sections:
        for text, start_line, end_line in _split_section_text(
            section.lines,
            section.start_line,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            cleaned = text.strip()
            if not cleaned:
                continue

            chunks.append(
                MarkdownChunk(
                    document_path=document.relative_path,
                    chunk_index=next(indexes),
                    heading_path=" > ".join(section.heading_path),
                    start_line=start_line,
                    end_line=end_line,
                    text=cleaned,
                    text_hash=sha256(cleaned.encode("utf-8")).hexdigest(),
                )
            )

    return chunks


class _Section:
    __slots__ = ("heading_path", "lines", "start_line")

    def __init__(self, heading_path: list[str], lines: list[str], start_line: int) -> None:
        self.heading_path = heading_path
        self.lines = lines
        self.start_line = start_line


def _split_sections(content: str) -> list[_Section]:
    lines = content.splitlines()
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading_path: list[str] = ["Document"]
    current_start_line = 1
    in_code_fence = False

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            sections.append(
                _Section(
                    heading_path=current_heading_path.copy(),
                    lines=current_lines,
                    start_line=current_start_line,
                )
            )
            current_lines = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence

        heading = None if in_code_fence else _parse_heading(line)
        if heading is not None:
            level, title = heading
            flush()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading_path = [title for _, title in heading_stack]
            current_start_line = line_number
            current_lines = [line]
            continue

        if not current_lines:
            current_start_line = line_number
        current_lines.append(line)

    flush()
    return sections


def _parse_heading(line: str) -> tuple[int, str] | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None

    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes > 6 or len(stripped) <= hashes or stripped[hashes] != " ":
        return None

    title = stripped[hashes:].strip()
    return (hashes, title) if title else None


def _split_section_text(
    lines: list[str],
    start_line: int,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return [(text, start_line, start_line + len(lines) - 1)]

    blocks = _paragraph_blocks(lines, start_line)
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []

    for block, block_start, block_end in blocks:
        candidate = _join_blocks([*current, (block, block_start, block_end)])
        if current and len(candidate) > max_chars:
            chunks.append(_chunk_from_blocks(current))
            current = _overlap_blocks(current, overlap_chars)

        current.append((block, block_start, block_end))

    if current:
        chunks.append(_chunk_from_blocks(current))

    return chunks


def _paragraph_blocks(lines: list[str], start_line: int) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_start = start_line
    in_code_fence = False

    for offset, line in enumerate(lines):
        line_number = start_line + offset
        stripped = line.strip()
        fence_start = stripped.startswith("```") or stripped.startswith("~~~")

        if not current:
            current_start = line_number

        if fence_start:
            in_code_fence = not in_code_fence

        if not in_code_fence and not stripped:
            if current:
                blocks.append(("\n".join(current), current_start, line_number - 1))
                current = []
            continue

        current.append(line)

    if current:
        blocks.append(("\n".join(current), current_start, start_line + len(lines) - 1))

    return blocks


def _join_blocks(blocks: list[tuple[str, int, int]]) -> str:
    return "\n\n".join(block for block, _, _ in blocks)


def _chunk_from_blocks(blocks: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    return (_join_blocks(blocks), blocks[0][1], blocks[-1][2])


def _overlap_blocks(
    blocks: list[tuple[str, int, int]],
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    if overlap_chars <= 0:
        return []

    overlap: list[tuple[str, int, int]] = []
    total = 0
    for block in reversed(blocks):
        total += len(block[0])
        overlap.insert(0, block)
        if total >= overlap_chars:
            break
    return overlap
