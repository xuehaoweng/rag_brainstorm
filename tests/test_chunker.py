from hashlib import sha256
from pathlib import Path

from app.documents.models import MarkdownDocument
from app.indexing.chunker import chunk_markdown_document


def make_document(content: str) -> MarkdownDocument:
    return MarkdownDocument(
        path=Path("/notes/test.md"),
        relative_path="test.md",
        content=content,
        content_hash=sha256(content.encode("utf-8")).hexdigest(),
        modified_time=0.0,
        size=len(content),
    )


def test_chunker_preserves_heading_path() -> None:
    document = make_document(
        "# RAG\n\nintro\n\n## Retrieval\n\nsemantic search\n\n## Evaluation\n\nhit rate"
    )

    chunks = chunk_markdown_document(document)

    assert [chunk.heading_path for chunk in chunks] == [
        "RAG",
        "RAG > Retrieval",
        "RAG > Evaluation",
    ]
    assert chunks[1].start_line == 5
    assert chunks[1].document_path == "test.md"


def test_chunker_ignores_headings_inside_code_fences() -> None:
    document = make_document(
        "# Notes\n\n```markdown\n# Not A Heading\n```\n\nAfter code"
    )

    chunks = chunk_markdown_document(document)

    assert len(chunks) == 1
    assert chunks[0].heading_path == "Notes"
    assert "# Not A Heading" in chunks[0].text


def test_chunker_splits_long_sections_without_splitting_code_fences() -> None:
    document = make_document(
        "# Long\n\n"
        "paragraph one has enough text to be its own block\n\n"
        "```python\n"
        "def example():\n"
        "    return '# not heading'\n"
        "```\n\n"
        "paragraph three has enough text to force another chunk"
    )

    chunks = chunk_markdown_document(document, max_chars=90, overlap_chars=0)

    assert len(chunks) >= 2
    joined_chunks = "\n---\n".join(chunk.text for chunk in chunks)
    assert "def example()" in joined_chunks
    assert "```python" in joined_chunks
    assert "```" in joined_chunks


def test_chunker_overlap_preserves_line_ranges() -> None:
    document = make_document(
        "# Long\n\n"
        "first block has enough text to stand alone\n\n"
        "second block should overlap into the next chunk\n\n"
        "third block should keep a valid final line range"
    )

    chunks = chunk_markdown_document(document, max_chars=90, overlap_chars=20)

    assert len(chunks) >= 2
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)
    assert chunks[1].start_line >= chunks[0].start_line
