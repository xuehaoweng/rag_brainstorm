from hashlib import sha256
from pathlib import Path

from app.documents.models import MarkdownDocument


def scan_markdown_folder(root: Path) -> list[MarkdownDocument]:
    """Load Markdown files from a folder recursively."""

    resolved_root = root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Markdown root does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Markdown root is not a directory: {resolved_root}")

    documents: list[MarkdownDocument] = []
    for path in sorted(resolved_root.rglob("*.md")):
        if not path.is_file():
            continue

        content = path.read_text(encoding="utf-8")
        stat = path.stat()
        relative_path = path.relative_to(resolved_root).as_posix()
        documents.append(
            MarkdownDocument(
                path=path,
                relative_path=relative_path,
                content=content,
                content_hash=sha256(content.encode("utf-8")).hexdigest(),
                modified_time=stat.st_mtime,
                size=stat.st_size,
            )
        )

    return documents

