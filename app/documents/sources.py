"""Multi-source document loaders for RAG indexing."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import time
from typing import Protocol

from app.documents.models import MarkdownDocument


class DocumentSource(Protocol):
    """Protocol for document sources that can be indexed."""

    def load(self) -> list[MarkdownDocument]:
        """Load documents from this source."""
        ...


class LocalMarkdownSource:
    """Load markdown files from local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> list[MarkdownDocument]:
        from app.documents.loader import scan_markdown_folder

        return scan_markdown_folder(self.root)


class WebPageSource:
    """Load web pages and convert to markdown documents."""

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls

    def load(self) -> list[MarkdownDocument]:
        documents = []
        for url in self.urls:
            try:
                content = self._fetch_and_convert(url)
                documents.append(
                    MarkdownDocument(
                        path=Path(f"web/{_url_to_filename(url)}"),
                        relative_path=url,
                        content=content,
                        content_hash=sha256(content.encode()).hexdigest(),
                        modified_time=time(),
                        size=len(content.encode()),
                    )
                )
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
        return documents

    def _fetch_and_convert(self, url: str) -> str:
        """Fetch URL and convert to markdown."""
        try:
            import trafilatura
        except ImportError as exc:
            raise RuntimeError(
                "trafilatura is not installed. Install with: pip install trafilatura"
            ) from exc

        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Extract main content
        text = trafilatura.extract(response.text)
        if not text:
            raise ValueError(f"Failed to extract content from {url}")

        # Add URL as title
        return f"# {url}\n\n{text}"


class LarkDocSource:
    """Load documents from Feishu/Lark workspace."""

    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids

    def load(self) -> list[MarkdownDocument]:
        """Load Lark documents via lark-cli."""
        documents = []
        for doc_id in self.doc_ids:
            try:
                content = self._fetch_lark_doc(doc_id)
                documents.append(
                    MarkdownDocument(
                        path=Path(f"lark/{doc_id}.md"),
                        relative_path=f"lark://{doc_id}",
                        content=content,
                        content_hash=sha256(content.encode()).hexdigest(),
                        modified_time=time(),
                        size=len(content.encode()),
                    )
                )
            except Exception as e:
                print(f"Failed to fetch lark doc {doc_id}: {e}")
        return documents

    def _fetch_lark_doc(self, doc_id: str) -> str:
        """Fetch Lark document content via CLI."""
        import subprocess

        result = subprocess.run(
            ["lark-cli", "doc", "+fetch", doc_id, "--format", "markdown"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout


def _url_to_filename(url: str) -> str:
    """Convert URL to safe filename."""
    import re

    # Remove protocol
    name = re.sub(r"^https?://", "", url)
    # Replace special chars with underscore
    name = re.sub(r"[^\w\-.]", "_", name)
    # Limit length
    return name[:200] + ".md"
