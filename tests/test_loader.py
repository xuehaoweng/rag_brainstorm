from pathlib import Path

import pytest

from app.documents.loader import scan_markdown_folder


def test_scan_markdown_folder_loads_markdown_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nalpha", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.md").write_text("# B\n\nbeta", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignore", encoding="utf-8")

    documents = scan_markdown_folder(tmp_path)

    assert [document.relative_path for document in documents] == ["a.md", "nested/b.md"]
    assert all(document.content_hash for document in documents)


def test_scan_markdown_folder_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_markdown_folder(tmp_path / "missing")

