from pathlib import Path

import pytest

from encourage.utils import MarkdownIngestion


def test_load_file_extracts_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text(
        "---\nsource: paddledoc\npages: 3\nprofile: tiny\n---\n\n# Title\n\nHello world.",
        encoding="utf-8",
    )

    loader = MarkdownIngestion()
    doc = loader.load_file(path)

    assert doc.content.startswith("# Title")
    assert doc.meta_data["fm_source"] == "paddledoc"
    assert doc.meta_data["fm_pages"] == 3
    assert doc.meta_data["filename"] == "sample.md"


def test_load_directory_collects_markdown_only(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "a.md").write_text("A", encoding="utf-8")
    (docs_dir / "b.md").write_text("B", encoding="utf-8")
    (docs_dir / "ignore.txt").write_text("C", encoding="utf-8")

    loader = MarkdownIngestion()
    docs = loader.load_directory(docs_dir)

    assert len(docs) == 2
    assert {doc.meta_data["filename"] for doc in docs} == {"a.md", "b.md"}


def test_load_missing_path_raises() -> None:
    loader = MarkdownIngestion()
    with pytest.raises(FileNotFoundError):
        loader.load("/does/not/exist")
