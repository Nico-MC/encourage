"""Helpers to ingest Markdown files into encourage Document objects."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml

from encourage.prompts import Document, MetaData


class MarkdownIngestion:
    """Load markdown files from disk and convert them into encourage Documents.

    This class keeps the integration point intentionally small:
    - A single markdown file can be converted into one ``Document``.
    - A directory can be scanned to convert all ``.md`` files.
    - YAML frontmatter can be extracted into flat metadata tags.
    """

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
        pattern: str = "*.md",
        recursive: bool = True,
        extract_frontmatter: bool = True,
        keep_frontmatter_in_content: bool = False,
    ) -> None:
        self.encoding = encoding
        self.pattern = pattern
        self.recursive = recursive
        self.extract_frontmatter = extract_frontmatter
        self.keep_frontmatter_in_content = keep_frontmatter_in_content

    def load(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> list[Document]:
        """Load one markdown file or all markdown files from a directory."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")

        if resolved.is_file():
            return [self.load_file(resolved, extra_meta=extra_meta)]

        return self.load_directory(resolved, extra_meta=extra_meta)

    def load_directory(
        self,
        directory: str | Path,
        extra_meta: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Load markdown files from a directory into Documents."""
        root = Path(directory).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        iterator = root.rglob(self.pattern) if self.recursive else root.glob(self.pattern)
        documents: list[Document] = []
        for file_path in sorted(path for path in iterator if path.is_file()):
            documents.append(self.load_file(file_path, base_dir=root, extra_meta=extra_meta))
        return documents

    def load_file(
        self,
        file_path: str | Path,
        *,
        base_dir: str | Path | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Document:
        """Load a single markdown file into a Document."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"Expected markdown file, got: {path.suffix}")

        raw = path.read_text(encoding=self.encoding)
        frontmatter, content = self._extract_frontmatter(raw)
        document_content = raw if self.keep_frontmatter_in_content else content

        metadata_tags = {
            "source": "markdown",
            "loader": self.__class__.__name__,
            "filename": path.name,
            "filepath": str(path),
        }

        if base_dir is not None:
            base = Path(base_dir).expanduser().resolve()
            try:
                metadata_tags["relative_path"] = str(path.relative_to(base))
            except ValueError:
                metadata_tags["relative_path"] = path.name

        if self.extract_frontmatter:
            metadata_tags.update(self._flat_frontmatter_tags(frontmatter))

        if extra_meta:
            metadata_tags.update({key: self._normalize_meta_value(value) for key, value in extra_meta.items()})

        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, str(path))
        return Document(
            id=doc_id,
            content=document_content,
            meta_data=MetaData(tags=metadata_tags),
        )

    def _extract_frontmatter(self, raw_text: str) -> tuple[dict[str, Any], str]:
        if not self.extract_frontmatter or not raw_text.startswith("---\n"):
            return {}, raw_text

        marker = "\n---\n"
        end = raw_text.find(marker, 4)
        if end == -1:
            return {}, raw_text

        frontmatter_text = raw_text[4:end]
        content = raw_text[end + len(marker) :].lstrip("\n")

        parsed = yaml.safe_load(frontmatter_text)
        if not isinstance(parsed, dict):
            return {}, content
        return parsed, content

    def _flat_frontmatter_tags(self, frontmatter: dict[str, Any]) -> dict[str, str | int | float | bool]:
        tags: dict[str, str | int | float | bool] = {}
        for key, value in frontmatter.items():
            tags[f"fm_{key}"] = self._normalize_meta_value(value)
        return tags

    @staticmethod
    def _normalize_meta_value(value: Any) -> str | int | float | bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
