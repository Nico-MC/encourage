"""Helpers to ingest Markdown files into encourage Document objects."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, cast

import yaml
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    _HAS_LANGCHAIN_SPLITTERS = True
except ImportError:
    MarkdownHeaderTextSplitter = None
    RecursiveCharacterTextSplitter = None
    _HAS_LANGCHAIN_SPLITTERS = False

from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData


class MarkdownIngestion:
    """Load PaddleDoc-generated markdown files and convert them into encourage Documents.

    Typical source paths are files under the PaddleDoc repository, e.g.
    ``../PaddleDoc/backend/storage/results/.../*.md``.

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
        chunk_documents: bool = False,
        chunk_max_chars: int = 1200,
        chunk_overlap_chars: int = 150,
    ) -> None:
        self.encoding = encoding
        self.pattern = pattern
        self.recursive = recursive
        self.extract_frontmatter = extract_frontmatter
        self.keep_frontmatter_in_content = keep_frontmatter_in_content
        self.chunk_documents = chunk_documents
        self.chunk_max_chars = max(200, chunk_max_chars)
        self.chunk_overlap_chars = max(0, min(chunk_overlap_chars, self.chunk_max_chars // 2))

    def load(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> list[Document]:
        """Load one PaddleDoc markdown file or all markdown files from a PaddleDoc output directory."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")

        if resolved.is_file():
            doc = self.load_file(resolved, extra_meta=extra_meta)
            return self._chunk_document(doc) if self.chunk_documents else [doc]

        return self.load_directory(resolved, extra_meta=extra_meta)

    def load_directory(
        self,
        directory: str | Path,
        extra_meta: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Load PaddleDoc markdown files from a directory into Documents."""
        root = Path(directory).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        iterator = root.rglob(self.pattern) if self.recursive else root.glob(self.pattern)
        documents: list[Document] = []
        for file_path in sorted(path for path in iterator if path.is_file()):
            doc = self.load_file(file_path, base_dir=root, extra_meta=extra_meta)
            if self.chunk_documents:
                documents.extend(self._chunk_document(doc))
            else:
                documents.append(doc)
        return documents

    def load_file(
        self,
        file_path: str | Path,
        *,
        base_dir: str | Path | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Document:
        """Load a single PaddleDoc-generated markdown file into a Document."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"Expected markdown file, got: {path.suffix}")

        raw = path.read_text(encoding=self.encoding)
        frontmatter, content = self._extract_frontmatter(raw)
        document_content = raw if self.keep_frontmatter_in_content else content

        metadata_tags: dict[str, Any] = {
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
        if content.startswith("---\n"):
            content = content[4:].lstrip("\n")

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

    @staticmethod
    def _contains_markdown_table(block: str) -> bool:
        normalized = block.strip()
        if "|" not in normalized:
            return False

        # Typical markdown table format with header and divider lines.
        table_pattern = r"^\s*\|.+\|\s*$\n\s*\|\s*[-:| ]+\|\s*$"
        return bool(re.search(table_pattern, normalized, flags=re.MULTILINE))

    @staticmethod
    def _split_blocks(content: str) -> list[str]:
        normalized = content.replace("\r\n", "\n").strip()
        if not normalized:
            return []
        return [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]

    def _split_markdown_sections(self, content: str) -> list[str]:
        if not _HAS_LANGCHAIN_SPLITTERS:
            return [content.strip()]

        splitter_cls = cast(Any, MarkdownHeaderTextSplitter)
        splitter = splitter_cls(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )
        sections = [doc.page_content.strip() for doc in splitter.split_text(content) if doc.page_content.strip()]
        return sections or [content.strip()]

    def _split_large_block(self, block: str, *, overlap: int) -> list[str]:
        if not _HAS_LANGCHAIN_SPLITTERS:
            # Fallback: fixed-size chunking with optional overlap.
            step = max(1, self.chunk_max_chars - overlap)
            parts: list[str] = []
            for idx in range(0, len(block), step):
                part = block[idx : idx + self.chunk_max_chars].strip()
                if part:
                    parts.append(part)
            return parts

        splitter_cls = cast(Any, RecursiveCharacterTextSplitter)
        splitter = splitter_cls(
            chunk_size=self.chunk_max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return [part.strip() for part in splitter.split_text(block) if part.strip()]

    def _chunk_document(self, document: Document) -> list[Document]:
        content = document.content.strip()
        if not content:
            return []

        chunk_texts: list[str] = []

        for section in self._split_markdown_sections(content):
            blocks = self._split_blocks(section)
            if not blocks:
                continue

            for block in blocks:
                if len(block) <= self.chunk_max_chars:
                    chunk_texts.append(block)
                    continue

                overlap = 0 if self._contains_markdown_table(block) else self.chunk_overlap_chars
                chunk_texts.extend(self._split_large_block(block, overlap=overlap))

        if len(chunk_texts) <= 1:
            return [document]

        source_id = str(document.id)
        base_tags: dict[str, Any] = dict(document.meta_data.to_dict(truncated=False))
        chunk_count = len(chunk_texts)
        chunked_documents: list[Document] = []

        for chunk_index, chunk_text in enumerate(chunk_texts):
            tags: dict[str, Any] = dict(base_tags)
            tags.update(
                {
                    "source_document_id": source_id,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                }
            )

            chunked_documents.append(
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}::chunk::{chunk_index}"),
                    content=chunk_text,
                    score=document.score,
                    distance=document.distance,
                    meta_data=MetaData(tags=tags),
                )
            )

        return chunked_documents
