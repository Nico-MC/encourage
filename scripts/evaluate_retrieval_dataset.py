#!/usr/bin/env python3
"""Run a retrieval-only evaluation for a JSONL dataset.

The dataset is expected to contain one JSON object per line with at least:
- id
- question
- evidence_quote
- gold_answer
- source_document

Each row is evaluated by:
1. loading the source markdown file,
2. chunking it with Encourage's MarkdownIngestion,
3. mapping the evidence quote to one or more reference chunks,
4. retrieving top-k chunks for the question,
5. scoring the result with retrieval metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from encourage.llm import Response, ResponseWrapper
from encourage.metrics import HitRateAtK, MeanReciprocalRank, RecallAtK
from encourage.prompts import Context, Document, MetaData
from encourage.rag import BaseRAG
from encourage.rag.base.config import BaseRAGConfig
from encourage.utils.file_manager import FileManager
from encourage.utils.markdown_ingestion import MarkdownIngestion


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(text)))


def resolve_source_path(source_document: str, paddledoc_root: Path) -> Path:
    path = Path(source_document)
    if path.is_absolute():
        return path
    return (paddledoc_root / source_document).resolve()


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    return FileManager(dataset_path).load_jsonlines()


def pick_reference_documents(chunks: list[Document], evidence_quote: str) -> list[Document]:
    normalized_quote = normalize_text(evidence_quote)

    exact_matches = [
        chunk
        for chunk in chunks
        if normalized_quote and normalized_quote in normalize_text(chunk.content)
    ]
    if exact_matches:
        return exact_matches

    quote_tokens = tokenize(evidence_quote)
    if not quote_tokens:
        return [chunks[0]] if chunks else []

    best_score = 0.0
    best_chunks: list[Document] = []
    for chunk in chunks:
        chunk_tokens = tokenize(chunk.content)
        if not chunk_tokens:
            continue
        score = len(chunk_tokens & quote_tokens) / len(quote_tokens)
        if score > best_score:
            best_score = score
            best_chunks = [chunk]
        elif score == best_score and score > 0:
            best_chunks.append(chunk)

    if best_chunks:
        return best_chunks
    return [chunks[0]] if chunks else []


def build_rag(chunks: list[Document], collection_name: str, top_k: int, embedding_model: str) -> BaseRAG:
    config = BaseRAGConfig(
        context_collection=chunks,
        collection_name=collection_name,
        embedding_function=embedding_model,
        top_k=top_k,
        retrieval_only=True,
        runner=None,
        additional_prompt="",
        where=None,
        device="cpu",
        template_name="default.j2",
        batch_size_insert=1000,
        batch_size_query=200,
    )
    return BaseRAG(config)


def evaluate_group(
    rows: list[dict[str, Any]],
    *,
    paddledoc_root: Path,
    top_k: int,
    chunk_max_chars: int,
    chunk_overlap_chars: int,
    embedding_model: str,
) -> list[Response]:
    source_document = rows[0]["source_document"]
    source_path = resolve_source_path(source_document, paddledoc_root)

    ingestion = MarkdownIngestion(
        chunk_documents=True,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    chunks = ingestion.load(source_path)
    if not chunks:
        raise ValueError(f"No chunks produced from {source_path}")

    collection_name = f"eval_{source_path.stem}_{uuid.uuid4().hex[:8]}"
    rag = build_rag(chunks, collection_name, top_k, embedding_model)

    responses: list[Response] = []
    try:
        for row in rows:
            query = str(row["question"])
            evidence_quote = str(row.get("evidence_quote", ""))
            gold_answer = str(row.get("gold_answer", ""))

            retrieved_docs = rag.retrieve_contexts([query])[0]
            reference_docs = pick_reference_documents(chunks, evidence_quote)

            responses.append(
                Response(
                    request_id=str(row["id"]),
                    prompt_id=str(row["id"]),
                    sys_prompt="",
                    user_prompt=query,
                    response="",
                    context=Context.from_documents(retrieved_docs),
                    meta_data=MetaData(
                        tags={
                            "reference_answer": gold_answer,
                            "reference_document": reference_docs,
                            "question": query,
                            "source_document": source_document,
                        }
                    ),
                )
            )
    finally:
        try:
            rag.client.delete_collection(collection_name)
        except Exception:
            pass

    return responses


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for a JSONL dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(
            Path(__file__).resolve().parents[2]
            / "PaddleDoc"
            / "docs"
            / "evaluation"
            / "retrieval_dataset_template.jsonl"
        ),
        help="Path to the JSONL dataset.",
    )
    parser.add_argument(
        "--paddledoc-root",
        default=str(Path(__file__).resolve().parents[2] / "PaddleDoc"),
        help="Root directory of the PaddleDoc repository.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to score.")
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk during markdown ingestion.",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=150,
        help="Overlap between non-table chunks during markdown ingestion.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model used for retrieval.",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=3,
        help="K for Recall@K and HitRate@K.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    paddledoc_root = Path(args.paddledoc_root).expanduser().resolve()

    rows = load_dataset(dataset_path)
    if not rows:
        raise ValueError(f"No dataset rows found in {dataset_path}")

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["source_document"])] .append(row)

    all_responses: list[Response] = []
    for source_document, source_rows in grouped_rows.items():
        print(f"Evaluating {len(source_rows)} questions from {source_document}")
        all_responses.extend(
            evaluate_group(
                source_rows,
                paddledoc_root=paddledoc_root,
                top_k=args.top_k,
                chunk_max_chars=args.chunk_max_chars,
                chunk_overlap_chars=args.chunk_overlap_chars,
                embedding_model=args.embedding_model,
            )
        )

    responses = ResponseWrapper(all_responses)
    mrr = MeanReciprocalRank()(responses)
    recall = RecallAtK(args.recall_k)(responses)
    hit_rate = HitRateAtK(args.recall_k)(responses)

    summary = {
        "dataset": str(dataset_path),
        "questions": len(all_responses),
        "top_k": args.top_k,
        "recall_k": args.recall_k,
        "metrics": {
            "mrr": mrr.to_dict(),
            f"recall@{args.recall_k}": recall.to_dict(),
            f"hit_rate@{args.recall_k}": hit_rate.to_dict(),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())