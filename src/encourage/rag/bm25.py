"""BM25-only RAG implementation using sparse lexical retrieval."""

import logging
import re
import uuid
from dataclasses import replace
from typing import Any, Tuple, override

import numpy as np
from rank_bm25 import BM25Okapi

from encourage.prompts.context import Document
from encourage.rag.base.config import BM25RAGConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().translate(str.maketrans({
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
    }))
    return re.findall(r'[a-z0-9]+', normalized)


@RAGFactory.register(RAGMethod.BM25, BM25RAGConfig)
class BM25RAG(BaseRAG):
    """RAG method that only uses BM25 sparse retrieval.

    This class subclasses BaseRAG but bypasses any dense retrieval. It
    builds a BM25 index on initialization and ranks documents purely by
    BM25 score (optionally using positional fallbacks similar to
    HybridBM25RAG implementation).
    """

    def __init__(self, config: BM25RAGConfig, **kwargs: Any) -> None:
        super().__init__(config)
        # Create BM25 index using provided context collection
        self._create_bm25_index(self.context_collection)

    def _create_bm25_index(self, context_collection: list[Document]) -> None:
        self.documents = list(context_collection)
        self.document_texts = [_tokenize(doc.content) for doc in self.documents]
        self.bm25_index = BM25Okapi(self.document_texts)
        logger.info(f"BM25 index created with {len(self.document_texts)} documents.")

    def _retrieve_sparse_results(self, query: str) -> Tuple[list[Document], dict[uuid.UUID, float]]:
        tokenized_query = _tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Order documents by score descending
        top_indices = np.argsort(bm25_scores)[::-1]
        sparse_docs = [self.documents[i] for i in top_indices]

        # Normalize scores to [0,1]
        max_score = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        if max_score > 0:
            normalized = {
                self.documents[i].id: float(score) / float(max_score)
                for i, score in enumerate(bm25_scores)
                if score > 0
            }
        else:
            normalized = {}

        return sparse_docs, normalized

    def _rank_documents(self, query: str) -> list[Document]:
        """Rank documents purely by BM25 score and return top_k."""
        sparse_docs, normalized_scores = self._retrieve_sparse_results(query)
        total_documents = max(len(sparse_docs), 1)
        return [
            replace(
                document,
                score=normalized_scores.get(document.id, 0.0)
                + (total_documents - rank) * 1e-12,
                distance=None,
            )
            for rank, document in enumerate(sparse_docs[: self.top_k])
        ]

    # Override retrieve_contexts to skip dense retrieval entirely
    @override
    def retrieve_contexts(self, query_list: list[str], **kwargs: Any) -> list[list[Document]]:
        results = []
        for q in query_list:
            ranked = self._rank_documents(q)
            results.append(ranked)
        return results
