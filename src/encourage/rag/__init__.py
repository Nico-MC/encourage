from encourage.rag.base.config import (
    BaseRAGConfig,
    HydeRAGConfig,
    KnownContextConfig,
    NoContextConfig,
    RerankerRAGConfig,
    SelfRAGConfig,
    SummarizationContextRAGConfig,
    SummarizationRAGConfig,
)
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base.interface import RAGMethodInterface
from encourage.rag.base_impl import BaseRAG

__all__ = [
    "HydeRAG",
    "KnownContext",
    "BaseRAG",
    "RAGMethod",
    "SummarizationRAG",
    "NoContext",
    "RAGMethodInterface",
    "SummarizationContextRAG",
    "RerankerRAG",
    "HybridBM25RAG",
    "BM25RAG",
    "SelfRAG",
    "BaseRAGConfig",
    "HydeRAGConfig",
    "KnownContextConfig",
    "NoContextConfig",
    "RerankerRAGConfig",
    "SelfRAGConfig",
    "SummarizationContextRAGConfig",
    "SummarizationRAGConfig",
    "RAGFactory",
    "HydeRerankerRAG",
]


def __getattr__(name: str) -> object:
    if name == "BM25RAG":
        from encourage.rag.bm25 import BM25RAG

        return BM25RAG
    if name == "HybridBM25RAG":
        from encourage.rag.hybrid_bm25 import HybridBM25RAG

        return HybridBM25RAG
    if name == "HydeRAG":
        from encourage.rag.hyde import HydeRAG

        return HydeRAG
    if name == "HydeRerankerRAG":
        from encourage.rag.hyde_reranker import HydeRerankerRAG

        return HydeRerankerRAG
    if name == "KnownContext":
        from encourage.rag.known_context import KnownContext

        return KnownContext
    if name == "NoContext":
        from encourage.rag.no_context import NoContext

        return NoContext
    if name == "RerankerRAG":
        from encourage.rag.reranker import RerankerRAG

        return RerankerRAG
    if name == "SelfRAG":
        from encourage.rag.self_rag import SelfRAG

        return SelfRAG
    if name == "SummarizationContextRAG":
        from encourage.rag.summarize import SummarizationContextRAG

        return SummarizationContextRAG
    if name == "SummarizationRAG":
        from encourage.rag.summarize import SummarizationRAG

        return SummarizationRAG
    raise AttributeError(f"module 'encourage.rag' has no attribute {name!r}")
