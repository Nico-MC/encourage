from encourage.vector_store.chroma import ChromaClient
from encourage.vector_store.vector_store import VectorStore

__all__ = [
    "ChromaClient",
    "QdrantCustomClient",
    "VectorStore",
]


def __getattr__(name: str) -> object:
    if name == "QdrantCustomClient":
        from encourage.vector_store.qdrant import QdrantCustomClient

        return QdrantCustomClient
    raise AttributeError(f"module 'encourage.vector_store' has no attribute {name!r}")
