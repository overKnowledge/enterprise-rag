from functools import lru_cache

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider
from app.vectorstore.chroma import ChromaVectorStore


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    """One shared embedding model + Chroma client, built on first use."""
    return ChromaVectorStore(HuggingFaceEmbeddingProvider())