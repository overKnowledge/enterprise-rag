import chromadb
from chromadb.config import Settings as ChromaSettings

from app.embeddings.base import EmbeddingProvider

COLLECTION_NAME = "rag_chunks"


class ChromaVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider, persist_dir: str = "data/chroma"):
        self._embedder = embedding_provider
        self._client = chromadb.PersistentClient(
            path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self, chunk_ids: list[str], texts: list[str], metadatas: list[dict]
    ) -> None:
        vectors = self._embedder.embed_texts(texts)
        self._collection.upsert(
            ids=chunk_ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self._embedder.embed_query(query)
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()