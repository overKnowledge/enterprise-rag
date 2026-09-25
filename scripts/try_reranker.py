import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    reranker = CrossEncoderReranker()

    query = "Unlike LSTMs, what does the GRU NOT have?"
    hits = store.search(query, top_k=15)  # retrieve broadly first

    print("--- BEFORE reranking (vector search order) ---")
    for h in hits[:5]:
        print(f"  distance={h['distance']:.4f}  {h['metadata']['source_name']} p{h['metadata']['page_start']}  {h['text'][:80]}")

    reranked = reranker.rerank(query, hits, top_n=5)

    print("\n--- AFTER reranking (cross-encoder order) ---")
    for h in reranked:
        print(f"  score={h['rerank_score']:.4f}  {h['metadata']['source_name']} p{h['metadata']['page_start']}  {h['text'][:80]}")