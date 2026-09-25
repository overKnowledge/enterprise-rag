import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from app.retrieval.retriever import RerankingRetriever  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

QUERY = "What is the exact GitHub repository URL for the CRAG implementation mentioned in the paper?"

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    reranker = CrossEncoderReranker()
    retriever = RerankingRetriever(store, reranker, fetch_k=15, top_n=5)

    print("--- Plain vector search (baseline, top 5) ---")
    plain_hits = store.search(QUERY, top_k=5)
    for h in plain_hits:
        print(f"  {h['metadata']['source_name']} p{h['metadata']['page_start']}: {h['text'][:150]}")

    print("\n--- Reranked (top 5 of 15) ---")
    reranked_hits = retriever.retrieve(QUERY)
    for h in reranked_hits:
        print(f"  score={h['rerank_score']:.4f}  {h['metadata']['source_name']} p{h['metadata']['page_start']}: {h['text'][:150]}")