import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

QUERIES = [
    "What is multi-head attention?",
    "How does CRAG correct retrieval errors?",
    "What is the difference between GRU and LSTM gating?",
    "What is regression used for in machine learning?",
]

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)

    for query in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print("=" * 70)
        hits = store.search(query, top_k=3)
        for rank, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            print(f"\n[{rank}] {meta['source_name']} (page {meta['page_start']}-{meta['page_end']}) — distance={hit['distance']:.4f}")
            print(hit["text"][:200].replace("\n", " "))