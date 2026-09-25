import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

QUESTIONS = [
    "What is the vanishing gradient problem?",
    "What happens after training completes, in terms of how a model's weights get updated?",
    "Why might a recurrent model struggle with long sequences, and how does attention avoid that problem?",
]

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    reranker = CrossEncoderReranker()

    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        hits = store.search(q, top_k=15)
        pairs = [(q, h["text"]) for h in hits]
        scores = reranker._model.predict(pairs)
        for h, s in sorted(zip(hits, scores), key=lambda x: -x[1])[:5]:
            print(f"  score={s:.4f}  {h['metadata']['source_name']} p{h['metadata']['page_start']}: {h['text'][:100]}")