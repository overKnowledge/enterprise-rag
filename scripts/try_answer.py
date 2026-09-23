import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.generation.answerer import Answerer  # noqa: E402
from app.generation.groq_provider import GroqLLMProvider  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

QUESTIONS = [
    "What is multi-head attention?",
    "How does CRAG decide when to correct a retrieval?",
    "What is the capital of France?",  # should trigger "I don't know" — not in corpus
]

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    llm = GroqLLMProvider()
    answerer = Answerer(store, llm)

    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        result = answerer.answer(q)
        print(result["answer"])
        print("\nSources:")
        for s in result["sources"][:3]:
            print(f"  - {s['source_name']} (page {s['page_start']})")