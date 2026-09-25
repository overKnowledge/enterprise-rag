import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.generation.answerer import Answerer  # noqa: E402
from app.generation.groq_provider import GroqLLMProvider  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from app.retrieval.retriever import RerankingRetriever  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    reranker = CrossEncoderReranker()
    retriever = RerankingRetriever(store, reranker, fetch_k=15, top_n=5, min_top_score=-1.0)
    llm = GroqLLMProvider()
    answerer = Answerer(retriever, llm)

    hits = retriever.retrieve("What is the exact GitHub repository URL for the CRAG implementation mentioned in the paper?")
    print(f"Chunks surviving threshold: {len(hits)}")

    result = answerer.answer("What is the exact GitHub repository URL for the CRAG implementation mentioned in the paper?")
    print("\nAnswer:", result["answer"])
    print("Sources:", result["sources"])