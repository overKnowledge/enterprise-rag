import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.generation.answerer import Answerer  # noqa: E402
from app.generation.groq_provider import GroqLLMProvider  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    llm = GroqLLMProvider()
    answerer = Answerer(store, llm)

    result = answerer.answer("What model won the WMT 2014 competition outright, according to this paper?")
    print(result["answer"])
    print("\nSources:", result["sources"])