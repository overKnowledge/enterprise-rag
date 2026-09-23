from typing import Protocol


class EmbeddingProvider(Protocol):
    """Any embedding backend must implement this. Lets us swap HuggingFace for
    OpenAI/Mistral later without touching retrieval or ingestion code."""

    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...