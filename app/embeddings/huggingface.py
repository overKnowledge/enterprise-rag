from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddingProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity works cleanly on unit vectors
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]