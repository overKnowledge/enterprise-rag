from app.retrieval.reranker import CrossEncoderReranker
from app.vectorstore.chroma import ChromaVectorStore


class RerankingRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        reranker: CrossEncoderReranker,
        fetch_k: int = 15,
        top_n: int = 5,
        min_top_score: float | None = -1.0,
    ):
        self._store = vector_store
        self._reranker = reranker
        self._fetch_k = fetch_k
        self._top_n = top_n
        self._min_top_score = min_top_score

    def retrieve(self, query: str) -> list[dict]:
        hits = self._store.search(query, top_k=self._fetch_k)
        return self._reranker.rerank(
            query, hits, top_n=self._top_n, min_top_score=self._min_top_score
        )