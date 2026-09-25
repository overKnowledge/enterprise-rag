from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, hits: list[dict], top_n: int = 5, min_top_score: float | None = None
    ) -> list[dict]:
        pairs = [(query, hit["text"]) for hit in hits]
        scores = self._model.predict(pairs)

        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

        # Only reject the WHOLE result set if even the best candidate is weak —
        # don't filter individual mid-ranked chunks, since correct chunks can
        # legitimately score negative when the query doesn't share the source's vocabulary.
        if min_top_score is not None and ranked and ranked[0]["rerank_score"] < min_top_score:
            return []

        return ranked[:top_n]