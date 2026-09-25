from app.generation.base import LLMProvider
from app.generation.prompts import NO_ANSWER_PHRASE, SYSTEM_PROMPT, build_user_prompt


class Answerer:
    def __init__(self, retriever, llm: LLMProvider, top_k: int = 5):
        """`retriever` must expose either .search(query, top_k=N) -> list[dict]
        (plain vector store) or .retrieve(query) -> list[dict] (reranking retriever)."""
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k

    def _get_hits(self, question: str) -> list[dict]:
        if hasattr(self._retriever, "retrieve"):
            return self._retriever.retrieve(question)
        return self._retriever.search(question, top_k=self._top_k)

    def answer(self, question: str) -> dict:
        hits = self._get_hits(question)
        user_prompt = build_user_prompt(question, hits)
        answer_text = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        is_unanswered = answer_text.strip().startswith(NO_ANSWER_PHRASE)
        sources = (
            []
            if is_unanswered
            else [
                {
                    "source_name": h["metadata"]["source_name"],
                    "page_start": h["metadata"]["page_start"],
                    "page_end": h["metadata"]["page_end"],
                    "distance": h.get("distance"),
                }
                for h in hits
            ]
        )

        return {"answer": answer_text, "sources": sources}