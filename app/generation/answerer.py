from app.generation.base import LLMProvider
from app.generation.prompts import NO_ANSWER_PHRASE, SYSTEM_PROMPT, build_user_prompt
from app.vectorstore.chroma import ChromaVectorStore


class Answerer:
    def __init__(self, vector_store: ChromaVectorStore, llm: LLMProvider, top_k: int = 5):
        self._store = vector_store
        self._llm = llm
        self._top_k = top_k

    def answer(self, question: str) -> dict:
        hits = self._store.search(question, top_k=self._top_k)
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
                    "distance": h["distance"],
                }
                for h in hits
            ]
        )

        return {"answer": answer_text, "sources": sources}