from functools import lru_cache

from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.embeddings.huggingface import HuggingFaceEmbeddingProvider
from app.generation.answerer import Answerer
from app.generation.groq_provider import GroqLLMProvider
from app.schemas.chat import ChatRequest, ChatResponse
from app.vectorstore.chroma import ChromaVectorStore

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(verify_api_key)])


@lru_cache
def get_answerer() -> Answerer:
    """Built on the first /chat request, then reused for every later one."""
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)
    llm = GroqLLMProvider()
    return Answerer(store, llm)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, answerer: Answerer = Depends(get_answerer)) -> ChatResponse:
    result = answerer.answer(request.question)
    return ChatResponse(**result)