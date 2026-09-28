from functools import lru_cache

from fastapi import APIRouter, Depends

from app.api.deps import get_vector_store
from app.core.security import verify_api_key
from app.generation.answerer import Answerer
from app.generation.groq_provider import GroqLLMProvider
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(verify_api_key)])


@lru_cache
def get_answerer() -> Answerer:
    return Answerer(get_vector_store(), GroqLLMProvider())


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, answerer: Answerer = Depends(get_answerer)) -> ChatResponse:
    result = answerer.answer(request.question)
    return ChatResponse(**result)