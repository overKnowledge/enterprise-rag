from fastapi import APIRouter

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider
from app.generation.answerer import Answerer
from app.generation.groq_provider import GroqLLMProvider
from app.schemas.chat import ChatRequest, ChatResponse
from app.vectorstore.chroma import ChromaVectorStore
from fastapi import Depends
from app.core.security import verify_api_key

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(verify_api_key)])

# Loaded once at import time — the embedding model and Chroma connection are
# expensive to initialize and safe to share across requests.
_provider = HuggingFaceEmbeddingProvider()
_store = ChromaVectorStore(_provider)
_llm = GroqLLMProvider()
_answerer = Answerer(_store, _llm)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = _answerer.answer(request.question)
    return ChatResponse(**result)