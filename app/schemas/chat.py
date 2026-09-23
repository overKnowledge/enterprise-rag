from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    source_name: str
    page_start: int
    page_end: int
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]