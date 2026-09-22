from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    content_hash: str
    source_type: str
    source_name: str
    page_count: int
    chunk_count: int
    ingested_at: datetime
    was_duplicate: bool

    model_config = {"from_attributes": True}


class WebIngestRequest(BaseModel):
    url: str