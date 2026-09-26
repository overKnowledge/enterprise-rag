import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.security import verify_api_key
from app.db.models import Document
from app.db.session import get_session
from app.ingestion.loaders.docx import load_docx
from app.ingestion.loaders.pdf import load_pdf
from app.ingestion.loaders.web import load_web
from app.ingestion.pipeline import ingest_document
from app.ingestion.validation import content_matches_extension
from app.schemas.documents import DocumentResponse, WebIngestRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"], dependencies=[Depends(verify_api_key)])

SUPPORTED_EXTENSIONS = {".pdf": load_pdf, ".docx": load_docx}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile, session: Session = Depends(get_session)
) -> DocumentResponse:
    suffix = Path(file.filename or "").suffix.lower()
    loader = SUPPORTED_EXTENSIONS.get(suffix)
    if loader is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {list(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB",
        )

    if not content_matches_extension(content, suffix):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match its extension '{suffix}'",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        raw_doc = loader(tmp_path)
        raw_doc.source_name = file.filename or raw_doc.source_name
        result = ingest_document(raw_doc, session)
    finally:
        tmp_path.unlink(missing_ok=True)

    return DocumentResponse(
        **{c.name: getattr(result.document, c.name) for c in Document.__table__.columns},
        was_duplicate=result.was_duplicate,
    )


@router.post("/url", response_model=DocumentResponse, status_code=201)
async def ingest_from_url(
    body: WebIngestRequest, session: Session = Depends(get_session)
) -> DocumentResponse:
    raw_doc = load_web(body.url)
    result = ingest_document(raw_doc, session)

    return DocumentResponse(
        **{c.name: getattr(result.document, c.name) for c in Document.__table__.columns},
        was_duplicate=result.was_duplicate,
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentResponse]:
    documents = session.query(Document).all()
    return [
        DocumentResponse(
            **{c.name: getattr(d, c.name) for c in Document.__table__.columns},
            was_duplicate=False,
        )
        for d in documents
    ]