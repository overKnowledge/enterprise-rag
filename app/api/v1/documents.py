import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.models import Document
from app.db.session import get_session
from app.ingestion.loaders.docx import load_docx
from app.ingestion.loaders.pdf import load_pdf
from app.ingestion.loaders.web import load_web
from app.ingestion.pipeline import ingest_document
from app.schemas.documents import DocumentResponse, WebIngestRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf": load_pdf, ".docx": load_docx}


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

    # Loaders expect a filesystem path, so we stream the upload to a temp file first.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        raw_doc = loader(tmp_path)
        raw_doc.source_name = file.filename or raw_doc.source_name  # keep the original name
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