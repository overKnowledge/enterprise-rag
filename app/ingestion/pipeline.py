import logging

from sqlalchemy.orm import Session

from app.db.models import ChunkRecord, Document
from app.ingestion.chunker import chunk_document
from app.ingestion.loaders.base import RawDocument

logger = logging.getLogger(__name__)


class IngestResult:
    def __init__(self, document: Document, was_duplicate: bool, chunk_count: int):
        self.document = document
        self.was_duplicate = was_duplicate
        self.chunk_count = chunk_count


def ingest_document(doc: RawDocument, session: Session) -> IngestResult:
    """Chunk a loaded document and persist its metadata. Idempotent: re-ingesting
    a document with the same content_hash is a no-op that returns the existing record."""

    existing = (
        session.query(Document).filter_by(content_hash=doc.content_hash).one_or_none()
    )
    if existing is not None:
        logger.info(
            "Skipping duplicate document: %s (hash=%s)",
            doc.source_name,
            doc.content_hash[:12],
        )
        return IngestResult(document=existing, was_duplicate=True, chunk_count=existing.chunk_count)

    chunks = chunk_document(doc)

    document = Document(
        content_hash=doc.content_hash,
        source_type=doc.source_type,
        source_name=doc.source_name,
        page_count=len(doc.pages),
        chunk_count=len(chunks),
    )
    session.add(document)
    session.flush()  # assigns document.id before we reference it below

    for chunk in chunks:
        session.add(
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                token_count=chunk.token_count,
            )
        )

    session.commit()
    logger.info("Ingested %s: %d pages -> %d chunks", doc.source_name, len(doc.pages), len(chunks))

    return IngestResult(document=document, was_duplicate=False, chunk_count=len(chunks))