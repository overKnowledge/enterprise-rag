from pathlib import Path

import pymupdf  # PyMuPDF

from app.ingestion.hashing import hash_bytes
from app.ingestion.loaders.base import PageContent, RawDocument


def load_pdf(path: Path) -> RawDocument:
    raw_bytes = path.read_bytes()
    content_hash = hash_bytes(raw_bytes)

    pages: list[PageContent] = []
    with pymupdf.open(stream=raw_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:  # skip genuinely blank pages (common on paper PDFs)
                pages.append(PageContent(page_number=page_number, text=text))

    return RawDocument(
        source_type="pdf",
        source_name=path.name,
        content_hash=content_hash,
        pages=pages,
    )