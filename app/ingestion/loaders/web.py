import httpx
import trafilatura

from app.ingestion.hashing import hash_bytes
from app.ingestion.loaders.base import PageContent, RawDocument


def load_web(url: str) -> RawDocument:
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    raw_bytes = response.content
    content_hash = hash_bytes(raw_bytes)

    text = trafilatura.extract(raw_bytes, include_tables=True, include_links=False)
    if not text:
        raise ValueError(f"Could not extract readable content from {url}")

    return RawDocument(
        source_type="web",
        source_name=url,
        content_hash=content_hash,
        pages=[PageContent(page_number=1, text=text)],
    )