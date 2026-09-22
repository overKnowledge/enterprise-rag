from dataclasses import dataclass

import tiktoken

from app.ingestion.loaders.base import RawDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")  # used by OpenAI embedding models

CHUNK_SIZE_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 65  # ~15% of 450


@dataclass
class Chunk:
    chunk_id: str  # f"{content_hash}_{chunk_index}"
    document_hash: str
    chunk_index: int
    text: str
    token_count: int
    page_start: int
    page_end: int


def _build_page_offset_index(doc: RawDocument) -> list[tuple[int, int, int]]:
    """Return (char_start, char_end, page_number) for each page within full_text."""
    index = []
    cursor = 0
    for page in doc.pages:
        start = cursor
        end = start + len(page.text)
        index.append((start, end, page.page_number))
        cursor = end + 2  # account for the "\n\n" join separator in full_text
    return index


def _page_for_offset(offset: int, page_index: list[tuple[int, int, int]]) -> int:
    for start, end, page_number in page_index:
        if start <= offset < end:
            return page_number
    return page_index[-1][2]  # fall back to the last page if offset lands past the end


def chunk_document(doc: RawDocument) -> list[Chunk]:
    full_text = doc.full_text
    tokens = _ENCODING.encode(full_text)
    page_index = _build_page_offset_index(doc)

    chunks: list[Chunk] = []
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    chunk_index = 0

    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_SIZE_TOKENS]
        if not window:
            break

        chunk_text = _ENCODING.decode(window)

        # Map token window back to a character offset to find the page range.
        char_start = len(_ENCODING.decode(tokens[:start]))
        char_end = char_start + len(chunk_text)

        page_start = _page_for_offset(char_start, page_index)
        page_end = _page_for_offset(max(char_end - 1, char_start), page_index)

        chunks.append(
            Chunk(
                chunk_id=f"{doc.content_hash}_{chunk_index}",
                document_hash=doc.content_hash,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(window),
                page_start=page_start,
                page_end=page_end,
            )
        )
        chunk_index += 1

        if start + CHUNK_SIZE_TOKENS >= len(tokens):
            break

    return chunks