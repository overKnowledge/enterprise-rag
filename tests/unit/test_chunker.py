from app.ingestion.chunker import CHUNK_SIZE_TOKENS, chunk_document
from app.ingestion.loaders.base import PageContent, RawDocument


def _make_doc(word_count: int) -> RawDocument:
    """A document with enough distinct words to force multiple chunks."""
    text = " ".join(f"word{i}" for i in range(word_count))
    return RawDocument(
        source_type="pdf",
        source_name="test.pdf",
        content_hash="deadbeef",
        pages=[PageContent(page_number=1, text=text)],
    )


def test_single_short_document_produces_one_chunk():
    doc = _make_doc(word_count=20)
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_no_chunk_exceeds_the_token_limit():
    doc = _make_doc(word_count=2000)
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(c.token_count <= CHUNK_SIZE_TOKENS for c in chunks)


def test_consecutive_chunks_overlap():
    import tiktoken

    from app.ingestion.chunker import CHUNK_OVERLAP_TOKENS

    doc = _make_doc(word_count=2000)
    chunks = chunk_document(doc)

    # The chunker guarantees the last CHUNK_OVERLAP_TOKENS tokens of one chunk
    # reappear at the start of the next. Boundaries are token-based, not
    # word-based, so decoding with the same tokenizer is the correct check.
    encoding = tiktoken.get_encoding("cl100k_base")
    chunk0_tokens = encoding.encode(chunks[0].text)
    overlap_text = encoding.decode(chunk0_tokens[-CHUNK_OVERLAP_TOKENS:])

    assert overlap_text in chunks[1].text


def test_chunk_ids_are_stable_and_unique():
    doc = _make_doc(word_count=1500)
    chunks = chunk_document(doc)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # no duplicates
    assert all(c.chunk_id == f"{doc.content_hash}_{c.chunk_index}" for c in chunks)