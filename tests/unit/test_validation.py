from app.ingestion.validation import content_matches_extension


def test_real_pdf_signature_matches_pdf_extension():
    assert content_matches_extension(b"%PDF-1.4 rest of file...", ".pdf") is True


def test_real_docx_signature_matches_docx_extension():
    # DOCX files are ZIP archives; PK\x03\x04 is ZIP's magic bytes.
    assert content_matches_extension(b"PK\x03\x04 rest of file...", ".docx") is True


def test_plain_text_disguised_as_pdf_is_rejected():
    assert content_matches_extension(b"this is not a real pdf", ".pdf") is False


def test_unsupported_extension_is_rejected():
    assert content_matches_extension(b"%PDF-1.4", ".exe") is False