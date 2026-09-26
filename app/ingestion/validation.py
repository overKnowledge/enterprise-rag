# Minimal magic-byte signatures for the formats we accept.
_SIGNATURES = {
    ".pdf": [b"%PDF-"],
    ".docx": [b"PK\x03\x04"],  # DOCX is a zip archive
}


def content_matches_extension(content: bytes, suffix: str) -> bool:
    signatures = _SIGNATURES.get(suffix)
    if signatures is None:
        return False
    return any(content.startswith(sig) for sig in signatures)