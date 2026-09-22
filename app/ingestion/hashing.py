import hashlib


def hash_bytes(data: bytes) -> str:
    """Return a stable SHA-256 hex digest, used as a document's content ID."""
    return hashlib.sha256(data).hexdigest()