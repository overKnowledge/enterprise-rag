from app.ingestion.hashing import hash_bytes


def test_same_content_produces_the_same_hash():
    data = b"Attention Is All You Need"
    assert hash_bytes(data) == hash_bytes(data)


def test_different_content_produces_different_hashes():
    assert hash_bytes(b"content A") != hash_bytes(b"content B")


def test_hash_is_a_64_character_hex_string():
    digest = hash_bytes(b"anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)