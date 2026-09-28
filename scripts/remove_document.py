import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.db.models import Document  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.vectorstore.chroma import COLLECTION_NAME  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/remove_document.py <source_name>")
    source_name = sys.argv[1]

    session = SessionLocal()
    document = session.query(Document).filter_by(source_name=source_name).one_or_none()
    if document is None:
        sys.exit(f"No document named {source_name!r}")

    chunk_ids = [c.chunk_id for c in document.chunks]

    client = chromadb.PersistentClient(
        path="data/chroma", settings=ChromaSettings(anonymized_telemetry=False)
    )
    client.get_collection(COLLECTION_NAME).delete(ids=chunk_ids)

    for chunk in document.chunks:
        session.delete(chunk)
    session.delete(document)
    session.commit()
    print(f"Removed {source_name}: {len(chunk_ids)} chunks")