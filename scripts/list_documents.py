import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.db.models import Document  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402

if __name__ == "__main__":
    session = SessionLocal()
    documents = session.query(Document).all()

    total_chunks = 0
    for d in documents:
        print(f"[{d.id}] {d.source_name} ({d.source_type}) — {d.page_count} pages, {d.chunk_count} chunks")
        total_chunks += d.chunk_count

    print(f"\n{len(documents)} documents, {total_chunks} total chunks")
    session.close()
    