import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.ingestion.loaders.pdf import load_pdf  # noqa: E402
from app.ingestion.pipeline import ingest_document  # noqa: E402

if __name__ == "__main__":
    init_db()
    session = SessionLocal()

    doc = load_pdf(Path("data/raw/gated_recurrent_neural_networks.pdf"))

    print("--- first ingest ---")
    result1 = ingest_document(doc, session)
    print(f"was_duplicate: {result1.was_duplicate}, chunk_count: {result1.chunk_count}, doc.id: {result1.document.id}")

    print("\n--- second ingest (same file) ---")
    result2 = ingest_document(doc, session)
    print(f"was_duplicate: {result2.was_duplicate}, chunk_count: {result2.chunk_count}, doc.id: {result2.document.id}")

    session.close()