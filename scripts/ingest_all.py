import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.ingestion.loaders.pdf import load_pdf  # noqa: E402
from app.ingestion.loaders.web import load_web  # noqa: E402
from app.ingestion.pipeline import ingest_document  # noqa: E402

WEB_URL = "https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/"

if __name__ == "__main__":
    init_db()
    session = SessionLocal()

    raw_dir = Path("data/raw")
    pdf_paths = sorted(raw_dir.glob("*.pdf"))

    for path in pdf_paths:
        doc = load_pdf(path)
        result = ingest_document(doc, session)
        status = "duplicate (skipped)" if result.was_duplicate else "ingested"
        print(f"{doc.source_name}: {status}, chunks={result.chunk_count}, doc.id={result.document.id}")

    web_doc = load_web(WEB_URL)
    result = ingest_document(web_doc, session)
    status = "duplicate (skipped)" if result.was_duplicate else "ingested"
    print(f"{web_doc.source_name}: {status}, chunks={result.chunk_count}, doc.id={result.document.id}")

    session.close()