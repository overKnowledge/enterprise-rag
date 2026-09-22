import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.loaders.pdf import load_pdf  # noqa: E402
from app.ingestion.loaders.web import load_web  # noqa: E402

WEB_URL = "https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/"

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    pdf_paths = sorted(raw_dir.glob("*.pdf"))

    all_chunks = []

    for path in pdf_paths:
        doc = load_pdf(path)
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"{doc.source_name}: {len(doc.pages)} pages -> {len(chunks)} chunks")

    web_doc = load_web(WEB_URL)
    web_chunks = chunk_document(web_doc)
    all_chunks.extend(web_chunks)
    print(f"{web_doc.source_name}: {len(web_doc.pages)} page -> {len(web_chunks)} chunks")

    print(f"\nTotal chunks across all sources: {len(all_chunks)}")