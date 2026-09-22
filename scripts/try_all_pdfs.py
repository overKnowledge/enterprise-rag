import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.pdf import load_pdf  # noqa: E402

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    pdf_paths = sorted(raw_dir.glob("*.pdf"))

    print(f"Found {len(pdf_paths)} PDF(s) in {raw_dir}\n")

    for path in pdf_paths:
        doc = load_pdf(path)
        total_chars = sum(len(p.text) for p in doc.pages)
        print(f"{doc.source_name}")
        print(f"  hash:  {doc.content_hash[:16]}...")
        print(f"  pages: {len(doc.pages)}")
        print(f"  chars: {total_chars}")
        print()