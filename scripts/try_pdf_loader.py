import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.pdf import load_pdf  # noqa: E402

if __name__ == "__main__":
    pdf_path = Path("data/raw/attention_is_all_you_need.pdf")  # adjust to your filename
    doc = load_pdf(pdf_path)

    print(f"source: {doc.source_name}")
    print(f"hash:   {doc.content_hash[:16]}...")
    print(f"pages:  {len(doc.pages)}")
    print("--- first 300 chars of page 1 ---")
    print(doc.pages[0].text[:300])