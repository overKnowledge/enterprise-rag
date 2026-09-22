import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.web import load_web  # noqa: E402

if __name__ == "__main__":
    url = "https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/"
    doc = load_web(url)

    print(f"source: {doc.source_name}")
    print(f"hash:   {doc.content_hash[:16]}...")
    print(f"pages:  {len(doc.pages)}")
    print("--- first 300 chars ---")
    print(doc.pages[0].text[:300])