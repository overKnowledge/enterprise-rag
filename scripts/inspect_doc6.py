import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.web import load_web  # noqa: E402

if __name__ == "__main__":
    doc = load_web("https://www.geeksforgeeks.org/deep-learning/deep-learning-tutorial/")
    print(f"chars extracted: {len(doc.full_text)}")
    print(doc.full_text[:500])