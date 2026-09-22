import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.pdf import load_pdf  # noqa: E402

if __name__ == "__main__":
    doc = load_pdf(Path("data/raw/attention_is_all_you_need.pdf"))
    print(doc.pages[0].text)