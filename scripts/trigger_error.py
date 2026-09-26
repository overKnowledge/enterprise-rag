import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders.pdf import load_pdf  # noqa: E402

if __name__ == "__main__":
    # Deliberately load a file that doesn't exist, to see what an unhandled error looks like.
    load_pdf(Path("data/raw/this_file_does_not_exist.pdf"))