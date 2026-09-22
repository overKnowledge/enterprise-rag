import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.loaders.pdf import load_pdf  # noqa: E402

if __name__ == "__main__":
    doc = load_pdf(Path("data/raw/attention_is_all_you_need.pdf"))
    chunks = chunk_document(doc)

    print(f"document: {doc.source_name}")
    print(f"total chunks: {len(chunks)}\n")

    for c in chunks[:3]:
        print(f"--- chunk {c.chunk_index} (tokens={c.token_count}, pages={c.page_start}-{c.page_end}) ---")
        print(c.text[:200])
        print("...\n")

    # Sanity check: does the overlap actually overlap?
    if len(chunks) > 1:
        end_of_first = chunks[0].text[-100:]
        start_of_second = chunks[1].text[:200]
        print("--- checking overlap between chunk 0 and chunk 1 ---")
        print("end of chunk 0:", end_of_first)
        print("start of chunk 1:", start_of_second)