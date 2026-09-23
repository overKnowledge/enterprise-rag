import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402
from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.loaders.pdf import load_pdf  # noqa: E402
from app.ingestion.loaders.web import load_web  # noqa: E402
from app.vectorstore.chroma import ChromaVectorStore  # noqa: E402

WEB_URL = "https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/"

if __name__ == "__main__":
    provider = HuggingFaceEmbeddingProvider()
    store = ChromaVectorStore(provider)

    raw_dir = Path("data/raw")
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    docs = [load_pdf(p) for p in pdf_paths] + [load_web(WEB_URL)]

    total = 0
    start = time.time()
    for doc in docs:
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "source_name": doc.source_name,
                "source_type": doc.source_type,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        store.add_chunks(ids, texts, metadatas)
        total += len(chunks)
        print(f"{doc.source_name}: embedded {len(chunks)} chunks")

    print(f"\nTotal indexed: {total} chunks in {time.time() - start:.1f}s")
    print(f"Collection count: {store.count()}")