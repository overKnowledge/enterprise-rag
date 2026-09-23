import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.huggingface import HuggingFaceEmbeddingProvider  # noqa: E402

if __name__ == "__main__":
    print("Loading model (first run downloads it, ~130MB)...")
    start = time.time()
    provider = HuggingFaceEmbeddingProvider()
    print(f"Loaded in {time.time() - start:.1f}s. Dimension: {provider.dimension}")

    texts = [
        "The Transformer architecture relies entirely on attention mechanisms.",
        "Gated Recurrent Units simplify LSTM's gating structure.",
        "Regression predicts a continuous numerical output.",
    ]
    vectors = provider.embed_texts(texts)

    print(f"\nEmbedded {len(vectors)} texts, each with {len(vectors[0])} dimensions")
    print(f"First 5 values of vector 0: {vectors[0][:5]}")

    # Sanity check: similar sentences should have higher cosine similarity
    import numpy as np

    v0, v1, v2 = (np.array(v) for v in vectors)
    print(f"\nsim(transformer, gru)       = {np.dot(v0, v1):.4f}")
    print(f"sim(transformer, regression) = {np.dot(v0, v2):.4f}")
    print(f"sim(gru, regression)         = {np.dot(v1, v2):.4f}")