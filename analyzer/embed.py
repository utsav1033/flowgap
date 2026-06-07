"""
Embed caller turns into vectors.
Default: local sentence-transformers (all-MiniLM-L6-v2).
Set EMBEDDING_PROVIDER=openai in .env to use OpenAI instead.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()


def _load_local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


_LOCAL_MODEL = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 array of embeddings."""
    if PROVIDER == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def _embed_local(texts: list[str]) -> np.ndarray:
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        print("Loading sentence-transformers model (first run may download ~80MB)...")
        _LOCAL_MODEL = _load_local_model()
    vecs = _LOCAL_MODEL.encode(texts, batch_size=64, show_progress_bar=len(texts) > 100, convert_to_numpy=True)
    return vecs.astype(np.float32)


def _embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    batch = 100
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts[i:i+batch])
        results.extend([d.embedding for d in resp.data])
    return np.array(results, dtype=np.float32)


if __name__ == "__main__":
    sample = ["I want to book an appointment", "What are your timings?", "I need my lab report"]
    vecs = embed_texts(sample)
    print(f"Embedded {len(sample)} texts -> shape {vecs.shape}")
