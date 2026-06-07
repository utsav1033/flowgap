"""
Cluster embedded calls: UMAP → HDBSCAN.

UMAP collapses the 384-dim MiniLM space to ~8 dimensions before HDBSCAN runs.
This is the standard recipe: high-dim cosine embeddings have poor density structure
for HDBSCAN, but after UMAP the clusters become compact blobs in euclidean space.
"""

import numpy as np


def cluster_embeddings(
    vectors: np.ndarray,
    n_components: int = 8,
    n_neighbors: int = 15,
    min_cluster_size: int = 5,
    min_samples: int = 1,
) -> np.ndarray:
    """
    UMAP → HDBSCAN. Returns int array of cluster labels (-1 = noise).
    Falls back to KMeans(k=9) if HDBSCAN yields <3 meaningful clusters.
    """
    # Step 1: UMAP reduction (cosine in high-dim → euclidean in low-dim)
    import umap as umap_lib
    print(f"      UMAP: {vectors.shape[1]}-dim -> {n_components}-dim "
          f"(n_neighbors={n_neighbors}, metric=cosine)...")
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(vectors)

    # Step 2: HDBSCAN on the compact low-dim space
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).sum() / len(labels)
    print(f"      HDBSCAN: {n_clusters} clusters, {noise_frac*100:.1f}% noise")

    if n_clusters < 3:
        print(f"      Too few clusters ({n_clusters}) — falling back to KMeans(k=9)")
        labels = _kmeans_fallback(vectors, k=9)

    return labels


def _kmeans_fallback(vectors: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(vectors)


def cluster_summary(labels: np.ndarray, texts: list[str], n_samples: int = 3) -> dict:
    """Return {cluster_id: {size, samples}} for inspection."""
    from collections import defaultdict
    buckets: dict[int, list[str]] = defaultdict(list)
    for label, text in zip(labels, texts):
        buckets[label].append(text)
    return {
        cid: {"size": len(samples), "samples": samples[:n_samples]}
        for cid, samples in sorted(buckets.items())
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analyzer.embed import embed_texts
    from analyzer.parse import load_calls
    calls = load_calls()
    all_turns = [t for c in calls for t in c["caller_turns"]]
    print(f"Embedding {len(all_turns)} caller turns...")
    vecs = embed_texts(all_turns)
    labels = cluster_embeddings(vecs)
    summary = cluster_summary(labels, all_turns)
    for cid, info in list(summary.items())[:5]:
        tag = "NOISE" if cid == -1 else f"cluster_{cid}"
        print(f"  {tag} ({info['size']} turns): {info['samples'][0][:80]}")
