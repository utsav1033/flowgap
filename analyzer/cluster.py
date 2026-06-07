"""
Cluster embedded caller turns using HDBSCAN.
Returns cluster_id per turn (-1 = noise).
"""

import numpy as np


def cluster_embeddings(
    vectors: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int = 3,
) -> np.ndarray:
    """
    Run HDBSCAN. Returns int array of cluster labels (-1 = noise).
    Falls back to KMeans if HDBSCAN yields <3 meaningful clusters.
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vectors)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).sum() / len(labels)
    print(f"HDBSCAN: {n_clusters} clusters, {noise_frac*100:.1f}% noise")

    if n_clusters < 3:
        print("  Too few clusters — falling back to KMeans(k=8)")
        labels = _kmeans_fallback(vectors, k=8)

    return labels


def _kmeans_fallback(vectors: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(vectors)


def cluster_summary(labels: np.ndarray, texts: list[str], n_samples: int = 3) -> dict:
    """Return {cluster_id: [sample_texts]} for inspection."""
    from collections import defaultdict
    buckets: dict[int, list[str]] = defaultdict(list)
    for label, text in zip(labels, texts):
        buckets[label].append(text)
    summary = {}
    for cid in sorted(buckets):
        samples = buckets[cid]
        summary[cid] = {
            "size": len(samples),
            "samples": samples[:n_samples],
        }
    return summary


if __name__ == "__main__":
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
