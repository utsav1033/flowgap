"""
Orchestrates the full analysis pipeline:
  parse -> embed -> cluster -> label -> graph -> gaps -> nodegen
Writes analysis.json to data/.

Embedding is per-call (concatenated caller turns), not per-turn, so that
each call gets one vector representing its primary intent rather than
fragmenting into per-turn junk clusters.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analyzer.parse import load_calls
from analyzer.embed import embed_texts
from analyzer.cluster import cluster_embeddings, cluster_summary
from analyzer.label import label_clusters
from analyzer.graph import build_graph
from analyzer.gaps import detect_gaps, compute_headline_metric
from analyzer.nodegen import generate_all_node_specs


def _call_text(call: dict) -> str:
    """One representative text per call: join caller turns, skip generation failures."""
    real = [t for t in call["caller_turns"] if t.strip() != "[generation failed]"]
    return " ".join(real) if real else "[generation failed]"


def run_pipeline() -> dict:
    print("=== FlowGap Analysis Pipeline ===\n")

    print("[1/6] Loading and parsing transcripts...")
    calls = load_calls()
    print(f"      Loaded {len(calls)} calls")

    # One text per call — captures full caller intent rather than individual turns
    call_texts = [_call_text(c) for c in calls]
    failed = sum(1 for t in call_texts if t == "[generation failed]")
    if failed:
        print(f"      ({failed} calls have no real content — [generation failed])")

    print(f"[2/6] Embedding {len(call_texts)} calls (one vector per call)...")
    vectors = embed_texts(call_texts)

    print("[3/6] Clustering...")
    # min_cluster_size=5 suits per-call scale (400 calls vs thousands of turns)
    labels = cluster_embeddings(vectors, min_cluster_size=5, min_samples=2)

    label_counts = Counter(labels.tolist())
    n_clusters = sum(1 for k in label_counts if k != -1)
    noise = label_counts.get(-1, 0)
    print(f"      {n_clusters} clusters, {noise} noise calls")
    for cid, count in sorted(label_counts.items(), key=lambda x: -x[1])[:12]:
        tag = "  noise" if cid == -1 else f"  cluster_{cid:2d}"
        print(f"        {tag}: {count} calls")

    # call_cluster_map: direct 1-to-1 since we embedded per call
    call_cluster_map = {calls[i]["call_id"]: int(labels[i]) for i in range(len(calls))}

    print("[4/6] Labeling clusters with LLM...")
    cluster_labels = label_clusters(labels.tolist(), call_texts)
    for cid, info in sorted(cluster_labels.items()):
        if cid != -1:
            print(f"        cluster_{cid:2d}: {info['intent_name']!r} ({info['size']} calls)")

    graph = build_graph(calls, call_cluster_map, cluster_labels)
    print(f"      Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    print("[5/6] Detecting gaps...")
    gaps = detect_gaps(graph, cluster_labels)
    metric = compute_headline_metric(graph, gaps)
    print(f"      Headline: {metric['gap_calls']}/{metric['total_calls']} calls "
          f"({metric['gap_rate']*100:.1f}%) hit a gap")
    for g in gaps:
        print(f"        GAP: {g['intent_name']!r} — {g['call_count']} calls, "
              f"{g['transfer_rate']*100:.0f}% transfer, sim={g.get('max_flow_similarity', '?')}")

    print("[6/6] Generating node specs for gaps...")
    gaps_with_specs = generate_all_node_specs(gaps, labels.tolist(), call_texts)

    analysis = {
        "metric": metric,
        "graph": graph,
        "gaps": gaps_with_specs,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "call_cluster_map": call_cluster_map,
    }

    out_path = ROOT / "data" / "analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] analysis.json written to {out_path}")
    return analysis


if __name__ == "__main__":
    run_pipeline()
