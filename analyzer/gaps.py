"""
Detect gaps: discovered intent clusters not covered by intended_flow.yaml.

Matching is centroid-based, not label-based:
  - Each cluster's centroid = mean of its member call embeddings (384-dim MiniLM).
  - Flow nodes are embedded from their id + tool descriptions for richer context.
  - A cluster is "in flow" if its centroid has cosine similarity >= SIM_THRESHOLD
    to any non-utility flow node.

This is robust to garbage LLM/keyword labels because it operates on the actual
call content, not the label string.

transfer_rate is kept as a severity/ranking field on gaps, NOT part of the in-flow
decision (that would be circular -- it re-reads planted ground truth).
"""

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INTENDED_FLOW_PATH = ROOT / "gen" / "intended_flow.yaml"

SIM_THRESHOLD = 0.60  # configurable; print scores so caller can tune

# Routing/utility nodes -- excluded from the "does this cluster match the flow?" check
_UTILITY_NODES = {"greeting", "intent_classification", "anything_else", "transfer_to_human"}


def load_intended_nodes() -> list[dict]:
    """
    Return non-utility flow nodes as {id, text} dicts.
    'text' uses the node's description field when present (rich natural-language +
    Hinglish phrases that reflect how callers actually express that intent), falling
    back to the node id when absent. Tools are appended as extra signal either way.
    """
    with open(INTENDED_FLOW_PATH) as f:
        flow = yaml.safe_load(f)
    nodes = []
    for n in flow.get("nodes", []):
        nid = n["id"]
        if nid in _UTILITY_NODES:
            continue
        desc = n.get("description", "")
        base = desc.strip() if desc else nid.replace("_", " ")
        tool_text = " ".join(t.replace("_", " ") for t in n.get("tools", []))
        text = f"{base} {tool_text}".strip()
        nodes.append({"id": nid, "text": text})
    return nodes


def detect_gaps(
    graph: dict,
    cluster_labels: dict[int, dict],
    cluster_centroids: dict[int, np.ndarray],
    sim_threshold: float = SIM_THRESHOLD,
) -> list[dict]:
    """
    Parameters
    ----------
    graph             : output of graph.build_graph()
    cluster_labels    : {cluster_id: {intent_name, description, size}}
    cluster_centroids : {cluster_id: mean_embedding_vector (384-dim)}
    sim_threshold     : centroid cosine sim to best flow node required to be "in flow"

    Returns list of gap dicts sorted by call_count desc.
    """
    from analyzer.embed import embed_texts

    flow_nodes = load_intended_nodes()
    if not flow_nodes:
        return []

    # Embed flow node descriptions in the same MiniLM 384-dim space as call embeddings
    fv = embed_texts([n["text"] for n in flow_nodes])
    fv_n = fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-9)
    flow_ids = [n["id"] for n in flow_nodes]

    candidates = [
        node for node in graph["nodes"]
        if node["cluster_id"] != -1 and node["call_count"] >= 3
    ]
    if not candidates:
        return []

    print(f"      Centroid similarity to flow (threshold={sim_threshold}):")

    gaps = []
    for node in candidates:
        cid = node["cluster_id"]
        centroid = cluster_centroids.get(cid)
        if centroid is None:
            continue

        # Normalise centroid and compute cosine similarity to every flow node
        c_n = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = c_n @ fv_n.T        # shape (n_flow_nodes,)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        best_node = flow_ids[best_idx]

        in_flow = max_sim >= sim_threshold
        verdict = "[in flow]" if in_flow else "[GAP]   "
        print(
            f"        cluster_{cid:2d}  {node['label']!r:32s}  "
            f"best={best_node!r:25s}  sim={max_sim:.3f}  {verdict}"
        )

        if not in_flow:
            gaps.append({
                "cluster_id": cid,
                "intent_name": node["label"],
                "description": node["description"],
                "call_count": node["call_count"],
                "transfer_count": node["transfer_count"],
                "transfer_rate": node["transfer_rate"],
                "in_intended_flow": False,
                "max_flow_similarity": round(max_sim, 3),
                "best_flow_match": best_node,
                "reason": "not_in_flow",
            })

    gaps.sort(key=lambda g: -g["call_count"])
    return gaps


def compute_headline_metric(graph: dict, gaps: list[dict]) -> dict:
    total = graph["total_calls"]
    gap_calls = sum(g["call_count"] for g in gaps)
    return {
        "total_calls": total,
        "gap_calls": gap_calls,
        "gap_rate": gap_calls / total if total > 0 else 0.0,
        "n_gaps": len(gaps),
    }
