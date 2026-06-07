"""
Detect gaps: discovered intent clusters not covered by intended_flow.yaml,
or clusters with high transfer rates.

Matching is embedding-based (cosine similarity) so labels like
"lab report retrieval" correctly match against "lab_reports" in the flow,
and fallback labels like "cluster_12" correctly fail to match.
"""

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INTENDED_FLOW_PATH = ROOT / "gen" / "intended_flow.yaml"
TRANSFER_RATE_THRESHOLD = 0.5
SIM_THRESHOLD = 0.60  # cosine similarity: cluster label vs intended-flow node text


def load_intended_nodes() -> list[str]:
    """Return list of node IDs from intended_flow.yaml."""
    with open(INTENDED_FLOW_PATH) as f:
        flow = yaml.safe_load(f)
    return [n["id"] for n in flow.get("nodes", [])]


def detect_gaps(
    graph: dict,
    cluster_labels: dict[int, dict],
) -> list[dict]:
    """
    A cluster is a gap if:
      (a) its label has cosine similarity < SIM_THRESHOLD to every intended-flow node, AND
      (b) it has meaningful call volume (>= 3 calls)
      OR (c) its transfer_rate > TRANSFER_RATE_THRESHOLD regardless of (a)

    Returns list of gap dicts sorted by call_count desc.
    """
    from analyzer.embed import embed_texts

    intended_ids = load_intended_nodes()
    # readable text for embedding
    flow_texts = [n.replace("_", " ") for n in intended_ids]

    candidates = [
        node for node in graph["nodes"]
        if node["cluster_id"] != -1 and node["call_count"] >= 3
    ]
    if not candidates:
        return []

    cand_texts = [node["label"].replace("_", " ") for node in candidates]

    # Single embedding batch: all candidate labels + all flow node texts
    vecs = embed_texts(cand_texts + flow_texts)
    cv = vecs[:len(cand_texts)]
    fv = vecs[len(cand_texts):]

    # Normalised cosine similarity matrix  (n_candidates × n_flow_nodes)
    cv_n = cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-9)
    fv_n = fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-9)
    sims = cv_n @ fv_n.T
    max_sims = sims.max(axis=1)  # best flow-node match per candidate

    gaps = []
    for i, node in enumerate(candidates):
        max_sim = float(max_sims[i])
        in_flow = max_sim >= SIM_THRESHOLD
        high_transfer = node["transfer_rate"] >= TRANSFER_RATE_THRESHOLD

        if not in_flow or high_transfer:
            gaps.append({
                "cluster_id": node["cluster_id"],
                "intent_name": node["label"],
                "description": node["description"],
                "call_count": node["call_count"],
                "transfer_count": node["transfer_count"],
                "transfer_rate": node["transfer_rate"],
                "in_intended_flow": in_flow,
                "max_flow_similarity": round(max_sim, 3),
                "reason": (
                    "not_in_flow" if not in_flow
                    else "high_transfer_rate" if high_transfer
                    else "both"
                ),
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
