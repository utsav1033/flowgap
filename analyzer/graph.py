"""
Build node/edge transition graph from cluster assignments.
Nodes = discovered intent clusters + hub + terminal.
Edges = intent_classification → cluster (hub-to-spoke) and
        cluster → transfer_to_human (for clusters with transfers).
All edge endpoints match node IDs exactly — no phase name mismatches.
"""

from collections import defaultdict


def build_graph(
    calls: list[dict],
    call_cluster_map: dict[str, int],
    cluster_labels: dict[int, dict],
) -> dict:
    """
    calls: output of parse.load_calls()
    call_cluster_map: {call_id: cluster_id} — one cluster per call
    cluster_labels: output of label.label_clusters()

    Returns graph dict with nodes and edges.
    Node IDs and edge from/to all use cluster intent_name strings.
    """
    transfer_counts: dict[int, int] = defaultdict(int)
    call_counts: dict[int, int] = defaultdict(int)

    for call in calls:
        cid = call_cluster_map.get(call["call_id"], -1)
        call_counts[cid] += 1
        if call["ended_in_transfer"]:
            transfer_counts[cid] += 1

    total_calls = sum(call_counts.values())
    transfer_total = sum(transfer_counts.values())

    # ── Cluster nodes ─────────────────────────────────────────────────────────
    nodes = []
    for cid, info in cluster_labels.items():
        nodes.append({
            "id":            info["intent_name"],
            "cluster_id":   cid,
            "label":        info["intent_name"],
            "description":  info["description"],
            "call_count":   call_counts.get(cid, 0),
            "transfer_count": transfer_counts.get(cid, 0),
            "transfer_rate": (
                transfer_counts.get(cid, 0) / call_counts[cid]
                if call_counts.get(cid, 0) > 0 else 0.0
            ),
        })

    # ── Hub node ──────────────────────────────────────────────────────────────
    nodes.append({
        "id":            "intent_classification",
        "cluster_id":   -2,
        "label":        "intent_classification",
        "description":  "Routes incoming caller intent to the appropriate cluster",
        "call_count":   total_calls,
        "transfer_count": 0,
        "transfer_rate": 0.0,
    })

    # ── Terminal transfer node ─────────────────────────────────────────────────
    if transfer_total > 0:
        nodes.append({
            "id":            "transfer_to_human",
            "cluster_id":   -3,
            "label":        "transfer_to_human",
            "description":  "Calls escalated to a human agent",
            "call_count":   transfer_total,
            "transfer_count": transfer_total,
            "transfer_rate": 1.0,
        })

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Both endpoints are cluster intent_name IDs — guaranteed to match node IDs.
    edges = []
    for cid, info in cluster_labels.items():
        if cid == -1:
            continue  # skip noise cluster — don't wire noise to hub
        intent_name = info["intent_name"]
        cc = call_counts.get(cid, 0)
        tc = transfer_counts.get(cid, 0)
        if cc > 0:
            edges.append({"from": "intent_classification", "to": intent_name, "count": cc})
        if tc > 0 and transfer_total > 0:
            edges.append({"from": intent_name, "to": "transfer_to_human", "count": tc})

    return {
        "nodes": nodes,
        "edges": edges,
        "total_calls": total_calls,
        "total_transfers": transfer_total,
        "transfer_rate": transfer_total / total_calls if total_calls > 0 else 0.0,
    }


def assign_call_clusters(
    calls: list[dict],
    turn_labels: list[int],
) -> dict[str, int]:
    """
    Assign each call its majority cluster based on its caller turns.
    Returns {call_id: cluster_id}.
    """
    from collections import Counter
    idx = 0
    result = {}
    for call in calls:
        n = len(call["caller_turns"])
        call_turn_labels = turn_labels[idx:idx + n]
        idx += n
        if not call_turn_labels:
            result[call["call_id"]] = -1
            continue
        counter = Counter(call_turn_labels)
        non_noise = {k: v for k, v in counter.items() if k != -1}
        if non_noise:
            result[call["call_id"]] = max(non_noise, key=non_noise.get)
        else:
            result[call["call_id"]] = -1
    return result
