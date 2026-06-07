"""
Build node/edge transition graph from call phase sequences.
Nodes = discovered intent clusters. Edges = phase->phase transitions with counts.
"""

from collections import defaultdict


def build_graph(
    calls: list[dict],
    call_cluster_map: dict[str, int],
    cluster_labels: dict[int, dict],
) -> dict:
    """
    calls: output of parse.load_calls()
    call_cluster_map: {call_id: cluster_id} — majority cluster for that call
    cluster_labels: output of label.label_clusters()

    Returns graph dict with nodes and edges.
    """
    edge_counts: dict[tuple, int] = defaultdict(int)
    transfer_counts: dict[int, int] = defaultdict(int)
    call_counts: dict[int, int] = defaultdict(int)

    for call in calls:
        cid = call_cluster_map.get(call["call_id"], -1)
        call_counts[cid] += 1
        if call["ended_in_transfer"]:
            transfer_counts[cid] += 1

        phases = call["agent_phases"]
        for i in range(len(phases) - 1):
            src = phases[i]
            dst = phases[i + 1]
            edge_counts[(src, dst)] += 1

    nodes = []
    for cid, info in cluster_labels.items():
        nodes.append({
            "id": info["intent_name"],
            "cluster_id": cid,
            "label": info["intent_name"],
            "description": info["description"],
            "call_count": call_counts.get(cid, 0),
            "transfer_count": transfer_counts.get(cid, 0),
            "transfer_rate": (
                transfer_counts.get(cid, 0) / call_counts[cid]
                if call_counts.get(cid, 0) > 0 else 0.0
            ),
        })

    edges = []
    for (src, dst), count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        edges.append({"from": src, "to": dst, "count": count})

    total_calls = sum(call_counts.values())
    transfer_total = sum(transfer_counts.values())

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
        # majority vote, excluding noise (-1) if possible
        counter = Counter(call_turn_labels)
        non_noise = {k: v for k, v in counter.items() if k != -1}
        if non_noise:
            result[call["call_id"]] = max(non_noise, key=non_noise.get)
        else:
            result[call["call_id"]] = -1
    return result
