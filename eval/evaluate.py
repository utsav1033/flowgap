"""
Phase 6 Eval: compare detected gaps to ground_truth.json.

Matching strategy: for each detected gap cluster, look up which calls belong to
it (via call_cluster_map saved in analysis.json), find the dominant planted intent
of those calls from ground_truth.json (majority vote), and treat the cluster as a
true positive if the dominant intent is a planted gap.

This is ground-truth-based, not label-string-based, so it works regardless of
whether label.py returned a good human-readable name or a fallback "cluster_N".
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth.json"
ANALYSIS_PATH = ROOT / "data" / "analysis.json"


def evaluate(analysis: dict | None = None) -> dict:
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt = json.load(f)

    if analysis is None:
        with open(ANALYSIS_PATH, encoding="utf-8") as f:
            analysis = json.load(f)

    gt_gap_intents = {v["intent"] for v in gt.values() if v["is_gap"]}

    # Build cluster_id → [planted_intents] using the call→cluster map saved in analysis.json
    raw_map = analysis.get("call_cluster_map", {})
    if not raw_map:
        raise ValueError(
            "analysis.json has no 'call_cluster_map' key — re-run analyzer/run.py to regenerate it"
        )
    call_cluster_map: dict[str, int] = {k: int(v) for k, v in raw_map.items()}

    cluster_to_intents: dict[int, list[str]] = defaultdict(list)
    for call_id, cluster_id in call_cluster_map.items():
        if call_id in gt:
            cluster_to_intents[cluster_id].append(gt[call_id]["intent"])

    detected_gaps = analysis.get("gaps", [])
    detected_gt_intents: set[str] = set()
    rows = []

    for gap in detected_gaps:
        cid = int(gap["cluster_id"])
        intents = cluster_to_intents.get(cid, [])
        if intents:
            counter = Counter(intents)
            dominant, dom_count = counter.most_common(1)[0]
            dom_frac = dom_count / len(intents)
        else:
            dominant, dom_frac = "unknown", 0.0

        is_tp = dominant in gt_gap_intents
        if is_tp:
            detected_gt_intents.add(dominant)

        rows.append({
            "cluster_id": cid,
            "detected_label": gap["intent_name"],
            "size": gap["call_count"],
            "dominant_planted_intent": dominant,
            "dominant_fraction": round(dom_frac, 2),
            "is_true_positive": is_tp,
        })

    tp = len(detected_gt_intents & gt_gap_intents)
    fp = sum(1 for r in rows if not r["is_true_positive"])
    fn = len(gt_gap_intents - detected_gt_intents)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "planted_gaps": sorted(gt_gap_intents),
        "cluster_breakdown": rows,
        "matched_planted_gaps": sorted(detected_gt_intents),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "planted_k": len(gt_gap_intents),
        "detected_k": len(detected_gaps),
    }


def print_eval(result: dict):
    print("\n=== FlowGap Eval ===")
    print(f"Planted gaps ({result['planted_k']}): {', '.join(result['planted_gaps'])}")
    print(f"\nCluster breakdown ({result['detected_k']} detected gaps):")
    for row in result["cluster_breakdown"]:
        marker = "✓ TP" if row["is_true_positive"] else "✗ FP"
        print(
            f"  [{marker}] cluster_{row['cluster_id']:2d}  "
            f"label: {row['detected_label']!r:35s}  "
            f"size: {row['size']:4d}  "
            f"dominant intent: {row['dominant_planted_intent']} ({row['dominant_fraction']*100:.0f}%)"
        )
    print(f"\nTP={result['true_positives']}  FP={result['false_positives']}  FN={result['false_negatives']}")
    print(f"Precision={result['precision']:.2f}  Recall={result['recall']:.2f}  F1={result['f1']:.2f}")


if __name__ == "__main__":
    result = evaluate()
    print_eval(result)
    out = ROOT / "data" / "eval_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Eval results saved to {out}")
