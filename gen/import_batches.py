"""
Import manually-generated batch transcripts into the pipeline format used by
analyzer/parse.py and eval/evaluate.py. Makes NO API calls.

Output format matches gen/generate.py exactly:
  data/transcripts/{call_id}.json  →  {call_id, intent, is_gap, ended_in_transfer, transcript}
  data/ground_truth.json           →  {call_id: {intent, is_gap, ended_in_transfer}}

call_id on disk = filename stem (e.g. "batch_041"), because parse.py uses path.stem,
not record["call_id"]. Ground truth keys must match those stems for eval to work.

Processes one batch file at a time — no batch is kept in memory past its write step.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
BATCHES_DIR = ROOT / "data" / "raw_batches"
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth.json"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def turns_to_transcript(turns: list) -> str:
    """Convert [{speaker, text}, ...] to the AGENT:/CALLER: line format parse.py expects."""
    lines = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip().upper()
        text = str(turn.get("text", "")).strip()
        if not text or speaker not in ("AGENT", "CALLER"):
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def is_broken(transcript: str) -> bool:
    return "[generation failed]" in transcript.lower()


def has_valid_turns(transcript: str) -> bool:
    has_caller = bool(re.search(r"^CALLER:", transcript, re.MULTILINE | re.IGNORECASE))
    has_agent = bool(re.search(r"^AGENT:", transcript, re.MULTILINE | re.IGNORECASE))
    return has_caller and has_agent and len(transcript.strip()) > 50


# ---------------------------------------------------------------------------
# Step 1 — delete broken transcripts
# ---------------------------------------------------------------------------

def delete_broken_transcripts() -> int:
    """Remove any transcript file that contains the generation-failure placeholder."""
    deleted = 0
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if is_broken(record.get("transcript", "")):
                path.unlink()
                deleted += 1
        except Exception:
            # Unparseable files are also stale — remove them
            path.unlink()
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Step 3 — process one batch at a time
# ---------------------------------------------------------------------------

def convert_call(obj: dict) -> dict | None:
    """
    Validate one raw call object and convert to pipeline-ready fields.
    Returns None when the object is malformed or produces an invalid transcript.
    """
    if not isinstance(obj, dict):
        return None
    turns = obj.get("turns")
    intent = obj.get("intent")
    if not intent or not isinstance(turns, list):
        return None
    transcript = turns_to_transcript(turns)
    if not has_valid_turns(transcript) or is_broken(transcript):
        return None
    return {
        "intent": str(intent).strip(),
        "is_gap": bool(obj.get("is_gap", False)),
        "ended_in_transfer": bool(obj.get("ended_in_transfer", False)),
        "transcript": transcript,
    }


def process_batch(
    batch_path: Path,
    used_ids: set[str],
) -> tuple[int, int]:
    """
    Read one batch file, write its calls to data/transcripts/, update used_ids.
    Returns (written, skipped).
    """
    try:
        data = json.loads(batch_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  SKIP {batch_path.name}: cannot parse JSON — {e}")
        return 0, 1

    # Normalise: single object or array of objects
    calls = [data] if isinstance(data, dict) else data if isinstance(data, list) else None
    if calls is None:
        print(f"  SKIP {batch_path.name}: not a JSON object or array")
        return 0, 1

    written = skipped = 0
    for i, obj in enumerate(calls):
        record = convert_call(obj)
        if record is None:
            loc = batch_path.name if len(calls) == 1 else f"{batch_path.name}[{i}]"
            print(f"    skip {loc}: malformed or no valid turns")
            skipped += 1
            continue

        # Filename = authoritative call_id (must match what parse.py uses as path.stem)
        base_id = batch_path.stem if len(calls) == 1 else f"{batch_path.stem}_{i:02d}"
        call_id = base_id
        dup = 0
        while call_id in used_ids:
            dup += 1
            call_id = f"{base_id}_dup{dup}"
        if dup:
            print(f"    dedup {base_id!r} → {call_id!r}")
        used_ids.add(call_id)

        out = {
            "call_id": call_id,
            "intent": record["intent"],
            "is_gap": record["is_gap"],
            "ended_in_transfer": record["ended_in_transfer"],
            "transcript": record["transcript"],
        }
        (TRANSCRIPTS_DIR / f"{call_id}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# Step 4 — rebuild ground_truth from all transcripts on disk
# ---------------------------------------------------------------------------

def rebuild_ground_truth() -> tuple[dict, list[str]]:
    """
    Scan every transcript file and rebuild ground_truth.json.
    Uses path.stem as call_id (same as parse.py) so eval matching is correct.
    Returns (ground_truth_dict, list_of_errors).
    """
    ground_truth: dict = {}
    errors: list[str] = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        call_id = path.stem
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            t = record.get("transcript", "")
            if is_broken(t):
                errors.append(f"{path.name}: still contains '[generation failed]'")
                continue
            if not t.strip():
                errors.append(f"{path.name}: transcript is empty")
                continue
            ground_truth[call_id] = {
                "intent": record.get("intent", "unknown"),
                "is_gap": bool(record.get("is_gap", False)),
                "ended_in_transfer": bool(record.get("ended_in_transfer", False)),
            }
        except Exception as e:
            errors.append(f"{path.name}: {e}")
    return ground_truth, errors


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(ground_truth: dict, rebuild_errors: list[str]) -> None:
    total = len(ground_truth)
    if total == 0:
        print("\n❌ ground_truth is empty — nothing was imported.")
        sys.exit(1)

    intent_counts = Counter(v["intent"] for v in ground_truth.values())
    gap_calls = sum(1 for v in ground_truth.values() if v["is_gap"])
    transfer_calls = sum(1 for v in ground_truth.values() if v["ended_in_transfer"])
    gap_intents = {v["intent"] for v in ground_truth.values() if v["is_gap"]}

    print(f"\n{'='*50}")
    print(f"  Total calls on disk:   {total}")
    print(f"  Gap calls:             {gap_calls}  ({gap_calls/total*100:.1f}%)")
    print(f"  Transfer-ending calls: {transfer_calls}")
    print(f"  Gap intents:           {', '.join(sorted(gap_intents))}")
    print(f"\n  Per-intent breakdown:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        tag = "  [GAP]" if intent in gap_intents else ""
        print(f"    {intent:<35s}  {count:>4d}{tag}")

    # Hard checks
    errors = list(rebuild_errors)

    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        if "[generation failed]" in raw.lower():
            errors.append(f"{path.name}: contains '[generation failed]'")
        if not path.stat().st_size:
            errors.append(f"{path.name}: empty file")

    print(f"{'='*50}")
    if errors:
        print(f"\n❌ Validation FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)

    print(f"\n[OK] All {total} transcripts valid -- zero empty, zero '[generation failed]'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== FlowGap Batch Importer ===\n")

    # 1 — clean up stale transcripts
    deleted = delete_broken_transcripts()
    print(f"[1/4] Deleted {deleted} broken transcripts from data/transcripts/\n")

    # 2 — discover batch files
    batch_files = sorted(BATCHES_DIR.glob("batch_*.json"))
    if not batch_files:
        sys.exit(f"❌ No batch files found in {BATCHES_DIR}")
    print(f"[2/4] Found {len(batch_files)} batch files in data/raw_batches/\n")

    # 3 — process one batch at a time
    print(f"[3/4] Importing batches (one file at a time)...")
    used_ids: set[str] = set()
    total_written = total_skipped = 0

    for batch_path in batch_files:
        written, skipped = process_batch(batch_path, used_ids)
        total_written += written
        total_skipped += skipped
        status = f"wrote {written}"
        if skipped:
            status += f", skipped {skipped}"
        print(f"  {batch_path.name}: {status}")

    print(f"\n  Batches done — written: {total_written}, skipped: {total_skipped}\n")

    # 4 — rebuild ground_truth.json
    print(f"[4/4] Rebuilding data/ground_truth.json from all transcripts on disk...")
    ground_truth, rebuild_errors = rebuild_ground_truth()
    GROUND_TRUTH_PATH.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      {len(ground_truth)} entries written to ground_truth.json\n")

    # Validation
    validate(ground_truth, rebuild_errors)

    print(f"\n  data/transcripts/    : {TRANSCRIPTS_DIR}")
    print(f"  data/ground_truth.json: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
