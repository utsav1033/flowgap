"""
Load transcripts and extract caller turns + per-call phase sequence.
Output: list of dicts with call_id, intent, is_gap, caller_turns, phases.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth.json"


def load_ground_truth() -> dict:
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


def extract_turns(transcript: str) -> tuple[list[str], list[str]]:
    """Return (caller_turns, agent_turns) from a transcript string."""
    caller, agent = [], []
    for line in transcript.splitlines():
        line = line.strip()
        if line.upper().startswith("CALLER:"):
            text = re.sub(r"^CALLER:\s*", "", line, flags=re.IGNORECASE).strip()
            if text:
                caller.append(text)
        elif line.upper().startswith("AGENT:"):
            text = re.sub(r"^AGENT:\s*", "", line, flags=re.IGNORECASE).strip()
            if text:
                agent.append(text)
    return caller, agent


_TRANSFER_PHRASES = ["transfer", "connect you", "human", "receptionist", "put you through"]


def _agent_ends_in_transfer(agent_turns: list[str]) -> bool:
    """Re-derive transfer status from last two agent turns."""
    for turn in agent_turns[-2:]:
        if any(p in turn.lower() for p in _TRANSFER_PHRASES):
            return True
    return False


def infer_phases(agent_turns: list[str]) -> list[str]:
    """
    Heuristic phase labels from agent turns — used for transition graph.
    These are coarse: greeting / information_gathering / resolution / transfer / closing.
    """
    phases = []
    for turn in agent_turns:
        t = turn.lower()
        if any(w in t for w in ["welcome", "thank you for calling", "good morning", "good afternoon", "namaskar"]):
            phases.append("greeting")
        elif any(w in t for w in ["transfer", "connect you", "human", "receptionist", "put you through"]):
            phases.append("transfer")
        elif any(w in t for w in ["is there anything else", "anything else", "have a great", "take care", "goodbye", "thank you for calling"]):
            phases.append("closing")
        elif any(w in t for w in ["could you", "can you", "please share", "what is your", "may i know", "your name", "your number", "date of birth"]):
            phases.append("information_gathering")
        else:
            phases.append("resolution")
    return phases


def load_calls() -> list[dict]:
    gt = load_ground_truth()
    calls = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        call_id = path.stem
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        caller_turns, agent_turns = extract_turns(record["transcript"])
        phases = infer_phases(agent_turns)
        ended_in_transfer = (
            record.get("ended_in_transfer", False)
            or _agent_ends_in_transfer(agent_turns)
        )
        calls.append({
            "call_id": call_id,
            "intent": record.get("intent", gt.get(call_id, {}).get("intent", "unknown")),
            "is_gap": record.get("is_gap", gt.get(call_id, {}).get("is_gap", False)),
            "ended_in_transfer": ended_in_transfer,
            "caller_turns": caller_turns,
            "agent_phases": phases,
            "transcript": record["transcript"],
        })
    return calls


if __name__ == "__main__":
    calls = load_calls()
    print(f"Loaded {len(calls)} calls")
    for c in calls[:3]:
        print(f"  {c['call_id']} intent={c['intent']} turns={len(c['caller_turns'])} phases={c['agent_phases'][:3]}")
