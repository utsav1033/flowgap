"""
Generate synthetic clinic call transcripts using Google Gemini.
Writes N transcripts to data/transcripts/ and data/ground_truth.json.

Failure policy: each call is retried 3x with exponential backoff. If all retries
fail, the call is SKIPPED — nothing is written to disk. No "[generation failed]"
placeholders. A post-generation validation pass confirms every file on disk is valid.
"""

import json
import os
import re
import random
import time
import uuid
from pathlib import Path

import yaml
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DIST_PATH = ROOT / "gen" / "distribution.yaml"
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth.json"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.0-flash"


def build_system_prompt(disfluency: bool, code_switch: bool, paraphrase: bool) -> str:
    parts = [
        "You are a realistic call transcript generator for an Indian multispecialty clinic voice line.",
        "Generate natural phone call transcripts between a caller and an AI voice agent.",
        "Format: alternate lines of CALLER: and AGENT: turns. Start with AGENT: greeting. End clearly.",
        "Use varied Indian names and 10-digit phone numbers starting with 9 or 8.",
    ]
    if disfluency:
        parts.append('Include natural disfluencies: "umm", "uh", "actually", false starts, self-corrections.')
    if code_switch:
        parts.append('Mix Hindi-English code-switching (e.g., "haan", "theek hai", "bhaiya", "ji", "ek second", "thoda wait karo").')
    if paraphrase:
        parts.append("Use varied, natural phrasing for the caller's need — never open two calls the same way.")
    return "\n".join(parts)


HANDLED_TEMPLATE = """Generate a realistic {min_turns}-{max_turns} turn phone call for intent: {intent}
Clinic: {clinic_name}
The call RESOLVES successfully. The agent handles it fully — no human transfer.
End with the agent confirming completion.
"""

GAP_TEMPLATE = """Generate a realistic {min_turns}-{max_turns} turn phone call for intent: {intent}
Clinic: {clinic_name}
The agent has NO process for this. The call MUST end with the agent transferring to a human receptionist (transfer_to_human).
The caller should be mildly frustrated. The agent apologises and offers the transfer.
"""


def load_distribution():
    with open(DIST_PATH) as f:
        data = yaml.safe_load(f)
    clinic = data["clinic"]
    gen = data["generation"]
    intents = data["intents"]
    total = gen["total_calls"]
    weights = [i["weight"] for i in intents]
    total_weight = sum(weights)
    for i in intents:
        i["count"] = round(i["weight"] / total_weight * total)
    diff = total - sum(i["count"] for i in intents)
    intents[0]["count"] += diff
    return clinic, gen, intents, total


def generate_transcript(intent: dict, clinic: dict, gen: dict) -> str:
    """Generate one transcript via Gemini. Retries 3x with backoff. Raises RuntimeError on final failure."""
    is_handled = intent.get("handled", True)
    template = HANDLED_TEMPLATE if is_handled else GAP_TEMPLATE
    min_t, max_t = gen["turns_per_call"]
    prompt = template.format(
        intent=intent["id"].replace("_", " "),
        clinic_name=clinic["name"],
        min_turns=min_t,
        max_turns=max_t,
    )
    use_code_switch = random.random() < gen["code_switch_ratio"]
    system = build_system_prompt(gen["disfluency"], use_code_switch, gen["paraphrase_intents"])

    model = genai.GenerativeModel(model_name=MODEL, system_instruction=system)
    generation_config = {"temperature": gen["temperature"], "max_output_tokens": 800}

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = model.generate_content(prompt, generation_config=generation_config)
            return resp.text.strip()
        except Exception as e:
            last_exc = e
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
            print(f"    API error (attempt {attempt + 1}/3) [{intent['id']}]: {e}")
            if attempt < 2:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"All 3 attempts failed for {intent['id']}: {last_exc}")


def parse_ended_in_transfer(transcript: str) -> bool:
    lower = transcript.lower()
    return any(
        phrase in lower
        for phrase in [
            "transfer", "connecting you", "human", "receptionist",
            "put you through", "someone who can", "colleague",
        ]
    )


def validate_transcripts(transcripts_dir: Path) -> tuple[int, list[str]]:
    """Returns (valid_count, invalid_filenames). A valid transcript has CALLER: and AGENT: turns."""
    valid, invalid = 0, []
    for path in sorted(transcripts_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            t = record.get("transcript", "")
            has_caller = bool(re.search(r"^CALLER:", t, re.MULTILINE | re.IGNORECASE))
            has_agent = bool(re.search(r"^AGENT:", t, re.MULTILINE | re.IGNORECASE))
            if has_caller and has_agent and len(t) > 50:
                valid += 1
            else:
                invalid.append(path.name)
        except Exception as e:
            invalid.append(f"{path.name} (parse error: {e})")
    return valid, invalid


def main():
    clinic, gen, intents, total = load_distribution()

    # Always start clean — generate.py is a full-rebuild script
    stale = list(TRANSCRIPTS_DIR.glob("*.json"))
    if stale:
        print(f"Clearing {len(stale)} existing transcripts for a fresh run...")
        for f in stale:
            f.unlink()
    if GROUND_TRUTH_PATH.exists():
        GROUND_TRUTH_PATH.unlink()

    ground_truth: dict = {}
    skipped: list[dict] = []
    generated = 0

    print(f"Generating {total} transcripts for {clinic['name']}...")

    for intent in intents:
        n = intent["count"]
        is_gap = not intent.get("handled", True)
        print(f"  {intent['id']} (gap={is_gap}): {n} calls")
        for _ in range(n):
            call_id = str(uuid.uuid4())[:8]
            try:
                transcript = generate_transcript(intent, clinic, gen)
            except RuntimeError as e:
                print(f"    SKIP {call_id}: {e}")
                skipped.append({"intent": intent["id"], "call_id": call_id})
                continue  # nothing written to disk

            ended_in_transfer = (
                intent.get("ends_in_transfer", False)
                or parse_ended_in_transfer(transcript)
            )

            record = {
                "call_id": call_id,
                "intent": intent["id"],
                "is_gap": is_gap,
                "ended_in_transfer": ended_in_transfer,
                "transcript": transcript,
            }
            (TRANSCRIPTS_DIR / f"{call_id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ground_truth[call_id] = {
                "intent": intent["id"],
                "is_gap": is_gap,
                "ended_in_transfer": ended_in_transfer,
            }
            generated += 1
            if generated % 20 == 0:
                print(f"    ...{generated} written so far")

    GROUND_TRUTH_PATH.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    gap_calls = sum(1 for v in ground_truth.values() if v["is_gap"])
    transfer_calls = sum(1 for v in ground_truth.values() if v["ended_in_transfer"])
    pct = gap_calls / generated * 100 if generated else 0
    print(f"\n✅ {generated} transcripts written — {gap_calls} gap calls ({pct:.1f}%), {transfer_calls} ended in transfer")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} calls (all retries failed):")
        for s in skipped:
            print(f"    {s['call_id']}  intent={s['intent']}")
    print(f"   Transcripts: {TRANSCRIPTS_DIR}")
    print(f"   Ground truth: {GROUND_TRUTH_PATH}")

    # Post-generation validation
    print("\n--- Validation ---")
    valid, invalid_files = validate_transcripts(TRANSCRIPTS_DIR)
    print(f"Valid transcripts: {valid} / {generated}")
    if invalid_files:
        print(f"Invalid ({len(invalid_files)}):")
        for name in invalid_files:
            print(f"  {name}")
        threshold = max(5, int(generated * 0.05))
        if len(invalid_files) > threshold:
            raise SystemExit(
                f"\n❌ {len(invalid_files)} invalid transcripts exceed 5% threshold ({threshold}). "
                "Check Gemini output — something is wrong with the format."
            )
    else:
        print("All transcripts valid.")


if __name__ == "__main__":
    main()
