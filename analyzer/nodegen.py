"""
Generate Agent Flow node specs for each detected gap cluster.
Uses Google Gemini (gemini-2.0-flash) with the node schema from BUILD.md.
Without an API key, a template spec is filled in from the gap itself.
"""

import re
from collections import defaultdict
from pathlib import Path

import yaml

from analyzer import llm

_DIST_PATH = Path(__file__).parent.parent / "gen" / "distribution.yaml"


def _load_clinic_name() -> str:
    try:
        with open(_DIST_PATH) as f:
            return yaml.safe_load(f)["clinic"]["name"]
    except Exception:
        return "a multispecialty clinic"


NODE_SPEC_PROMPT = """You are building a node spec for a voice AI Agent Flow (similar to a dialog manager).
The flow is for an Indian clinic front-desk voice agent called "{clinic_name}".

A gap has been detected: callers are asking about "{intent_name}" but the flow has no node for it.
Description: {description}

Sample caller utterances that hit this gap:
{utterances}

Generate a YAML node spec matching this exact schema:
```yaml
node:
  id: "<snake_case_id>"
  intent: "<one sentence describing what the caller wants>"
  instructions: "<2-3 sentence agent instruction: what to ask, how to handle, what to do if unavailable>"
  tools: ["<tool1>", "<tool2>"]
  output_variables: ["<var1>", "<var2>"]
  edges:
    - from: "intent_classification"
      condition: "intent == <id>"
    - to: "anything_else"
      condition: "<resolved_condition> == true"
```

Rules:
- tools should be plausible clinic tools (e.g. get_lab_report, send_report_link, lookup_doctor_schedule)
- output_variables should capture what the agent collects
- instructions should sound warm and professional, with a fallback to transfer_to_human if needed
- Output ONLY the yaml block, no explanation."""


def _template_node_spec(gap: dict) -> str:
    """Skeleton spec for review when no LLM is available — same schema as the prompt."""
    node_id = re.sub(r"[^a-z0-9_]", "_", gap["intent_name"].lower()).strip("_") or f"gap_{gap['cluster_id']}"
    spec = {"node": {
        "id": node_id,
        "intent": gap.get("description") or f"Callers asking about {node_id.replace('_', ' ')}",
        "instructions": "TODO: describe what the agent should ask and do. "
                        "Fall back to transfer_to_human if the request cannot be resolved.",
        "tools": [],
        "output_variables": [],
        "edges": [
            {"from": "intent_classification", "condition": f"intent == {node_id}"},
            {"to": "anything_else", "condition": f"{node_id}_resolved == true"},
        ],
    }}
    return "# template spec (no LLM) — fill in the TODOs\n" + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def generate_node_spec(
    gap: dict,
    sample_turns: list[str],
    n_samples: int = 6,
    org_name: str | None = None,
) -> str:
    if not llm.available():
        return _template_node_spec(gap)
    utterances = "\n".join(f"- {t}" for t in sample_turns[:n_samples])
    prompt = NODE_SPEC_PROMPT.format(
        clinic_name=org_name or _load_clinic_name(),
        intent_name=gap["intent_name"],
        description=gap["description"],
        utterances=utterances,
    )
    try:
        raw = llm.generate(prompt, temperature=0.3, max_output_tokens=600)
        # strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return raw
    except Exception as e:
        return f"# node spec generation failed: {e}\nnode:\n  id: {gap['intent_name']}\n"


def generate_all_node_specs(
    gaps: list[dict],
    turn_labels: list[int],
    all_turns: list[str],
    org_name: str | None = None,
) -> list[dict]:
    """
    For each gap, gather representative turns and generate a node spec.
    Returns gaps list with 'node_spec' field added.
    """
    label_to_turns: dict[int, list[str]] = defaultdict(list)
    for label, text in zip(turn_labels, all_turns):
        label_to_turns[label].append(text)

    enriched = []
    for gap in gaps:
        cid = gap["cluster_id"]
        sample_turns = label_to_turns.get(cid, [])
        print(f"  Generating node spec for: {gap['intent_name']} ({len(sample_turns)} turns)")
        spec = generate_node_spec(gap, sample_turns, org_name=org_name)
        enriched.append({**gap, "node_spec": spec, "sample_turns": sample_turns[:6]})

    return enriched
