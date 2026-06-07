"""
Label each cluster with a human intent name.
Uses Google Gemini (gemini-2.0-flash) with retry-backoff on rate limits.
Falls back to top-keyword extraction if all retries fail.
"""

import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.0-flash"

LABEL_PROMPT = """Label this cluster of caller utterances from an Indian clinic phone line.
Output ONLY a JSON object with exactly two keys:
  "intent_name": snake_case, max 3 words
  "description": one plain-English sentence

Utterances:
{utterances}"""

_STOPWORDS = {
    'i','me','my','we','our','you','the','a','an','is','are','was','were',
    'want','like','to','of','in','for','and','or','it','this','that','can',
    'do','did','have','had','need','please','would','could','get','know',
    'just','also','hi','hello','okay','ok','yes','no','so','but','not','be',
    'has','with','from','um','uh','actually','hmm','haan','ji','thoda','ek',
    'acha','theek','bhaiya','sir','call','calling','clinic','help','speak',
    'tell','check','ask','about','some','little','very','really','sorry',
    'generation','failed',
}


def _keyword_label(turns: list[str]) -> dict:
    """Extract top keywords as label when LLM is unavailable."""
    words = []
    for t in turns:
        words.extend(re.findall(r'\b[a-z]{4,}\b', t.lower()))
    top = [w for w, _ in Counter(
        w for w in words if w not in _STOPWORDS
    ).most_common(4)]
    if not top:
        return {"intent_name": "unknown_intent", "description": "Cluster content unclear."}
    name = "_".join(top[:3])
    return {
        "intent_name": name,
        "description": f"Keyword-derived label: {', '.join(top)}",
    }


def _call_llm(prompt: str, max_retries: int = 3) -> str | None:
    """Call Gemini with exponential backoff on rate-limit errors. Returns raw text or None."""
    model = genai.GenerativeModel(model_name=MODEL)
    generation_config = {"temperature": 0.2, "max_output_tokens": 120}
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt, generation_config=generation_config)
            return resp.text.strip()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "quota" in msg or "limit" in msg:
                wait = 10 * (2 ** attempt)   # 10s → 20s → 40s
                print(f"    rate limit (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    LLM error (non-rate): {e}")
                return None
    print(f"    All {max_retries} retries exhausted — using keyword fallback")
    return None


def label_clusters(
    labels: list[int],
    texts: list[str],
    n_representatives: int = 8,
) -> dict[int, dict]:
    """
    Returns {cluster_id: {"intent_name": str, "description": str, "size": int}}.
    Cluster -1 (noise) gets a generic label.
    """
    buckets: dict[int, list[str]] = defaultdict(list)
    for label, text in zip(labels, texts):
        buckets[label].append(text)

    cluster_labels = {}
    for cid, turns in sorted(buckets.items()):
        if cid == -1:
            cluster_labels[cid] = {
                "intent_name": "noise_unclustered",
                "description": "Utterances that did not fit any cluster.",
                "size": len(turns),
            }
            continue

        representatives = turns[:n_representatives]
        utterances_str = "\n".join(f"- {t[:120]}" for t in representatives)
        prompt = LABEL_PROMPT.format(utterances=utterances_str)

        raw = _call_llm(prompt)
        parsed = None

        if raw is not None:
            # strip markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'"intent_name"\s*:\s*"([^"]+)"', raw)
                d = re.search(r'"description"\s*:\s*"([^"]+)"', raw)
                if m:
                    parsed = {
                        "intent_name": m.group(1),
                        "description": d.group(1) if d else "",
                    }

        if parsed:
            cluster_labels[cid] = {
                "intent_name": parsed.get("intent_name", f"cluster_{cid}"),
                "description": parsed.get("description", ""),
                "size": len(turns),
            }
        else:
            fb = _keyword_label(representatives)
            print(f"    cluster_{cid}: keyword fallback → {fb['intent_name']!r}")
            cluster_labels[cid] = {**fb, "size": len(turns)}

    return cluster_labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analyzer.parse import load_calls
    from analyzer.embed import embed_texts
    from analyzer.cluster import cluster_embeddings

    calls = load_calls()
    all_turns = [t for c in calls for t in c["caller_turns"]]
    vecs = embed_texts(all_turns)
    labels = cluster_embeddings(vecs)
    result = label_clusters(labels.tolist(), all_turns)
    for cid, info in result.items():
        tag = "NOISE" if cid == -1 else f"cluster_{cid}"
        print(f"  {tag}: {info['intent_name']} — {info['description']} ({info['size']} turns)")
