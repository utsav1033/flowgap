"""
Label each cluster with a human intent name.
Uses Google Gemini (gemini-2.0-flash) with retry-backoff on rate limits.
Falls back to top-keyword extraction if all retries fail or no API key is set.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyzer import llm

LABEL_PROMPT = """Label this cluster of caller utterances from an Indian clinic phone line.
Output ONLY a JSON object with exactly two keys:
  "intent_name": snake_case, max 3 words
  "description": one plain-English sentence

Utterances:
{utterances}"""

# Filler that carries no intent: conversational English plus common Hinglish
_STOPWORDS = {
    'want','wanted','like','need','please','would','could','know','just','also','hello',
    'okay','actually','hmm','haan','thoda','acha','accha','theek','bhaiya','call','calling',
    'clinic','help','speak','tell','check','ask','about','some','little','really','sorry',
    'generation','failed','thank','thanks','yeah','umm','that','this','there','what','your',
    'have','with','from','will','when','which','been','were','they','them','then','than',
    'into','only','should','think','said','says','right','sure','fine','good','maam','madam',
    'nahi','mujhe','mera','meri','mere','karna','karni','karne','chahiye','liye','kar','hoga',
    'raha','rahi','rahe','tha','thi','bhi','toh','kuch','abhi','woh','yeh','kaise','kitna',
    'wala','wali','koi','sakte','sakta','sakti','batana','bata','dijiye','humko','naam',
    'number','aapka','aapke','aapki','matlab','hain','haina','mein','kyunki','lekin','phir',
    'agar','kaun','kahan','yahan','wahan','dekh','dekhiye','boliye','achha','arre','bas',
    'great','regarding','haven','maybe','main','hoon','shukriya','dhanyavaad','dhanyawad',
    'bahut','theek','karenge','karunga','karungi','chahta','chahti','pata','sakenge',
}


def _distinctive_keywords(buckets: dict[int, list[str]], k: int = 3) -> dict[int, list[str]]:
    """
    Class-based TF-IDF: words frequent in this cluster but rare in the others.
    Raw counts pick filler ("should", "your"); this picks what sets a cluster apart
    ("surgery", "wound", "fever").
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    cids = [c for c in sorted(buckets) if c != -1]
    if not cids:
        return {}
    vec = CountVectorizer(
        token_pattern=r"\b[a-z]{4,}\b",
        stop_words=list(ENGLISH_STOP_WORDS | _STOPWORDS),
    )
    vec.fit(t.lower() for c in cids for t in buckets[c])
    vocab = vec.get_feature_names_out()
    counts = np.zeros((len(cids), len(vocab)))
    spread = np.zeros_like(counts)   # share of the cluster's calls that use each word
    for i, c in enumerate(cids):
        per_call = vec.transform(t.lower() for t in buckets[c]).toarray()
        counts[i] = per_call.sum(axis=0)
        spread[i] = (per_call > 0).mean(axis=0)

    tf = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
    avg_words = counts.sum() / len(cids)
    idf = np.log(1 + avg_words / (counts.sum(axis=0) + 1e-9))
    # Words used by only a call or two (caller names, one-off details) don't describe the intent
    scores = tf * idf * (spread >= 0.2)
    return {
        cid: [vocab[j] for j in np.argsort(-scores[i])[:k] if scores[i, j] > 0]
        for i, cid in enumerate(cids)
    }


def _keyword_label(top: list[str]) -> dict:
    """Label from distinctive keywords when the LLM is unavailable."""
    if not top:
        return {"intent_name": "unknown_intent", "description": "Cluster content unclear."}
    return {
        "intent_name": "_".join(top),
        "description": f"Keyword-derived label: {', '.join(top)}",
    }


def _call_llm(prompt: str, max_retries: int = 3) -> str | None:
    """Call Gemini with exponential backoff on rate-limit errors. Returns raw text or None."""
    if not llm.available():
        return None
    for attempt in range(max_retries):
        try:
            return llm.generate(prompt, temperature=0.2, max_output_tokens=120)
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

    keywords = _distinctive_keywords(buckets)
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
            fb = _keyword_label(keywords.get(cid, []))
            print(f"    cluster_{cid}: keyword fallback -> {fb['intent_name']!r}")
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
