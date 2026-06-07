# BUILD.md — FlowGap

A coverage analyzer for voice-AI Agent Flows. Reads call transcripts, reconstructs how
calls *actually* flow as a node/edge graph, and finds where the designed flow leaks —
intents that have no node and dead-end into human transfer. For each gap, it generates a
drop-in node spec in the Agent Flow format.

Built as a demo to pitch Osvi AI (voice agents for Indian businesses; their flows are
hand-authored directed graphs with a global layer, scoped nodes, and intent-routed edges).

---

## The one thing this must prove

> "Of N calls, X% hit an intent your flow can't handle and got transferred to a human.
>  Here are the exact nodes that would recover them."

Every design choice serves that sentence. The headline metric is **% of calls lost to an
unhandled intent**, and the payload is **a generated node spec, ready to drop in**.
If a feature doesn't sharpen that, cut it.

---

## Scope

**In scope (1–2 week demo):**
- Synthetic transcript generation with a controlled, known intent distribution
- Intent discovery via embedding + clustering (NOT a fixed classifier — it must *discover*)
- Transition graph reconstruction
- Gap detection against a declared "intended flow"
- Node-spec generation for each gap
- A ground-truth eval (planted gaps → precision/recall)
- A single-page Next.js frontend: headline number → before/after graph → node spec panel

**Out of scope (do NOT build):**
- Real ASR / TTS / telephony — transcripts are text, always
- Auth, multi-tenant, accounts, databases beyond local files
- Production deploy / Docker / CI — local `dev` is enough for the demo
- Live integration with any real Osvi system
- Anything that claims to analyze Osvi's *real* calls (demo is on synthetic data, stated honestly)

---

## Stack

- **Python 3.11**, FastAPI (API), `uv` or venv
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) by default — local, free, fast.
  Make it swappable to OpenAI embeddings via an env flag.
- **Clustering:** HDBSCAN (finds natural clusters + leaves noise unclustered; no need to pick k).
  Fall back to KMeans only if HDBSCAN gives garbage on the demo set.
- **LLM (frontier):** Anthropic or OpenAI, used ONLY offline for three jobs:
  (1) generate transcripts, (2) label clusters, (3) generate node specs. Never in a hot path.
- **Frontend:** Next.js + React Flow (graph), Tailwind. One page.
- Keep API keys in `.env`. Never commit them.

---

## Repo structure

```
flowgap/
  BUILD.md                  # this file
  CLAUDE.md                 # "read BUILD.md, build phase by phase, keep the eval"
  .env.example
  gen/
    distribution.yaml       # intents, weights, which are planted gaps
    intended_flow.yaml      # the nodes/edges the "designer" built (ground truth for gaps)
    generate.py             # frontier LLM -> N transcripts + ground_truth.json
  data/
    transcripts/            # generated call transcripts (JSON)
    ground_truth.json       # which call = which intent; which intents are gaps
  analyzer/
    parse.py                # transcripts -> caller turns + per-call phase sequence
    embed.py                # caller turns -> vectors (sentence-transformers | openai)
    cluster.py              # HDBSCAN -> clusters
    label.py                # frontier LLM -> human intent name per cluster
    graph.py                # build node/edge transition graph with frequencies
    gaps.py                 # detect gaps vs intended_flow.yaml
    nodegen.py              # gap cluster -> Agent Flow node spec (frontier LLM)
    run.py                  # orchestrates the full pipeline -> analysis.json
  api/
    main.py                 # FastAPI: serves analysis.json (graph, gaps, node specs, metrics)
  web/                      # Next.js single-page demo
  eval/
    evaluate.py             # detected gaps vs ground_truth -> precision / recall
```

---

## Build phases

Build and verify each phase before starting the next. Stop and show output at each ✅.

### Phase 1 — Synthetic data
- Define `gen/distribution.yaml`: a clinic voice line. ~6 handled intents (booking, reschedule,
  insurance/TPA, billing, clinic info, prescription refill) + 2–3 **planted gap intents**
  (e.g. lab reports, post-op care question, doctor-specific availability) that the intended
  flow will NOT cover. Each intent has a weight. Gaps are ~8–12% of total.
- Define `gen/intended_flow.yaml`: the nodes + edges a designer "built" — i.e. only the handled
  intents. This is what we measure reality against.
- `gen/generate.py`: loop N times (target N≈400), sample an intent, prompt the frontier model to
  emit a realistic phone-call transcript (caller + agent turns). For handled intents the call
  resolves; for gap intents the agent must end in `transfer_to_human` / fallback. Mix in
  disfluencies ("umm", "haan"), some Hindi-English code-switching, varied names/numbers.
- Write each transcript to `data/transcripts/` and a `data/ground_truth.json` mapping
  call_id -> {intent, is_gap, ended_in_transfer}.
- ✅ Acceptance: N transcripts on disk; ground_truth.json complete; gaps are ~10% and all
  end in transfer. Spot-read 5 transcripts — they should sound like real calls.

### Phase 2 — Embed + cluster
- `parse.py`: load transcripts, extract caller turns, and the ordered phase sequence per call.
- `embed.py`: embed the caller turns (default MiniLM; OpenAI behind a flag).
- `cluster.py`: HDBSCAN over the embeddings. Output cluster_id per turn/call.
- ✅ Acceptance: number of clusters is in the right ballpark of true intents; print cluster
  sizes and a few sample turns per cluster — they should be coherent.

### Phase 3 — Label + graph
- `label.py`: for each cluster, send representative turns to the frontier model, get a short
  intent name + one-line description.
- `graph.py`: build the transition graph — nodes = discovered intents/phases, edges = observed
  phase->phase transitions with counts. Mark which calls ended in transfer.
- ✅ Acceptance: graph JSON with labeled nodes and weighted edges; transfer endpoints visible.

### Phase 4 — Gap detection
- `gaps.py`: compare discovered intents against `intended_flow.yaml`.
  A gap = a discovered intent cluster that (a) isn't in the intended flow, and/or
  (b) overwhelmingly ends in transfer/fallback. Rank by call volume.
- Compute the headline metric: % of total calls landing in a gap.
- ✅ Acceptance: the planted gap intents show up as detected gaps; the metric is sensible.

### Phase 5 — Node generation
- `nodegen.py`: for each top gap cluster, feed representative transcripts to the frontier model
  with the Agent Flow node schema (below) and emit a valid node spec: scoped instructions,
  node tools, output variables, and the routing edge that wires it into the existing flow.
- ✅ Acceptance: each gap yields a syntactically valid node spec that reads like it belongs
  in their flow.

### Phase 6 — Eval (do not skip — this is the flex)
- `eval/evaluate.py`: compare detected gaps to `ground_truth.json`. Report precision/recall:
  "planted K gaps, detected K, 0 false positives." This is what makes the demo credible.
- ✅ Acceptance: a printed eval line you can screenshot.

### Phase 7 — API + frontend
- `api/main.py`: serve `analysis.json` (metric, graph, gaps, node specs, eval).
- `web/`: ONE page, top to bottom:
  1. **Headline banner** — the big number + cost framing ("X% → human transfer").
  2. **Before/after graph** (React Flow) — intended flow vs reality; gap nodes in red;
     a toggle that "patches in" the generated nodes and clears the red.
  3. **Gap panel** — ranked gaps, each expandable to show the generated node spec.
- ✅ Acceptance: load page, see number, toggle before/after, expand a node spec. Record a
  60-second Loom narrating problem -> red appears -> patch closes it -> saved cost.

---

## Agent Flow node schema (match this exactly in nodegen)

Mirror the structure from Osvi's Agent Flows post so generated nodes look native.

```yaml
# Global layer (shared by all nodes) — for reference, not generated per-gap
global:
  objective: "Front desk agent for <clinic>"
  personality: ["warm", "patient", "respectful"]
  guardrails: ["no medical advice", "data privacy", "answer from knowledge base only"]
  universal_tools: ["end_call", "transfer_to_human", "send_voicemail"]

# A node = what nodegen.py emits per gap
node:
  id: "lab_reports"
  intent: "Caller wants to retrieve lab/diagnostic reports"
  instructions: "Help the caller get their lab report. Ask for patient ID and report date.
    Look it up; if unavailable, offer SMS/email delivery."
  tools: ["get_lab_report", "send_report_link"]
  output_variables: ["patient_id", "report_date", "delivery_method"]
  edges:
    - from: "intent_classification"   # where it wires in
      condition: "intent == lab_reports"
    - to: "anything_else"             # where it returns
      condition: "report_delivered == true"
```

---

## Key decisions (don't relitigate these)

- **Cluster, don't classify.** The point is *discovering* unforeseen intents. A fixed-label
  classifier can't find a gap it was never told about. HDBSCAN's "noise" bucket is itself a signal.
- **Frontier is offline-only.** Data gen, labeling, node gen. Never call it per-turn / in any
  latency-sensitive path — that's the whole architectural point of the pitch.
- **Gaps are defined against an explicit intended flow** (`intended_flow.yaml`), so detection is
  measurable, not vibes. Reality minus intent = gap.
- **Planting gaps is the eval, not cheating.** Known ground truth lets you report precision/recall.
- **The product is the metric + the node spec**, not the graph. The graph is the proof; the number
  is the hook; the node spec is the close.

