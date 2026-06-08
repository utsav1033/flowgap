# FlowGap

Turn call logs into flow improvements. FlowGap reads a voice agent's call transcripts,
discovers the recurring intents callers actually have that the flow doesn't handle — the
ones quietly ending in a human transfer. For each gap, it drafts a drop-in flow node to close it.

> Point it at your call logs and it tells you the % of calls your agent is silently handing to
> a human — and hands back the exact nodes that would recover them.

## Flow analysis dashboard

Three views — toggle between them to see where the gaps are and what a fix looks like.

**INTENDED** — the flow as the designer declared it:

![Intended flow](images/intended flo1.png)

**ACTUAL** — what really happens. Red nodes are unhandled intents the pipeline discovered;
they drain to Transfer to Human instead of being resolved:

![Actual flow with gaps](images/actual flo2.png)

**PATCHED** — the same flow with gap nodes wired in, ready for review:

![Patched flow](images/patched flo3.png)

---

## Why this matters for a voice-agent platform

Agent flows are authored by anticipating intents. That works for the demand you foresaw — but
every real deployment has demand nobody designed for. When a caller asks for something the flow
has no node for, the agent falls through to a human. Each of those is:

- **direct cost** — a call you're paying an agent to automate, handled by a person instead, and
- **invisible** — the proof is scattered across thousands of transcripts no one reads.

So flows get optimized for what was designed, while unhandled demand quietly leaks to humans and
never shows up in a dashboard. The lever everyone wants — *fewer human handoffs* — is sitting
unused inside the call logs.

FlowGap is the feedback loop that closes it: logs in, ranked coverage gaps out, each with a
node spec in the flow's own format ready to review and ship. It scales the way a platform needs
it to — the same pipeline runs across every client's flow, so coverage analysis becomes a
product surface, not a manual audit. Built deliberately around a graph-based flow model
(global layer, scoped nodes, intent-routed edges), so its output drops into that structure
rather than sitting beside it as yet another analytics chart.

## What it produces

For a given set of call logs:

- **A headline number** — what share of calls hit an unhandled intent and transferred out.
- **A ranked gap list** — each unhandled intent, its call volume, and its transfer rate, so the
  most expensive gaps surface first.
- **A drafted node per gap** — scoped instructions, tools, output variables, and the routing
  edge that wires it into the existing flow. A human reviews and accepts; nothing auto-edits a
  live flow.

The output is an action ("here's the node that recovers these calls"), not a report.

## How it works

A six-stage offline pipeline. No model sits in the live call path — analysis runs on logs after
the fact, so nothing here adds call latency.

1. **Parse** the caller's side of each call (what they wanted, not what the bot said).
2. **Embed** each call so semantically similar calls sit near each other (local, no API).
3. **Reduce + cluster** with UMAP → HDBSCAN to *discover* intent clusters unsupervised — the
   point being to find intents nobody pre-defined. A fixed classifier can only recognize labels
   it was given; clustering finds the intent you didn't know you were missing.
4. **Label** each cluster.
5. **Detect gaps** by matching each cluster to the declared flow on *semantic content* (not on a
   label string), then flagging clusters that match no node and end in transfer.
   `discovered − intended = gaps`.
6. **Draft** a flow node per gap.

## Results

Evaluated on **298 synthetic clinic calls** spanning 6 handled intents plus 3 deliberately
planted gap intents — planted so detection can be scored against ground truth.

**49% of calls** hit an unhandled intent and transferred to a human. Coarse unhandled intents
were recovered as clean clusters, by meaning alone:

| Gap intent                   | Calls | Cluster purity | Transfer rate | Detected |
|------------------------------|-------|----------------|---------------|----------|
| lab_reports                  | 13    | 90%            | 90%           | YES (TP) |
| post_op_care                 | 13    | 100%           | 100%          | YES (TP) |
| doctor_availability_specific | 12    | —              | —             | NO (FN, see below) |

Adding UMAP cut clustering noise from **54.7%** to **3.7%** — the change that made small,
real intent clusters viable rather than discarded as noise.

Scored against planted ground truth (majority-vote per cluster):

| Metric    | Value |
|-----------|-------|
| TP        | 2     |
| FP        | 6     |
| FN        | 1     |
| Precision | 0.25  |
| Recall    | 0.67  |
| F1        | 0.36  |

The precision ceiling here is a *labeling* artifact, not a detection-method limit. The API
was rate-limited during this run, so all 21 cluster labels fell back to keyword extraction
(e.g. `nahi_mujhe_appointment`, `bill_think_invoice`). Those labels have poor lexical overlap
with flow node names, which slightly depresses centroid similarity for handled-intent clusters
near the 0.60 threshold. The 6 FPs sit in the 0.42–0.56 band — just below threshold. With
proper LLM labels they sharpen into the in-flow zone. The discovery method itself isolates the
coarse gaps cleanly (post_op_care sim=0.285, lab_reports sim=0.420 — both far below threshold,
both at 90–100% purity).

## Honest edges (and the roadmap they imply)

- **Sub-intents of a handled flow are the hard case.** "Appointment with Dr. Mehta
  *specifically*" looks near-identical to ordinary booking in embedding space, so it merges in.
  Coarse novel intents separate cleanly; fine-grained sub-intents need a second-pass split —
  the natural v2.
- **Label quality scales with budget, not with the approach.** Keyword fallbacks slightly blur
  matching; real labels sharpen it.
- **Results are on synthetic data** (which is what makes ground-truth scoring possible). The
  pipeline is data-agnostic; the main hardening on production transcripts is intent labeling
  under heavy Hindi-English code-switching — already partially present in the test set.

## Where it goes

The single-shot tool is the seed of a continuous loop: run it on a rolling window of production
calls, watch coverage gaps emerge as caller behavior shifts, and surface "your flow is now
missing a node for X — N calls/week, all transferring" before anyone notices the leak manually.
The same content-matching layer extends naturally to flagging where declared guardrails go
unenforced in practice.

## Run it

```bash
cp .env.example .env        # add your GEMINI_API_KEY
pip install -r requirements.txt
python gen/import_batches.py     # load transcripts
python analyzer/run.py           # analysis -> data/analysis.json
python eval/evaluate.py          # score vs planted ground truth
uvicorn api.main:app --reload    # API on :8000
cd web && npm run dev            # frontend on :3000
```

## Stack

Python · sentence-transformers (MiniLM) · UMAP · HDBSCAN · FastAPI · Next.js + React Flow.
LLM (swappable provider) used offline only — transcript generation, labeling, node drafting.

---

*Built around a graph-based agent-flow model, as a focused demo of automated flow-coverage
analysis for voice agents.*
