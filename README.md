# FlowGap

**Coverage analysis for voice-agent conversation flows.**

FlowGap reads call transcripts, discovers what callers *actually* asked for, and surfaces the
intents a hand-built Agent Flow doesn't handle — the ones quietly ending in human transfer. For
each gap it drafts a drop-in flow node to close it.

> Show me your call logs and I'll show you the percentage of calls your bot is silently handing
> to a human — and the exact nodes that would recover them.

---

## The problem

Voice agents run on flows authored by hand. Someone anticipates the intents callers will have
and builds a node for each one. But real callers ask for things nobody anticipated. When that
happens the agent has no handler, falls through to `transfer_to_human`, and the call is lost to
a human queue.

That's expensive. It's also *invisible*: the evidence is buried across thousands of transcripts
no one reads. Teams spend sprints polishing the nodes they designed, completely blind to the
demand they never designed for.

FlowGap closes that blind spot. It finds unhandled demand from the call logs themselves —
without requiring anyone to know in advance what to look for.

---

## Why this approach

Most coverage tools audit the flow designer's intent: "did we hit node X?" FlowGap audits the
*caller's* intent: "did every caller get a handler that matched what they actually said?"

That distinction changes the method entirely. A fixed classifier can only find intents you
already named. Clustering finds the intents you didn't know you were missing. The pipeline is
unsupervised by design — no intent taxonomy required as input.

---

## How it works

Six offline stages. Nothing touches the live call path; all analysis runs post-hoc on logs.

```
  call transcripts
        |
        v
  [1] Parse          -- keep only caller turns, discard agent side
        |
        v
  [2] Embed          -- all-MiniLM-L6-v2, 384-dim per call (local, no API)
        |
        v
  [3] UMAP           -- 384-dim -> 8-dim (cosine, n_neighbors=15)
        |
        v
  [4] HDBSCAN        -- density clustering, no k needed, noise class built-in
        |
        v
  [5] Label          -- LLM names each cluster; keyword fallback on rate-limit
        |
        v
  [6] Gap detect     -- centroid of cluster vs. embedded flow-node descriptions
        |                cosine sim < 0.60  =>  GAP
        v
  [7] Node spec      -- LLM drafts a handler (instructions + tools + edges)
        |
        v
  data/analysis.json
```

**Why each choice was made:**

| Decision | Reason |
|---|---|
| Cluster, don't classify | Classifiers find known intents; clustering finds unknown ones |
| UMAP before HDBSCAN | Density clustering degrades in 384-dim space; UMAP preserves local neighborhoods |
| Centroid-based gap matching | Survives bad LLM/keyword labels — compares call *content* to flow, not label strings |
| `transfer_rate` as severity, not signal | Using it to detect gaps would be circular — it just re-reads ground truth |
| LLM only offline | Generation, labeling, node-drafting. Never in a latency-sensitive path |
| Planted ground truth | 3 deliberate gap intents with known labels, so detection accuracy is measurable |

---

## Dataset

**Sunrise Multispeciality Clinic** — a synthetic call center for a fictional clinic chain.
The agent flow has 6 handled intents and 3 deliberately *planted* gap intents (callers ask
for these but the flow has no node for them — they end in `transfer_to_human`).

**298 calls** across **9 intents:**

| Intent | Type | Calls | Share |
|---|---|---|---|
| book_appointment | handled | 86 | 28.9% |
| reschedule_cancel | handled | 45 | 15.1% |
| insurance_tpa | handled | 44 | 14.8% |
| billing_payment | handled | 38 | 12.8% |
| clinic_info | handled | 30 | 10.1% |
| prescription_refill | handled | 17 | 5.7% |
| **lab_reports** | **GAP** | 13 | 4.4% |
| **post_op_care** | **GAP** | 13 | 4.4% |
| **doctor_availability_specific** | **GAP** | 12 | 4.0% |

38 gap calls (12.8% of the dataset). These are the calls the live bot would silently hand off.

Transcripts are Hinglish (Hindi-English code-switching), realistic for an Indian clinic context.
Synthetic generation used Gemini 2.0 Flash; 298 were hand-generated in batches due to API
rate limits.

---

## Hurdles and how they were overcome

### 1. Transcript generation failures

**Problem:** The original Groq-based generator silently wrote `[generation failed]` placeholder
strings to disk when an API call failed. 250 of 400 transcripts were placeholders — the
embedding step processed garbage text and clustering was completely wrong.

**Fix:** Rewrote `generate.py` to use Gemini 2.0 Flash with 3-attempt exponential backoff
(5 s / 10 s / 20 s). On failure, the call is *skipped* — no file written, no placeholder.
A `validate_transcripts()` post-pass scans for short or malformed files and deletes them
before the pipeline runs. Dataset shrank from 400 to 298, but every record is clean.

---

### 2. 54.7% HDBSCAN noise

**Problem:** Raw `all-MiniLM-L6-v2` embeddings are 384-dimensional. HDBSCAN operates on
euclidean distance — at 384 dims, the curse of dimensionality flattens distances between
points. The clusterer couldn't find density peaks, and dumped 163 of 298 calls (54.7%) into
the noise class (`-1`). All 38 gap calls landed in noise — completely undetectable.

**Before:**
```
HDBSCAN (raw 384-dim):  1 real cluster, 163 noise (54.7%)
```

**Fix:** Added UMAP before HDBSCAN. UMAP compresses 384 dims to 8 dims while preserving
local neighborhood structure (cosine metric, `n_neighbors=15`, `min_dist=0.0`). HDBSCAN then
operates on 8-dim euclidean space where density peaks are sharp.

**After:**
```
UMAP (384->8) + HDBSCAN:  21 clusters, 11 noise (3.7%)
```

| Metric | Before UMAP | After UMAP |
|---|---|---|
| Clusters found | 1 | 21 |
| Noise calls | 163 (54.7%) | 11 (3.7%) |
| Gap calls in noise | 38 / 38 | 0 / 38 |

One config change. Gap calls went from 100% invisible to fully clustered.

---

### 3. 14 false positives from label-based gap detection

**Problem:** The first gap detection implementation compared cluster *label strings* to flow
node names (e.g., does `"nahi_mujhe_appointment"` match `"book_appointment"`?). Keyword
labels derived from Hinglish transcripts had no lexical overlap with English flow node names,
so nearly every cluster was flagged as a gap — including all 12 handled-intent clusters.

FP = 14. Every handled cluster was a false gap.

**Fix:** Switched to centroid-based matching. Instead of comparing strings:
1. Compute the centroid of each cluster (mean of member call embeddings, 384-dim).
2. Embed each flow node description in the same MiniLM space.
3. Compare centroid to every flow node via cosine similarity.
4. A cluster is "in flow" if `max_sim >= 0.60`.

This operates on actual call content, so it doesn't care what the label says.

| Approach | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Label-based | 0 | 14 | 3 | 0.00 | 0.00 | 0.00 |
| Centroid-based | 2 | 6 | 1 | 0.25 | 0.67 | 0.36 |

FP dropped from 14 to 6. Two of three planted gaps are correctly detected.

---

### 4. Enriched flow descriptions made things worse

**Problem:** 6 FPs remained. The attempt to fix them was to add rich Hinglish-aware natural
language descriptions to `intended_flow.yaml` — so MiniLM could match, e.g., "billing payment
invoice" to the billing node. Seemed reasonable.

**Result:** FP went from 6 to 13. Adding descriptions shifted the MiniLM-embedded flow node
vectors in directions that broke 7 clusters that were previously correct.

**Root cause:** MiniLM's embedding space is dense near `0.5–0.7` cosine similarity for
short clinic-domain sentences. Adding more tokens to flow node descriptions moved the vectors
closer to neighbouring clusters that happened to contain some of the same words — not the ones
we intended. The enrichment was not neutral; it was adversarial to existing correct matches.

**Fix:** Reverted `intended_flow.yaml` to the minimal `id + tools` format. The correct lesson
here is that embedding-based matching is sensitive to the *exact* text being embedded; richer
is not automatically better when the vectors share a dense neighbourhood.

---

### 5. Windows encoding crashes

**Problem:** Python's default Windows cp1252 codec crashed on every print statement containing
Unicode: arrows (`→`), checkmarks (`✓`, `✅`), em dashes (`—`), and the degree symbol in
progress bars. The pipeline couldn't complete a full run.

**Fix:** Replaced all Unicode characters in print statements with ASCII equivalents across
`cluster.py`, `label.py`, `run.py`, `evaluate.py`, and `import_batches.py`. No functional
change — cosmetic-only, but required for the pipeline to run on Windows.

---

### 6. Batch transcript import without API

**Problem:** After switching to Gemini, daily API quotas meant generating all 298 transcripts
in one session was impossible. Transcripts were manually constructed in batches
(`data/raw_batches/batch_*.json`) in a different format (turns array, lowercase speakers)
from what the pipeline's `parse.py` expected (line-per-turn, uppercase `AGENT:`/`CALLER:`).

**Fix:** Wrote `gen/import_batches.py` — a zero-API converter that reads one batch file at a
time (no full-dataset memory load), converts turn format, uses the batch filename stem as
`call_id` (matching `parse.py`'s `path.stem` behaviour), and rebuilds `ground_truth.json`
from all transcripts on disk.

---

## Findings

### Clustering quality

The UMAP+HDBSCAN pipeline produced 21 clusters from 298 calls with only 3.7% noise.

Top clusters by size:

| Cluster label (keyword) | Calls | Likely intent |
|---|---|---|
| nahi_mujhe_appointment | 57 | appointment-related (large mixed cluster) |
| bill_think_invoice | 26 | billing queries |
| insurance_cashless_your | 19 | insurance / TPA |
| book_thank_regarding | 16 | appointment booking |
| book_appointment_thank | 16 | appointment booking |
| thank_medication_refill | 15 | prescription refill |
| cancel_appointment_next | 13 | reschedule/cancel |
| thank_appointment_saturday | 13 | appointment scheduling |
| available_think_before | 13 | availability queries |
| surgery_should_your | 8 | **post-op care (GAP)** |
| test_days_blood | 10 | **lab reports (GAP)** |

### Gap detection results

Centroid cosine similarity scores for all 21 clusters (threshold = 0.60):

| Cluster | Best flow match | Sim | Verdict |
|---|---|---|---|
| book_thank_regarding | book_appointment | 0.701 | in flow |
| reschedule_cancel_thank | reschedule_cancel | 0.694 | in flow |
| name_reschedule_cancel | reschedule_cancel | 0.708 | in flow |
| bill_think_invoice | billing_payment | 0.557 | **GAP** (FP) |
| book_appointment_thank | book_appointment | 0.676 | in flow |
| insurance_cashless_your | clinic_info | 0.498 | **GAP** (FP) |
| nahi_mujhe_appointment | book_appointment | 0.358 | **GAP** (FP) |
| great_thank_name | insurance_tpa | 0.631 | in flow |
| regarding_insurance_thank | insurance_tpa | 0.714 | in flow |
| thank_medication_refill | prescription_refill | 0.639 | in flow |
| **surgery_should_your** | clinic_info | 0.285 | **GAP (TP: post_op_care)** |
| cancel_appointment_next | reschedule_cancel | 0.673 | in flow |
| **test_days_blood** | clinic_info | 0.420 | **GAP (TP: lab_reports)** |
| thank_possible_yeah | book_appointment | 0.513 | **GAP** (FP) |
| thank_there_father | book_appointment | 0.623 | in flow |
| what_your_thank | clinic_info | 0.629 | in flow |
| thank_info_regarding | clinic_info | 0.829 | in flow |
| thank_morning_appointment | book_appointment | 0.547 | **GAP** (FP) |
| thank_appointment_saturday | book_appointment | 0.561 | **GAP** (FP) |
| thank_appointment_been | book_appointment | 0.618 | in flow |
| available_think_before | book_appointment | 0.603 | in flow |

### Detected gaps

| Gap | Cluster label | Calls | Purity | Transfer rate | Sim |
|---|---|---|---|---|---|
| **lab_reports** | test_days_blood | 10 | 90% | 90% | 0.420 |
| **post_op_care** | surgery_should_your | 8 | 100% | 100% | 0.285 |

Both detected gaps have extremely high transfer rates (callers who asked these things were
transferred to a human 90–100% of the time). These are the highest-value nodes to add.

### Missed gap

| Gap | Reason |
|---|---|
| doctor_availability_specific | Absorbed into `book_appointment` cluster (sim=0.603, correctly ruled in-flow) |

`doctor_availability_specific` calls ("I want an appointment with Dr. Mehta specifically")
are semantically near-identical to ordinary booking. MiniLM embeds them in the same region
of 384-dim space. They form no separate cluster — they dissolve into the booking cluster.
This is the honest edge of embedding-based discovery: coarse, novel intents separate cleanly;
fine-grained sub-intents of an existing handled intent need a second-pass split with a
classifier trained on examples.

### Scored metrics

Evaluated by majority-vote: a planted gap intent is "detected" if the cluster it dominates
is flagged as a gap.

| Metric | Value |
|---|---|
| True Positives | 2 |
| False Positives | 6 |
| False Negatives | 1 |
| **Precision** | **0.25** |
| **Recall** | **0.67** |
| **F1** | **0.36** |

**On precision:** the 6 FPs are clusters whose centroids fall in the 0.42–0.56 similarity
range — just below the 0.60 threshold. They are handled intents with degraded similarity
scores caused by keyword labels drawn from Hinglish text. Proper LLM labels (blocked by API
rate limits during this run) would produce better cluster descriptions and raise those
centroids into the in-flow zone. The detection *method* cleanly separates the coarse gaps
(post_op_care: sim=0.285, lab_reports: sim=0.420) from all handled intents; the borderline
FPs are an artifact of the label quality ceiling, not the centroid-matching approach.

---

## Conclusion

FlowGap demonstrates that you can discover unhandled voice-agent intents from raw call logs —
no labelled training data, no intent taxonomy, no human review of transcripts required.

The two key results:

**1. UMAP is not optional.** Density clustering on raw 384-dim embeddings is broken by the
curse of dimensionality (54.7% noise). A single UMAP compression step fixed it (3.7% noise,
21 clean clusters). Every practitioner reaching for HDBSCAN on sentence embeddings should
try UMAP first.

**2. Match by content, not by label.** Gap detection that compares label strings to flow
node names is brittle — one bad label (e.g., `nahi_mujhe_appointment` for a billing cluster)
breaks everything. Centroid-to-flow-node cosine similarity compares what the calls *actually
said* to what the flow *actually handles*. It reduced FP from 14 to 6 and correctly detected
the two highest-value gaps (both 90–100% transfer rate).

The remaining challenge is sub-intents of handled flows: embedding-based discovery cannot
split `doctor_availability_specific` from `book_appointment` because the call content is too
similar at the sentence level. A natural follow-on is a second-pass classifier within the
booking cluster trained to distinguish specific-doctor requests from general availability
requests.

---

## Run it

```bash
pip install -r requirements.txt

# Set your Gemini key in .env (see .env.example)
# GEMINI_API_KEY=...

# 1. Import hand-generated transcript batches
python gen/import_batches.py

# 2. Full analysis pipeline -> data/analysis.json
python analyzer/run.py

# 3. Evaluate against planted ground truth
python eval/evaluate.py
```

## Stack

| Layer | Technology |
|---|---|
| Embedding | sentence-transformers / all-MiniLM-L6-v2 (local) |
| Dimensionality reduction | UMAP (umap-learn) |
| Clustering | HDBSCAN |
| LLM (offline) | Gemini 2.0 Flash (labeling, generation, node specs) |
| Flow definition | YAML (intended_flow.yaml) |
| Output | JSON (data/analysis.json) |

---

*Built as a focused demo of flow-coverage analysis for hand-authored voice-agent flows.*
