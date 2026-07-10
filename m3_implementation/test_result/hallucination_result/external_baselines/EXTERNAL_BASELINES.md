# Off-the-Shelf Baselines — Unmodified External Detectors on Our Test Set

Generated: 2026-07-10 · Raw data: `results_external_baselines.json` (includes
per-case raw scores) · Figures: `figures/fig9`, `figures/fig10` ·
Log: `external_run.log`

---

## 1. Why this experiment exists

The main evaluation (`../RESULTS.md`) compares against **re-implemented**
established methods (SummaC-style naive NLI, RAGAS-style LLM judge). To
remove any doubt that the re-implementations were weak stand-ins, this
experiment additionally runs **unmodified, released, citable tools** on the
same 238-case test set. The viva answer becomes: *"we compare against the
actual released detectors: Vectara HHEM-2.1, SummaC-Conv, and
LettuceDetect."*

## 1b. Why these three tools — selection criteria and rationale

Tools were selected against four criteria:

1. **Citable** — published paper or widely recognised industry tool.
2. **Same task** — built for "given context + generated text, detect
   ungrounded content" (not generic QA accuracy).
3. **Runnable unmodified, locally, free** — so the comparison is against the
   real released artifact, not a re-interpretation of it.
4. **Coverage of the method space** — each represents a different school of
   thought, closing off the objection "you picked weak baselines".

| Tool | School it represents | Why it matters here |
|---|---|---|
| **SummaC-Conv** | the academic classic — canonical NLI-based inconsistency detection (TACL 2022) | Same method family as our checker's NLI core, and the method our re-implemented naive-NLI baseline was modeled on. Running the *official* release proves the re-implementation was not a weakened strawman (official: F1 0.643 — same region as expected). |
| **Vectara HHEM-2.1** | the industry standard — the model behind the public LLM hallucination leaderboard | The most widely recognised production-grade consistency model; shows what general summarization-consistency training buys on this task (F1 0.509 — least). |
| **LettuceDetect** | the modern RAG-specialist — 2025 detector trained on RAGTruth | Embodies "use an existing RAG hallucination benchmark": it is literally a detector trained on the best available one. Strongest of the three (F1 0.688), still 0.29 F1 below the domain-aware checker — even RAG-specialised training does not cover catalog-grounded association errors. |

**Considered and rejected:** Lynx (Patronus) — 8B/70B judge LLM, needs a
large GPU; AlignScore — same NLI family as SummaC, adds little; the RAGAS
library — its faithfulness metric is an LLM judge, already covered by the
main evaluation's Groq-judge baseline; FactCC — a 2020 classifier requiring
in-domain retraining, so no fair "as released" comparison exists.

**Why three different schools matter:** all three fail the same way on
cross-item swaps (5–53% detection). Three independent tools sharing one
blind spot turns the lock-map argument from a claim into a replicated
phenomenon: presence-checking approaches are structurally blind to
association errors.

## 1c. The process, in plain terms

1. **Install safely.** The project venv's libraries were pinned via a pip
   constraints file so nothing could be up/downgraded. summac's stale PyPI
   pin (`transformers==4.8.1`) was bypassed with `--no-deps`; three small
   repairs followed (nltk tokenizer data, official conv weights from GitHub,
   a 3-line tokenizer-kwarg shim in *our* script, not in the libraries).
   Each fix was verified with a smoke test before proceeding.
2. **Serialize the exam.** External tools read plain text, so each case's
   structured evidence was rendered as fixed-format sentences
   ("Item 1: London dress, type Dress, colour Black, price £11.08. ...").
   The 238 cases and their labels are identical to the main evaluation.
3. **Run unmodified.** Each tool judged every case at its conventional 0.5
   threshold; raw scores stored. Runtime on CPU: HHEM ~4 min, SummaC
   ~100 min, LettuceDetect ~3 min. No failures.
4. **Score identically.** The same `compute_metrics` /
   `recall_by_corruption` functions as the main evaluation (imported, not
   duplicated) produce the tables below.

## 2. Tools and setup

| Tool | What it is | How used |
|---|---|---|
| **Vectara HHEM-2.1-open** | the hallucination-detection model behind Vectara's public LLM hallucination leaderboard | HuggingFace, `trust_remote_code`; consistency score < 0.5 → hallucinated |
| **SummaC-Conv** (Laban et al., TACL 2022) | NLI-based inconsistency detector, official package + released conv weights | score < 0.5 → hallucinated |
| **LettuceDetect** (KRLabs, 2025) | RAG hallucination detector trained on RAGTruth (ModernBERT) | any predicted hallucination span → hallucinated |

The tools consume free-text contexts, so each case's structured evidence is
serialized to plain text (`serialize_evidence()`:
`"Item 1: London dress, type Dress, colour Black, price £11.08. ..."`).
The tools themselves run exactly as released, at their conventional 0.5
thresholds; raw scores are stored for threshold analysis.

Environment notes (documented for reproducibility): `summac` was installed
`--no-deps` (its PyPI pin `transformers==4.8.1` is stale) plus a 3-line
runtime shim dropping the legacy `truncation_strategy` tokenizer kwarg; its
trained conv weights (`summac_conv_vitc_sent_perc_e.bin`, 1.8 KB) come from
the official GitHub repo (not bundled on PyPI); nltk `punkt/punkt_tab` data
required.

## 3. Results (same 238 cases: 205 corrupted + 33 clean)

| System | Precision | Recall | F1 | Balanced acc. |
|---|---|---|---|---|
| **Our checker (v3)** | **1.000** | **0.951** | **0.975** | **0.976** |
| LettuceDetect | 0.957 | 0.537 | 0.688 | 0.693 |
| SummaC-Conv | 0.911 | 0.498 | 0.643 | 0.597 |
| Vectara HHEM-2.1 | 0.923 | 0.351 | 0.509 | 0.585 |

Recall by corruption type:

| Corruption | Ours | HHEM | SummaC | LettuceDetect |
|---|---|---|---|---|
| colour_swap | 0.896 | 0.354 | 0.542 | 0.542 |
| price_change | 0.982 | 0.286 | 0.464 | 0.750 |
| name_swap | 0.948 | 0.603 | 0.466 | 0.690 |
| **cross_item_swap** | **0.977** | **0.093** | **0.535** | **0.047** |

## 4. Findings

1. **All three released tools miss roughly half or more of the planted
   lies** (recall 0.35–0.54) despite decent precision. General-purpose
   consistency models are weakly sensitive to exact-value mismatches
   (a wrong price or product name in otherwise fluent, on-topic text).
2. **The cross-item swap result is the headline** (fig10): HHEM detects
   9.3% and LettuceDetect 4.7% of value swaps between items — because every
   swapped value *is present* in the context; only the item association is
   wrong. Presence-checking approaches are structurally blind to
   association errors. The item→sentence lock map detects 97.7% —
   direct empirical evidence for the architectural novelty.
3. **Ranking sanity check**: LettuceDetect (RAG-trained) > SummaC > HHEM
   (summarization-trained) — the closer a tool's training distribution is
   to grounded-response checking, the better it does, yet none approaches
   the domain-aware checker.

## 5. Fairness caveats (state in the write-up)

- The external tools were built for other tasks/domains (summarization
  consistency, RAG QA); this comparison shows they do not transfer to
  catalog-grounded recommendation **as released** — not that they are bad
  at their own tasks.
- Their 0.5 default thresholds were not tuned on our data (neither was our
  checker's threshold tuned on the test set — see the threshold sweep in
  the main evaluation). Raw scores are stored in
  `results_external_baselines.json` should threshold curves be needed.
- Evidence serialization involves choices (field order, wording); the
  serialization is fixed, committed, and identical for all tools.

## 6. Reproduction

```powershell
# one-time setup
venv\Scripts\python.exe -m pip install --no-deps summac
venv\Scripts\python.exe -m pip install lettucedetect nltk sentencepiece
venv\Scripts\python.exe -m nltk.downloader punkt punkt_tab

# run (CPU: HHEM ~4 min, SummaC ~100 min, LettuceDetect ~3 min)
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\external_baselines\run_external_baselines.py

# figures
venv\Scripts\python.exe m3_implementation\test_result\hallucination_result\external_baselines\make_external_figures.py
```
