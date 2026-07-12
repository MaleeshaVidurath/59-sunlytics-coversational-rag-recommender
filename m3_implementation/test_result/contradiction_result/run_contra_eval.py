# m3_implementation/test_result/contradiction_result/run_contra_eval.py
#
# Step 3 of the contradiction detector evaluation — detection accuracy vs
# baselines (see EVALUATION_PLAN.md, section 2, Experiment A2/A3).
#
# Runs the detectors over labeled_test_set.jsonl (label = clean | contradiction
# | hard_negative) and reports Precision / Recall / F1 / Balanced Accuracy, plus
# recall by corruption type AND by turn distance — the signature cross-turn axis.
#
# SYSTEMS (positive class = contradiction):
#   ours          — full detector: reconstructed session graph + Groq claim
#                   extraction + values_contradict() + DeBERTa NLI confirmation
#                   (replicated offline from the production module's own
#                    functions — no MongoDB, identical decision logic)
#   string_only   — ablation/baseline: ours WITHOUT the NLI gate
#                   (values_contradict() decides alone)
#   history_nli   — DECODE/SummaC-style UNSTRUCTURED baseline: DeBERTa NLI over
#                   (every serialized session fact, every response sentence);
#                   flag if any pair is contradiction (softmax P(contra) > 0.5)
#   uttr_pair_nli — Nie et al. 2021 STRUCTURED baseline: NLI only on
#                   (fact about product X, response sentence mentioning X) pairs
#   llm_judge     — "just ask an LLM" baseline: Groq judges whether the response
#                   contradicts the established session facts
#
# The clean + hard_negative cases are the negative class (label != contradiction)
# and drive the false-alarm / precision numbers.
#
# Reuses compute_metrics / CIs / table printer from the hallucination eval
# (imported, not duplicated) so both chapters score identically.
#
# Run:  python test_result/contradiction_result/run_contra_eval.py
#         [--skip-llm] [--skip-ours] [--limit N] [--test-set PATH] [--out PATH]

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

import networkx as nx

# Reuse the hallucination eval's statistics + table printer verbatim.
from test_result.hallucination_result.run_detector_eval import (
    compute_metrics, print_metrics_table,
)

_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_SET  = os.path.join(_DIR, "labeled_test_set.jsonl")
RESULTS   = os.path.join(_DIR, "results_contra_eval.json")

NAIVE_CONTRA_PROB = 0.5   # softmax threshold for the NLI baselines

# Attributes we serialize into session facts / hypotheses.
_FACT_FIELDS = ("colour", "price", "product_type", "name")


# ── Fact serialization (shared by the NLI + LLM baselines) ────────────────────

def _node_facts(aid: str, node: dict) -> list[dict]:
    """Serializes a product's established values into fact sentences."""
    name = node.get("name") or f"product {aid}"
    facts = []
    if node.get("colour"):
        facts.append({"aid": aid, "field": "colour",
                      "text": f"The {name} is {node['colour']} in colour."})
    if node.get("price"):
        facts.append({"aid": aid, "field": "price",
                      "text": f"The {name} costs {node['price']}."})
    if node.get("product_type"):
        facts.append({"aid": aid, "field": "product_type",
                      "text": f"The {name} is a {node['product_type']}."})
    facts.append({"aid": aid, "field": "name",
                  "text": f"One product is called {name}."})
    return facts


def _session_facts(case: dict) -> list[dict]:
    """All established facts for this turn: prior graph nodes PLUS current
    evidence (the union is what the response should stay consistent with)."""
    merged: dict[str, dict] = {}
    for aid, node in (case.get("graph_before", {}) or {}).items():
        merged[str(aid)] = dict(node)
    for ref in case.get("product_refs", []):
        aid = str(ref.get("article_id", ""))
        if aid:
            merged.setdefault(aid, {}).update(
                {k: ref.get(k) for k in _FACT_FIELDS if ref.get(k)}
            )
    facts = []
    for aid, node in merged.items():
        facts.extend(_node_facts(aid, node))
    return facts


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


# ── Detector 1 + ablation: ours (offline replication) ────────────────────────

def _build_graph(graph_before: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for aid, attrs in (graph_before or {}).items():
        g.add_node(str(aid), **{k: v for k, v in attrs.items() if v is not None})
    return g


# Claim extraction for the eval. Faithful to production _extract_claims_groq —
# SAME prompt (_GROQ_EXTRACT_PROMPT), SAME product-list format, SAME JSON parsing
# — the ONLY difference is the model id is configurable. Production uses
# llama-3.1-8b-instant, but its 6000 TPM free-tier ceiling cannot sustain a
# 599-case sequential run (empty extractions crush recall as a rate-limit
# artifact, not a detection failure). GROQ_EXTRACT_MODEL defaults to Llama 4
# Scout (30000 TPM, separate daily bucket) so the extraction coverage — and thus
# the measured recall — reflects the detector, not the API quota. Documented in
# RESULTS.md as a deliberate eval-only deviation.
_EXTRACT_MODEL = os.getenv("GROQ_EXTRACT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
_SHOW_FIELDS = ("colour", "price", "product_type", "pattern",
                "index_group", "section", "garment_group")


async def _extract_claims_model(response_text: str, refs: list, model: str) -> dict:
    """Faithful copy of _extract_claims_groq with a configurable model id."""
    import re as _re
    import httpx
    from memory.core.contradiction_detector import _GROQ_EXTRACT_PROMPT

    lines = []
    for ref in refs:
        attrs = ", ".join(f"{f}={ref[f]}" for f in _SHOW_FIELDS if ref.get(f))
        lines.append(f"- {ref['article_id']}: {ref['name']}\n  [{attrs}]")
    prompt = _GROQ_EXTRACT_PROMPT.format(
        product_list="\n".join(lines), response_text=response_text[:1500])
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 300, "temperature": 0.0},
            )
        if resp.status_code == 429:
            return {}
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = _re.sub(r'^```[a-z]*\n?', '', content)
            content = _re.sub(r'\n?```$', '', content).strip()
        extracted = json.loads(content)
        if not isinstance(extracted, dict):
            return {}
        return {str(k): v for k, v in extracted.items()}
    except Exception:
        return {}


async def _extract_with_retry(response_text: str, refs: list) -> dict:
    """Model-configurable extraction with single-call pacing + one gentle retry.

    With Scout's 30000 TPM (~41 calls/min sustainable at ~720 tok/call), a ~2s
    cadence is comfortably safe, so retries rarely fire. One call, and at most
    one retry after a short wait as a backstop for transient 429s."""
    base_delay = float(os.getenv("OURS_EVAL_DELAY", "2.0"))
    ext = await _extract_claims_model(response_text, refs, _EXTRACT_MODEL)
    if not ext:
        await asyncio.sleep(base_delay + 3.0)
        ext = await _extract_claims_model(response_text, refs, _EXTRACT_MODEL)
    await asyncio.sleep(base_delay)
    return ext


async def eval_ours_dual(
    cases: list[dict],
) -> tuple[list[bool], list[bool], list[dict]]:
    """Replicates ContradictionDetector's decision using the production module's
    own functions. Runs the Groq extraction ONCE per case and derives BOTH the
    full detector (string gate + NLI confirmation) and the string_only ablation
    (string gate alone) from it. Also applies _fix_response_text on every
    confirmed contradiction so the correction (ON/OFF) experiment can be scored
    OFFLINE from the same single Groq pass — keeping the whole evaluation within
    one day's token budget.

    Returns (preds_ours_nli, preds_string_only, case_details) where each
    case_details entry is {case_id, ours_pred, string_pred, corrected_text}."""
    from memory.core.contradiction_detector import (
        _update_graph_nodes, values_contradict,
        _confirm_with_nli, _fix_response_text, _CHECKABLE_FIELDS,
    )
    preds_nli: list[bool] = []
    preds_str: list[bool] = []
    details:   list[dict] = []
    t0 = time.time()
    for i, case in enumerate(cases):
        refs = case.get("product_refs", []) or []
        graph = _build_graph(case.get("graph_before", {}))
        _update_graph_nodes(graph, refs, case.get("turn_id", "t"),
                            case.get("session_id", "s"))
        extracted = await _extract_with_retry(case["response_text"], refs)

        str_found = False           # string gate alone flagged a mismatch
        nli_found = False           # string gate + NLI confirmation
        corrected = case["response_text"]
        for aid, fields in (extracted or {}).items():
            if not graph.has_node(aid):
                continue
            node = graph.nodes[aid]
            for attr in _CHECKABLE_FIELDS:
                ex = fields.get(attr, "")
                if not ex or not values_contradict(node.get(attr, ""), ex):
                    continue
                str_found = True
                with contextlib.redirect_stdout(io.StringIO()):
                    is_c, _ = _confirm_with_nli(node, ex, attr)
                if is_c:
                    nli_found = True
                    corrected = _fix_response_text(corrected, ex, node.get(attr, ""))
        preds_nli.append(nli_found)
        preds_str.append(str_found)
        details.append({
            "case_id":        case.get("case_id"),
            "label":          case.get("label"),
            "ours_pred":      nli_found,
            "string_pred":    str_found,
            "corrected_text": corrected,
        })
        if (i + 1) % 25 == 0:
            print(f"  [ours+string] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds_nli, preds_str, details


# ── Detector 3: history NLI (unstructured DECODE/SummaC-style) ────────────────

def _softmax(row):
    import numpy as np
    e = np.exp(np.array(row) - np.max(row))
    return e / e.sum()


def eval_history_nli(cases: list[dict], structured: bool,
                     tag: str) -> list[bool]:
    """structured=False → all (fact, response-sentence) pairs (unstructured).
    structured=True  → only pairs whose fact product name appears in the
                       response sentence (Nie et al. utterance-pair style)."""
    from text_rag.core.hallucination_checker import _get_nli_model
    model = _get_nli_model()

    preds = []
    t0 = time.time()
    for i, case in enumerate(cases):
        facts = _session_facts(case)
        sents = _split_sentences(case["response_text"])
        if not facts or not sents:
            preds.append(False)
            continue

        pairs = []
        for f in facts:
            fname = None
            # crude product-name token for the structured pairing
            m = re.search(r"The (.+?) (?:is|costs)", f["text"])
            if m:
                fname = m.group(1).lower()
            for s in sents:
                if structured and fname and fname not in s.lower():
                    continue
                pairs.append((f["text"], s))
        if not pairs:
            preds.append(False)
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            scores = model.predict(pairs)
        flagged = any(
            (_softmax(r)[0] > NAIVE_CONTRA_PROB and _softmax(r)[0] > _softmax(r)[2])
            for r in scores
        )
        preds.append(flagged)
        if (i + 1) % 25 == 0:
            print(f"  [{tag}] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds


# ── Detector 5: LLM judge (Groq) ─────────────────────────────────────────────

_JUDGE_PROMPT = """You are a strict consistency judge for a fashion \
recommendation assistant in a multi-turn conversation.

ESTABLISHED FACTS about the products (ground truth from earlier in the session \
and the product database):
{facts}

CURRENT ASSISTANT RESPONSE to check:
{response}

Does the response state anything that CONTRADICTS an established fact — a wrong \
colour, wrong price, wrong product name, or wrong product type for a product? \
Extra detail merely absent from the facts is NOT a contradiction.

Answer with ONLY this JSON, nothing else:
{{"contradiction": true or false}}"""


def eval_llm_judge(cases: list[dict]) -> list[bool | None]:
    import httpx
    from text_rag.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

    if not GROQ_API_KEY:
        print("  [llm_judge] GROQ_API_KEY not set — skipping baseline")
        return [None] * len(cases)

    preds: list[bool | None] = []
    t0 = time.time()
    with httpx.Client(timeout=30) as client:
        for i, case in enumerate(cases):
            facts = "\n".join(f"- {f['text']}" for f in _session_facts(case))
            prompt = _JUDGE_PROMPT.format(facts=facts, response=case["response_text"])
            verdict = None
            for attempt in range(4):
                try:
                    r = client.post(
                        f"{GROQ_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={"model": GROQ_MODEL,
                              "messages": [{"role": "user", "content": prompt}],
                              "temperature": 0, "max_tokens": 30},
                    )
                    if r.status_code == 429:
                        time.sleep(min(2 ** attempt * 2, 20))
                        continue
                    r.raise_for_status()
                    m = r.json()["choices"][0]["message"]["content"].lower()
                    if '"contradiction": true' in m or "'contradiction': true" in m:
                        verdict = True
                    elif '"contradiction": false' in m or "'contradiction': false" in m:
                        verdict = False
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"  [llm_judge] case {case['case_id']} failed: {e}")
                    time.sleep(2)
            preds.append(verdict)
            time.sleep(0.4)  # stay under Groq free-tier rate limits
            if (i + 1) % 25 == 0:
                print(f"  [llm_judge] {i+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
    return preds


# ── Breakdown metrics ────────────────────────────────────────────────────────

def recall_by_corruption(cases, preds) -> dict:
    buckets: dict[str, list[bool]] = {}
    for case, pred in zip(cases, preds):
        if case["label"] != "contradiction":
            continue
        buckets.setdefault(case["corruption"]["type"], []).append(pred)
    return {
        t: {"detected": sum(v), "total": len(v),
            "recall": round(sum(v) / len(v), 4)}
        for t, v in sorted(buckets.items())
    }


def recall_by_distance(cases, preds) -> dict:
    """The signature cross-turn breakdown: recall as a function of how many
    turns back the contradicted product's ground truth was established."""
    buckets: dict[str, list[bool]] = {}
    for case, pred in zip(cases, preds):
        if case["label"] != "contradiction":
            continue
        d = case.get("turn_distance", 0) or 0
        key = "3+" if d >= 3 else str(d)
        buckets.setdefault(key, []).append(pred)
    order = {"0": 0, "1": 1, "2": 2, "3+": 3}
    return {
        k: {"detected": sum(v), "total": len(v),
            "recall": round(sum(v) / len(v), 4)}
        for k, v in sorted(buckets.items(), key=lambda kv: order.get(kv[0], 9))
    }


def false_alarms(cases, preds) -> dict:
    """Negative-class errors, split by clean vs hard_negative."""
    out = {"clean": {"fp": 0, "total": 0}, "hard_negative": {"fp": 0, "total": 0}}
    for case, pred in zip(cases, preds):
        if case["label"] == "contradiction":
            continue
        bucket = out["hard_negative"] if case["label"] == "hard_negative" else out["clean"]
        bucket["total"] += 1
        if pred:
            bucket["fp"] += 1
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def _stratified_sample(cases: list[dict], n: int, seed: int = 123) -> list[dict]:
    """Stratified subsample preserving (label, corruption_type, distance-bucket)
    proportions — so the expensive Groq/NLI systems are compared on a
    representative slice of the full set."""
    import random as _random
    rng = _random.Random(seed)

    def stratum(c):
        if c["label"] != "contradiction":
            return ("neg", c["label"])
        d = c.get("turn_distance", 0) or 0
        db = "3+" if d >= 3 else str(d)
        return ("con", c["corruption"]["type"], db)

    buckets: dict = {}
    for c in cases:
        buckets.setdefault(stratum(c), []).append(c)

    frac = n / len(cases)
    sample = []
    for _, group in buckets.items():
        k = max(1, round(len(group) * frac))
        sample.extend(rng.sample(group, min(k, len(group))))
    rng.shuffle(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-nli", action="store_true",
                    help="skip the DeBERTa NLI baselines (history + uttr-pair)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified sample of N cases (seed 123) for the "
                         "Groq/NLI-heavy comparison")
    ap.add_argument("--test-set", default=TEST_SET)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    with open(args.test_set, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if args.sample and args.sample < len(cases):
        cases = _stratified_sample(cases, args.sample)
        print(f"Stratified sample: {len(cases)} cases (seed 123)")

    y_true = [c["label"] == "contradiction" for c in cases]
    n_pos = sum(y_true)
    print(f"Loaded {len(cases)} cases "
          f"({n_pos} contradiction / {len(cases)-n_pos} negative "
          f"[clean+hard_negative])\n")

    results: dict = {"n_cases": len(cases), "n_contradiction": n_pos}
    named: dict[str, dict] = {}

    def record(name, preds):
        named[name] = compute_metrics(y_true, preds)
        results[name] = {
            "metrics": named[name],
            "recall_by_corruption": recall_by_corruption(cases, preds),
            "recall_by_distance":   recall_by_distance(cases, preds),
            "false_alarms":         false_alarms(cases, preds),
        }

    if not args.skip_ours:
        print("[ours + string_only] shared Groq extraction, dual gate...")
        ours_preds, string_preds, ours_details = asyncio.run(eval_ours_dual(cases))
        record("ours", ours_preds)
        record("string_only", string_preds)
        # Sidecar: per-case corrected text + detection flags, so the correction
        # (ON/OFF) experiment is scored offline from this single Groq pass.
        detail_path = os.path.join(os.path.dirname(args.out), "ours_case_detail.jsonl")
        with open(detail_path, "w", encoding="utf-8") as f:
            for d in ours_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  wrote per-case detail → {os.path.basename(detail_path)}")

    if not args.skip_nli:
        print("[history_nli] unstructured DECODE/SummaC-style...")
        record("history_nli", eval_history_nli(cases, structured=False, tag="history_nli"))
        print("[uttr_pair_nli] structured Nie et al. style...")
        record("uttr_pair_nli", eval_history_nli(cases, structured=True, tag="uttr_pair_nli"))

    if not args.skip_llm:
        print("[llm_judge] Groq judge...")
        judge = eval_llm_judge(cases)
        answered = [(t, p) for t, p in zip(y_true, judge) if p is not None]
        if answered:
            jt, jp = zip(*answered)
            named["llm_judge"] = compute_metrics(list(jt), list(jp))
            kept_cases = [c for c, p in zip(cases, judge) if p is not None]
            kept_preds = [p for p in judge if p is not None]
            results["llm_judge"] = {
                "metrics": named["llm_judge"],
                "n_answered": len(answered),
                "recall_by_corruption": recall_by_corruption(kept_cases, kept_preds),
                "recall_by_distance":   recall_by_distance(kept_cases, kept_preds),
                "false_alarms":         false_alarms(kept_cases, kept_preds),
            }

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DETECTION ACCURACY (positive class = contradiction)")
    print("=" * 60)
    print_metrics_table(named)

    print("\nRECALL BY TURN DISTANCE (signature cross-turn result)")
    for system in named:
        if system not in results or "recall_by_distance" not in results[system]:
            continue
        row = "  ".join(
            f"d={k}:{v['recall']:.2f}({v['detected']}/{v['total']})"
            for k, v in results[system]["recall_by_distance"].items()
        )
        print(f"  {system:<14} {row}")

    print("\nRECALL BY CORRUPTION TYPE")
    for system in named:
        if system not in results or "recall_by_corruption" not in results[system]:
            continue
        print(f"  {system}:")
        for ctype, r in results[system]["recall_by_corruption"].items():
            print(f"    {ctype:<20} {r['detected']:>3}/{r['total']:<3} recall={r['recall']:.3f}")

    print("\nFALSE ALARMS (negative class)")
    for system in named:
        if system not in results or "false_alarms" not in results[system]:
            continue
        fa = results[system]["false_alarms"]
        print(f"  {system:<14} clean={fa['clean']['fp']}/{fa['clean']['total']}  "
              f"hard_neg={fa['hard_negative']['fp']}/{fa['hard_negative']['total']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
