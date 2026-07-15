"""
ensemble_label.py
─────────────────
Reads real_data_from_mongodb.csv, sends every row where confidence < 0.90
to 3 free LLMs, and writes the majority-vote label into a new `majority_vote`
column in the same CSV file.

Voters (add the matching key to .env to activate):
  GEMINI_API_KEY   → Google Gemini 2.0 Flash
  MISTRAL_API_KEY  → Mistral AI  (mistral-small-latest)
  Ollama (local)   → no key needed; requires `ollama serve` running

The script is safe to resume — rows that already have a majority_vote are skipped.

Usage:
    python ensemble_label.py
    python ensemble_label.py --input data/real_data_from_mongodb.csv
    python ensemble_label.py --threshold 0.90
    python ensemble_label.py --ollama-model mistral
"""

import os
import csv
import time
import argparse
from collections import Counter
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

LABEL_NAMES = [
    "INITIAL_REQUEST",
    "REFINEMENT",
    "ATTRIBUTE_QUESTION",
    "EXPLANATION_WHY",
    "COMPARISON",
    "SELECTION_REFERENCE",
    "FEEDBACK",
    "CHITCHAT",
]
VALID_LABELS = frozenset(LABEL_NAMES)

_SYSTEM_PROMPT = """You classify user messages in a fashion shopping assistant into one of 8 intents.

INITIAL_REQUEST     — fresh product search, different category from prior context, or no history.
                      e.g. "I want a coat", "show me jeans", "I need boots under £50"
REFINEMENT          — narrows or changes the SAME product type already being discussed.
                      e.g. "make it cheaper", "in red instead", "something smaller"
ATTRIBUTE_QUESTION  — asks about a specific attribute of an already-shown product.
                      e.g. "what material is it?", "is it machine washable?", "what sizes?"
EXPLANATION_WHY     — asks why a product was recommended.
                      e.g. "why this one?", "why did you suggest this?"
COMPARISON          — compares two shown products.
                      e.g. "which is better quality?", "what's the difference?"
SELECTION_REFERENCE — requests more detail on one specific shown product.
                      e.g. "tell me more about the second one", "more on option 1"
FEEDBACK            — positive or negative reaction, no new product ask.
                      e.g. "I'll take it", "too expensive", "love it", "not for me"
CHITCHAT            — greeting or casual conversation.
                      e.g. "hello", "thanks", "ok"

Reply with ONLY the label name. Nothing else."""


# ── Label parsing ─────────────────────────────────────────────────────────────

def _parse_label(raw: str) -> str | None:
    """Extract a valid label name from an LLM response string."""
    cleaned = raw.strip().upper()
    if cleaned in VALID_LABELS:
        return cleaned
    for label in VALID_LABELS:
        if label in cleaned:
            return label
    return None


# ── Voter classes (one per LLM) ───────────────────────────────────────────────

class GeminiVoter:
    name = "Gemini"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def classify(self, input_text: str) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": input_text},
                ],
                max_tokens=15,
                temperature=0.0,
            )
            return _parse_label(resp.choices[0].message.content)
        except Exception as exc:
            print(f"    [Gemini] error: {exc}")
            return None


class MistralVoter:
    name = "Mistral"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in .env")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
        )

    def classify(self, input_text: str) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": input_text},
                ],
                max_tokens=15,
                temperature=0.0,
            )
            return _parse_label(resp.choices[0].message.content)
        except Exception as exc:
            print(f"    [Mistral] error: {exc}")
            return None


class OllamaVoter:
    name = "Ollama"

    def __init__(self, model: str = "llama3.2"):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def classify(self, input_text: str) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": input_text},
                ],
                max_tokens=15,
                temperature=0.0,
            )
            return _parse_label(resp.choices[0].message.content)
        except Exception as exc:
            print(f"    [Ollama] error: {exc}")
            return None


# ── Voter loading ─────────────────────────────────────────────────────────────

def _load_voters(ollama_model: str) -> list:
    """Initialise each voter. Voters with missing API keys are skipped."""
    voters = []
    candidates = [
        (GeminiVoter,  {}),
        (MistralVoter, {}),
        (OllamaVoter,  {"model": ollama_model}),
    ]
    for cls, kwargs in candidates:
        try:
            voters.append(cls(**kwargs))
            print(f"  [ok]      {cls.name}")
        except Exception as exc:
            print(f"  [skipped] {cls.name}: {exc}")
    return voters


# ── Voting logic ──────────────────────────────────────────────────────────────

def _majority_vote(votes: list) -> str | None:
    """
    Return the label with the most votes.
    Requires at least 2 voters to agree when 3 are available.
    Returns None if all voters disagreed (no majority).
    """
    if not votes:
        return None
    counts = Counter(votes)
    top_label, top_count = counts.most_common(1)[0]
    if len(votes) >= 3 and top_count < 2:
        return None
    return top_label


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _should_process(row: dict, threshold: float) -> bool:
    """True if the row needs ensemble labelling (low confidence, not yet done)."""
    conf = row.get("confidence", "")
    if conf == "" or conf is None:
        return False
    try:
        return float(conf) < threshold and not row.get("majority_vote", "")
    except ValueError:
        return False


def _save_csv(path: str, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ensemble-label low-confidence CSV rows")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "data", "real_data_from_mongodb.csv"),
        help="CSV file to update (default: data/real_data_from_mongodb.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Rows with confidence below this value are labelled (default: 0.90)",
    )
    parser.add_argument(
        "--ollama-model", default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls per voter (default: 0.5)",
    )
    parser.add_argument(
        "--save-every", type=int, default=25,
        help="Save CSV checkpoint every N processed rows (default: 25)",
    )
    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "majority_vote" not in fieldnames:
        fieldnames.append("majority_vote")
        for row in rows:
            row["majority_vote"] = ""

    # ── Init voters ───────────────────────────────────────────────────────────
    print("Initialising voters...")
    voters = _load_voters(args.ollama_model)
    if not voters:
        print("\nNo voters available. Add at least one API key to .env and retry.")
        return
    print()

    # ── Rows to process ───────────────────────────────────────────────────────
    pending = [(i, row) for i, row in enumerate(rows) if _should_process(row, args.threshold)]
    already_done = sum(1 for r in rows if r.get("majority_vote"))

    print(f"Total rows        : {len(rows)}")
    print(f"Already labelled  : {already_done}")
    print(f"To process now    : {len(pending)}  (confidence < {args.threshold})")
    print(f"Active voters     : {[v.name for v in voters]}")
    print()

    if not pending:
        print("Nothing to process. All low-confidence rows already have a majority_vote.")
        return

    # ── Ensemble labelling ────────────────────────────────────────────────────
    no_majority_count = 0

    for done_count, (row_idx, row) in enumerate(pending, start=1):
        input_text  = row.get("input_text", "")
        preview_msg = row.get("current_message", input_text)[:45]

        votes        = []
        vote_display = []

        for voter in voters:
            label = voter.classify(input_text)
            if label:
                votes.append(label)
                vote_display.append(f"{voter.name}={label}")
            else:
                vote_display.append(f"{voter.name}=FAIL")
            time.sleep(args.delay)

        result = _majority_vote(votes)
        rows[row_idx]["majority_vote"] = result or ""

        outcome = result if result else "NO_MAJORITY"
        print(f"  [{done_count:>4}/{len(pending)}]  {preview_msg:<45}  "
              f"{' | '.join(vote_display)}  =>  {outcome}")

        if not result:
            no_majority_count += 1

        if done_count % args.save_every == 0:
            _save_csv(args.input, rows, fieldnames)
            print(f"  [checkpoint saved at {done_count}]")

    # ── Final save ────────────────────────────────────────────────────────────
    _save_csv(args.input, rows, fieldnames)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_labelled = sum(1 for r in rows if r.get("majority_vote"))
    got_majority   = len(pending) - no_majority_count

    print()
    print("Done.")
    print(f"  Processed this run : {len(pending)}")
    print(f"  Got majority vote  : {got_majority}")
    print(f"  No majority        : {no_majority_count}  (all voters disagreed)")
    print(f"  Total labelled     : {total_labelled} / {len(rows)} rows")
    print()

    vote_counts = Counter(r["majority_vote"] for r in rows if r.get("majority_vote"))
    print("Majority vote distribution:")
    for label in LABEL_NAMES:
        count = vote_counts.get(label, 0)
        print(f"  {label:<25} {count:>4}")

    print(f"\nUpdated file: {args.input}")


if __name__ == "__main__":
    main()
