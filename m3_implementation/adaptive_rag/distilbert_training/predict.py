# =============================================================================
# predict.py
# =============================================================================
# Run this after training to classify new, unseen conversation turns.
# This is the script your actual CRS pipeline will call at inference time.
#
# The Predictor class can be imported into your larger system like this:
#
#   from predict import Predictor
#   predictor = Predictor()   # loads the model once
#   result = predictor.predict(history, current_message)
#   if result["retrieval_strategy"] == "FULL":
#       # run full catalog search
#   elif result["retrieval_strategy"] == "PARTIAL":
#       # run metadata lookup
#   else:
#       # skip retrieval entirely
# =============================================================================

import os
import re
import torch
from dotenv import load_dotenv

# Load .env from the project root (3 levels above this file's directory)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
)
from config import MODEL_SAVE_DIR, LABEL_NAMES, RETRIEVAL_STRATEGY_MAP, MAX_LEN

_VALID_LABELS = frozenset(LABEL_NAMES)
_LABEL_TO_ID  = {name: i for i, name in enumerate(LABEL_NAMES)}
_NAME_TO_STRATEGY = {LABEL_NAMES[i]: s for i, s in RETRIEVAL_STRATEGY_MAP.items()}

# ── Groq LLM judge ────────────────────────────────────────────────────────────
# Secondary classifier that verifies DistilBERT when it's uncertain.
# Triggered when confidence < JUDGE_THRESHOLD or in the known failure mode
# (INITIAL_REQUEST predicted with conversation history present).
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_GROQ_BASE_URL   = "https://api.groq.com/openai/v1"

_JUDGE_SYSTEM_PROMPT = """You classify user messages in a fashion shopping assistant into one of 8 intents.

INITIAL_REQUEST  — fresh product search, different category from prior context, or no history.
                   e.g. "I want a coat", "show me jeans", "I need boots under £50"
REFINEMENT       — narrows or changes the SAME product type already being discussed.
                   e.g. "make it cheaper", "in red instead", "something smaller"
ATTRIBUTE_QUESTION — asks about a specific attribute of an already-shown product.
                   e.g. "what material is it?", "is it machine washable?", "what sizes?"
EXPLANATION_WHY  — asks why a product was recommended.
                   e.g. "why this one?", "why did you suggest this?"
COMPARISON       — compares two shown products.
                   e.g. "which is better quality?", "what's the difference?"
SELECTION_REFERENCE — requests more detail on one specific shown product.
                   e.g. "tell me more about the second one", "more on option 1"
FEEDBACK         — positive or negative reaction, no new product ask.
                   e.g. "I'll take it", "too expensive", "love it", "not for me"
CHITCHAT         — greeting or casual conversation.
                   e.g. "hello", "thanks", "ok"

Reply with ONLY the label name. Nothing else."""


class _GroqJudge:
    """Calls Groq to confirm or override an uncertain DistilBERT prediction."""

    def __init__(self) -> None:
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        self._client = OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL)

    def classify(self, history: list[dict], current_message: str) -> str | None:
        """Returns a validated label name, or None if the call fails."""
        if history:
            lines = [f"{t['role'].upper()}: {t['content'][:120]}" for t in history[-4:]]
            user_content = "Context:\n" + "\n".join(lines) + f'\n\nClassify: "{current_message}"'
        else:
            user_content = f'No prior conversation.\n\nClassify: "{current_message}"'
        try:
            resp = self._client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=15,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip().upper()
            if raw in _VALID_LABELS:
                return raw
            for label in _VALID_LABELS:
                if label in raw:
                    return label
            return None
        except Exception as exc:
            print(f"[GroqJudge] call failed: {exc}")
            return None


def clean_text(text: str) -> str:
    """
    Applies the same cleaning that was applied to the training data.
    Always apply this to any text before passing it to the model,
    so the model sees inputs in the same format it was trained on.
    """
    text = text.replace('**', '')
    text = text.replace(' — ', ', ')
    text = text.replace('—', ', ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_input_text(history: list[dict], current_message: str) -> str:
    """
    Formats the conversation history and current message into the same
    [SEP]-joined string format that was used during training.

    Args:
        history: list of dicts like [{"role": "user", "content": "..."},
                                      {"role": "bot",  "content": "..."}]
                 Pass the last 1-3 exchanges (up to 6 turns).
        current_message: the user's latest message to classify.

    Returns:
        A single string ready to feed to the tokenizer.
    """
    # Use only the last 6 turns (= last 3 exchanges) to keep within MAX_LEN
    recent = history[-6:] if len(history) > 6 else history

    parts = []
    for turn in recent:
        role    = turn["role"].upper()
        content = clean_text(turn["content"])
        parts.append(f"{role}: {content}")

    parts.append(f"CURRENT: {clean_text(current_message)}")
    return " [SEP] ".join(parts)


class Predictor:
    """
    Wraps the trained model for easy inference.

    Load it once at application startup, then call predict() for each turn.
    Loading the model takes a few seconds — you don't want to do it on
    every single conversation turn.
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = MODEL_SAVE_DIR
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        print(f"Loading trained model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded and ready.")

        # Judge is initialised lazily on first predict() call so that
        # GROQ_API_KEY is already in os.environ (loaded by other modules at startup)
        self._judge: _GroqJudge | None = None
        self._judge_ready: bool = False

    def predict(
        self,
        history: list[dict],
        current_message: str,
    ) -> dict:
        """
        Classifies one conversation turn.

        Args:
            history:         list of prior turns (role + content dicts)
            current_message: the user's latest message

        Returns a dict with:
            label_id           : integer 0-7
            label_name         : e.g. "ATTRIBUTE_QUESTION"
            retrieval_strategy : "FULL", "PARTIAL", or "NO"
            confidence         : probability of the predicted class (0.0-1.0)
            all_probabilities  : dict of label_name → probability for all 8 classes
        """
        # Format the input exactly as during training
        # ── Lazy judge init (first call only, after all dotenv is loaded) ────
        if not self._judge_ready:
            self._judge_ready = True
            try:
                self._judge = _GroqJudge()
                print("[GroqJudge] initialised on first request — verifying every prediction")
            except Exception as exc:
                print(f"[GroqJudge] disabled: {exc}")

        input_text = format_input_text(history, current_message)

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass — no gradient computation needed at inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits   # shape: [1, 8]

        # Convert logits to probabilities using softmax
        probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        label_id   = int(probs.argmax())
        confidence = float(probs[label_id])
        label_name = LABEL_NAMES[label_id]

        # ── Groq judge: always verify, Groq wins on disagreement ─────────────
        print(f"[DistilBERT] label={label_name}  conf={confidence:.1%}")
        if self._judge is not None:
            groq_label = self._judge.classify(history, current_message)
            if groq_label:
                if groq_label == label_name:
                    print(f"[GroqJudge]  label={groq_label}  -> AGREE")
                else:
                    print(f"[GroqJudge]  label={groq_label}  -> OVERRIDE (DistilBERT={label_name})")
                    label_name = groq_label
                    label_id   = _LABEL_TO_ID[label_name]
            else:
                print("[GroqJudge]  API call failed — keeping DistilBERT result")
        else:
            print("[GroqJudge]  DISABLED (check GROQ_API_KEY in .env and openai package)")
        print(f"[FINAL]      label={label_name}  strategy={_NAME_TO_STRATEGY[label_name]}")

        return {
            "label_id":           label_id,
            "label_name":         label_name,
            "retrieval_strategy": _NAME_TO_STRATEGY[label_name],
            "confidence":         round(confidence, 4),
            "all_probabilities": {
                LABEL_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(LABEL_NAMES))
            },
        }


# =============================================================================
# Quick demo — run:  python predict.py
# =============================================================================
if __name__ == "__main__":
    predictor = Predictor()

    # --- Demo conversation 1: INITIAL REQUEST (should return FULL retrieval) ---
    result = predictor.predict(
        history=[],
        current_message="I need a casual dress for a garden party",
    )
    print("\n--- Demo 1: No history, fresh request ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # --- Demo 2: ATTRIBUTE QUESTION (should return PARTIAL retrieval) ---
    history = [
        {"role": "user", "content": "Show me some black dresses"},
        {"role": "bot",  "content": "Here are two options. Option 1 is the Valerie dress (black, dress): Short dress in a crisp cotton weave. Option 2 is the Angel (dark pink, dress): Short A-line dress."},
    ]
    result = predictor.predict(
        history=history,
        current_message="What material is the first one made of?",
    )
    print("\n--- Demo 2: After recommendation, asking about material ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # --- Demo 3: FEEDBACK (should return NO retrieval) ---
    result = predictor.predict(
        history=history,
        current_message="I love it, I'll take the first one",
    )
    print("\n--- Demo 3: Positive feedback ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # --- Demo 4: REFINEMENT (should return FULL retrieval) ---
    result = predictor.predict(
        history=history,
        current_message="Actually can you show me something in white instead?",
    )
    print("\n--- Demo 4: Preference change ---")
    print(f"  Label     : {result['label_name']}")
    print(f"  Strategy  : {result['retrieval_strategy']}")
    print(f"  Confidence: {result['confidence']:.2%}")
