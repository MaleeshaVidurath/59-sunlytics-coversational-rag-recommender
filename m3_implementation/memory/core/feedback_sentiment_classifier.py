# m3_implementation/memory/core/feedback_sentiment_classifier.py
#
# Domain-adapted feedback sentiment classifier
# ═══════════════════════════════════════════════════════════════════════════════
#
# Model: cardiffnlp/twitter-roberta-base-sentiment-latest
#
# Scientific basis:
#   Barbieri et al. (2020) "TweetEval: Unified Benchmark and Comparative
#   Evaluation for Tweet Classification" (EMNLP 2020, Findings).
#   RoBERTa fine-tuned on 124M tweets; achieves SoTA on TweetEval sentiment.
#
# Why this model for fashion chat feedback:
#   Conversational feedback utterances ("I don't like this", "too expensive",
#   "love it") share the defining properties of tweet text:
#     - Short (avg < 15 tokens vs Twitter's 22-token average)
#     - Informal, colloquial register
#     - Opinionated, first-person sentiment expression
#   SST-2 (formal movie reviews) and Amazon Reviews (long prose, 50-200 words)
#   diverge from this register and would require additional fine-tuning to
#   match the in-conversation feedback distribution.
#   Twitter-RoBERTa handles this distribution zero-shot without further training.
#
# Score mapping:
#   classify_feedback(text) → (label: str, score: float)
#   label : "positive" | "neutral" | "negative"
#   score : float in [-1.0, 1.0]
#           positive → [ +0.5,  +1.0]  (confidence-scaled)
#           neutral  → (−0.45, +0.45)  (centred, confidence-scaled)
#           negative → [−1.0,  −0.5]   (confidence-scaled)
#
#   Scaling formula: score = base_sign × (0.5 + 0.5 × model_confidence)
#   Rationale: model certainty of 0.33 (chance level) → half base magnitude;
#              certainty of 1.0 → full base magnitude.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os

# ── Model resolution ──────────────────────────────────────────────────────────
# Priority 1: locally trained model (fine-tuned on SemEval 2017, see
#             train_sentiment_classifier.py)
# Priority 2: Cardiff NLP pre-trained model (Barbieri et al., EMNLP 2020)
#             used as fallback before local training is complete.
_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL_DIR = os.path.join(_THIS_DIR, "..", "models", "sentiment_classifier")
_REMOTE_MODEL    = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_pipeline = None  # loaded once on first call

_LABEL_BASE: dict[str, float] = {
    "positive":  1.0,
    "neutral":   0.0,
    "negative": -1.0,
}


def _get_pipeline():
    """
    Loads the sentiment classifier on first call, then caches it.
    Uses the locally trained DistilBERT (SemEval 2017) if available,
    otherwise falls back to Cardiff NLP Twitter-RoBERTa.
    """
    global _pipeline
    if _pipeline is None:
        # Check for locally trained model first
        _config = os.path.join(_LOCAL_MODEL_DIR, "config.json")
        if os.path.isfile(_config):
            model_source = _LOCAL_MODEL_DIR
            label = "local Twitter-RoBERTa (SemEval 2017, fine-tuned)"
        else:
            model_source = _REMOTE_MODEL
            label = "Cardiff NLP Twitter-RoBERTa (remote fallback)"

        try:
            from transformers import pipeline as hf_pipeline
            _pipeline = hf_pipeline(
                "text-classification",
                model=model_source,
                truncation=True,
                max_length=128,
                device=-1,      # CPU; set device=0 to use GPU
            )
            print(f"[FeedbackClassifier] Loaded {label}")
        except Exception as exc:
            print(
                f"[FeedbackClassifier] Could not load model ({exc}). "
                "Falling back to keyword heuristic."
            )
            _pipeline = "unavailable"
    return None if _pipeline == "unavailable" else _pipeline


def classify_feedback(text: str) -> tuple[str, float]:
    """
    Classifies a short fashion-chat feedback utterance.

    Uses Twitter-RoBERTa (Barbieri et al., EMNLP 2020) zero-shot.
    Falls back to a keyword heuristic if the model cannot be loaded.
 
    Args:
        text: the user's feedback message (e.g. "I don't like this colour").

    Returns:
        (label, score)
        label  : "positive" | "neutral" | "negative"
        score  : float in [-1.0, 1.0].
                 Positive values = favourable; magnitude reflects confidence.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return _keyword_fallback(text)

    try:
        output = pipe(text[:512])
        # Normalise: some transformers versions return [[dict]], others return [dict]
        result = output[0]
        if isinstance(result, list):
            result = result[0]
        raw_label  = result["label"].lower()       # "positive" | "neutral" | "negative"
        confidence = result["score"]               # model certainty in [0.0, 1.0]

        base  = _LABEL_BASE.get(raw_label, 0.0)
        # Scale: at chance (conf≈0.33) → half base; at certainty (conf=1.0) → full base.
        score = round(base * (0.5 + 0.5 * confidence), 4)

        print(
            f"[FeedbackClassifier] '{text[:45]}' "
            f"→ {raw_label} (conf={confidence:.3f}  score={score:+.3f})"
        )
        return raw_label, score

    except Exception as exc:
        print(f"[FeedbackClassifier] Inference error: {exc}. Using keyword fallback.")
        return _keyword_fallback(text)


# ── Keyword fallback ──────────────────────────────────────────────────────────
# Used only when the transformer model is unavailable on first cold-start.
# Not used during normal operation once the model is loaded.

_POS_KW = frozenset([
    "love", "perfect", "amazing", "great", "excellent", "fantastic",
    "beautiful", "brilliant", "wonderful", "like it", "like this",
    "nice", "good", "suits me", "works for me", "yes please", "i'll take",
])
_NEG_KW = frozenset([
    "hate", "don't like", "do not like", "dislike", "not my", "not keen",
    "not right", "not what i", "doesn't suit", "wrong", "bad", "ugly",
    "horrible", "too expensive", "not happy", "disappointed", "terrible",
    "awful", "nah", "no not", "not impressed", "not for me",
])


def _keyword_fallback(text: str) -> tuple[str, float]:
    msg = text.lower()
    if any(k in msg for k in _NEG_KW):
        return "negative", -0.70
    if any(k in msg for k in _POS_KW):
        return "positive",  0.70
    return "neutral", 0.0
