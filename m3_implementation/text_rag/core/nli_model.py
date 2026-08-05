# m3_implementation/text_rag/core/nli_model.py
#
# Single shared DeBERTa-v3 NLI cross-encoder for the whole M3 stack.
#
# WHY THIS EXISTS:
#   Both the turn-local faithfulness check (HallucinationChecker) and the
#   cross-turn consistency check (ContradictionDetector) need the same
#   entailment model. Before this module each kept its own module-level
#   singleton, so `cross-encoder/nli-deberta-v3-base` was loaded into memory
#   twice (~1.4 GB of duplicated weights) and warmed up twice at first request.
#
# NLI LABEL ORDER for cross-encoder/nli-deberta-v3-base:
#   index 0 = CONTRADICTION   index 1 = NEUTRAL   index 2 = ENTAILMENT

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from text_rag.config import NLI_MODEL_NAME

_nli_model = None


def _softmax(values) -> list:
    """Numerically stable softmax over the three NLI logits."""
    shifted = [float(v) - max(float(x) for x in values) for v in values]
    exps    = [math.exp(v) for v in shifted]
    total   = sum(exps) or 1.0
    return [e / total for e in exps]


def get_nli_model():
    """Returns the process-wide NLI cross-encoder, loading it on first use."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
        print(f"[NLI] Shared model loaded: {NLI_MODEL_NAME}")
    return _nli_model


def nli_scores(premise: str, hypothesis: str) -> dict:
    """
    Scores one (premise, hypothesis) pair as PROBABILITIES summing to 1.

    CrossEncoder.predict returns raw logits, roughly in the range -5..+7. A
    caller comparing those against a "0.5 probability" threshold would be
    thresholding on a scale where 0.5 carries no meaning — nearly every pair
    clears it, and the decision would rest entirely on the incidental
    contradiction-vs-entailment comparison. Softmaxing here makes the threshold
    mean what it says.

    Note this helper is used by the cross-turn consistency layer only.
    HallucinationChecker deliberately keeps calling the model directly on
    logits with its own calibrated threshold, so its behaviour — and the
    evaluation numbers derived from it — are untouched by this.

    Returns {"contradiction": float, "neutral": float, "entailment": float},
    or all zeros on failure, which callers read as "no signal".
    """
    try:
        logits = get_nli_model().predict([(premise, hypothesis)])[0]
        contra, neutral, entail = _softmax(logits)
        return {"contradiction": contra, "neutral": neutral, "entailment": entail}
    except Exception as e:
        print(f"[NLI] scoring failed: {e}")
        return {"contradiction": 0.0, "neutral": 0.0, "entailment": 0.0}
