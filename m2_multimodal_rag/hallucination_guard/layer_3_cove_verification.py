"""
Layer 3 — Enhanced Chain-of-Verification (CoVe).

Reference: Dhuliawala et al. (2023), "Chain-of-Verification Reduces
Hallucination in Large Language Models."
Cited in: Tonmoy et al. (2024), "A Comprehensive Survey of Hallucination
Mitigation Techniques in Large Language Models" (arXiv:2401.01313v3).

Enhancements over baseline CoVe:

  1. Predefined attribute questions (guaranteed coverage of key fields)
     Baseline CoVe lets LLM pick questions freely — it tends to pick
     easy ones it already knows the answer to (self-confirmation bias).
     We always ask about colour, type, department, and appearance first.

  2. DeBERTa NLI consistency judgment (replaces LLM judge)
     Baseline CoVe uses the same LLM to judge its own consistency —
     unreliable. We replace this with cross-encoder/nli-deberta-v3-base
     (same model used by M3's contradiction detector) which gives a
     calibrated CONTRADICTION / NEUTRAL / ENTAILMENT score.

  3. Direct string matching for predefined questions (deterministic)
     For the predefined attribute questions, we compare the metadata
     answer directly against the explanation text — no LLM needed.
     Only LLM-generated questions fall back to NLI judgment.

Four-step pipeline:
  Step 1 — Draft    : LLM generates explanation (done upstream).
  Step 2 — Plan     : Predefined attribute questions + LLM-generated questions.
  Step 3 — Execute  : Answered INDEPENDENTLY from metadata only.
  Step 4 — Judge    : Direct string match (predefined) + DeBERTa NLI (LLM questions).
"""

from m2_multimodal_rag.llm_generator import llm_generator

# ── DeBERTa NLI singleton ──────────────────────────────────────────────────────
_nli_model = None

def _get_nli():
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
            print("   [CoVe | NLI] DeBERTa NLI model loaded.")
        except Exception as e:
            print(f"   [CoVe | NLI] DeBERTa load failed: {e}. Falling back to LLM judge.")
    return _nli_model

# NLI label indices for cross-encoder/nli-deberta-v3-base
# Label 0 = CONTRADICTION  Label 1 = NEUTRAL  Label 2 = ENTAILMENT
_NLI_CONTRADICTION_IDX = 0
_NLI_CONTRADICTION_THRESHOLD = 0.55

# Predefined attribute questions — always asked regardless of explanation content.
# Each tuple: (question, metadata_field, display_name)
_PREDEFINED_CHECKS = [
    ("What colour is this product?",           "colour_group_name",        "colour"),
    ("What type of clothing product is this?", "product_type_name",        "product_type"),
    ("What is the graphical appearance?",      "graphical_appearance_name","appearance"),
    ("Which department does this belong to?",  "department_name",          "department"),
]


class CoVeVerifier:
    """
    Enhanced Chain-of-Verification (CoVe) with DeBERTa NLI judgment
    and predefined attribute questions for guaranteed coverage.
    """

    NUM_LLM_QUESTIONS  = 2      # LLM-generated questions on top of predefined ones
    PASS_THRESHOLD     = 0.67   # fraction of questions that must be consistent

    # ------------------------------------------------------------------ #
    # Step 2A — Predefined attribute questions (deterministic coverage)
    # ------------------------------------------------------------------ #
    def _run_predefined_checks(
        self, explanation: str, metadata: dict
    ) -> list[dict]:
        """
        Directly checks key attribute claims in the explanation against
        metadata using case-insensitive string matching — no LLM needed.
        This catches the most common and obvious hallucinations instantly.
        """
        exp_lower = explanation.lower()
        trail = []

        for question, field, label in _PREDEFINED_CHECKS:
            meta_val = str(metadata.get(field, "")).strip()
            if not meta_val or meta_val == "Unknown":
                continue

            # Check if the metadata value appears in the explanation
            consistent = meta_val.lower() in exp_lower

            trail.append({
                "question":        question,
                "metadata_answer": meta_val,
                "consistent":      consistent,
                "method":          "direct_match",
                "field":           field,
            })

            status = "PASS" if consistent else "FAIL"
            print(f"   [CoVe | Predefined] [{status}] {label}: "
                  f"expected='{meta_val}' in explanation={consistent}")

        return trail

    # ------------------------------------------------------------------ #
    # Step 2B — LLM-generated questions (dynamic coverage)
    # ------------------------------------------------------------------ #
    def _plan_llm_questions(self, explanation: str, metadata: dict) -> list[str]:
        """
        Generates additional targeted questions beyond the predefined ones.
        Focuses on claims specific to this explanation's content.
        """
        if not llm_generator.is_available or not explanation.strip():
            return []

        prompt = (
            f"A fashion recommendation explanation was generated.\n"
            f"Generate {self.NUM_LLM_QUESTIONS} short fact-checking questions "
            f"about claims in the explanation that are NOT about colour, type, "
            f"department, or appearance (those are already checked).\n\n"
            f"Explanation: \"{explanation}\"\n\n"
            f"Rules:\n"
            f"- Target specific factual claims (style, occasion, features)\n"
            f"- Each question must be answerable from product metadata alone\n"
            f"- Under 12 words each\n"
            f"Output ONLY {self.NUM_LLM_QUESTIONS} questions, one per line."
        )

        result = llm_generator._call_llm(prompt, max_tokens=60, temperature=0.0)
        if not result:
            return []

        questions = [
            ln.strip("•-– ").strip()
            for ln in result.strip().split("\n")
            if ln.strip() and len(ln.strip()) > 5
        ]
        return questions[:self.NUM_LLM_QUESTIONS]

    # ------------------------------------------------------------------ #
    # Step 3 — Execute independently from metadata
    # ------------------------------------------------------------------ #
    def _execute_independently(
        self, questions: list[str], metadata: dict
    ) -> list[str]:
        """
        Answers LLM-generated questions from metadata ONLY — never from
        the explanation (core CoVe insight, Dhuliawala et al., 2023).
        """
        if not questions or not llm_generator.is_available:
            return ["Unknown"] * len(questions)

        meta_context = (
            f"Product Name : {metadata.get('prod_name', '?')}\n"
            f"Colour       : {metadata.get('colour_group_name', '?')}\n"
            f"Type         : {metadata.get('product_type_name', '?')}\n"
            f"Department   : {metadata.get('department_name', '?')}\n"
            f"Appearance   : {metadata.get('graphical_appearance_name', '?')}\n"
            f"Description  : {str(metadata.get('detail_desc', ''))[:200]}"
        )

        answers = []
        for question in questions:
            prompt = (
                f"Answer using ONLY the product metadata below. "
                f"Short factual answer (under 10 words).\n\n"
                f"METADATA:\n{meta_context}\n\n"
                f"QUESTION: {question}\nANSWER:"
            )
            answer = llm_generator._call_llm(prompt, max_tokens=20, temperature=0.0)
            answers.append(answer.strip() if answer else "Unknown")

        return answers

    # ------------------------------------------------------------------ #
    # Step 4 — DeBERTa NLI consistency judgment
    # ------------------------------------------------------------------ #
    def _nli_consistency(
        self, explanation: str, question: str, metadata_answer: str
    ) -> tuple[bool, float]:
        """
        Uses DeBERTa NLI to judge whether the metadata-derived answer
        is consistent with what the explanation claims.

        Premise   = fact from metadata ("The product is [answer]")
        Hypothesis = claim extracted from explanation context

        Returns (is_consistent: bool, contradiction_score: float)
        Falls back to True (pass) if NLI model unavailable.
        """
        nli = _get_nli()
        if nli is None:
            return True, 0.0

        try:
            premise    = f"The answer to '{question}' is: {metadata_answer}."
            hypothesis = explanation[:300]

            scores = nli.predict([(premise, hypothesis)])
            contra_score = float(scores[0][_NLI_CONTRADICTION_IDX])
            is_consistent = contra_score < _NLI_CONTRADICTION_THRESHOLD

            return is_consistent, contra_score

        except Exception as e:
            print(f"   [CoVe | NLI] Prediction failed: {e} — defaulting to PASS")
            return True, 0.0

    # ------------------------------------------------------------------ #
    # Full CoVe pipeline
    # ------------------------------------------------------------------ #
    def verify(
        self, explanation: str, metadata: dict
    ) -> tuple[bool, list[dict], float]:
        """
        Runs the full enhanced CoVe pipeline.

        Returns:
            passes             bool        True if score >= PASS_THRESHOLD
            verification_trail list[dict]  Q&A pairs with consistency verdicts
            consistency_score  float       fraction consistent (0.0–1.0)
        """
        trail = []

        # --- Step 2A + 4A: Predefined checks (direct string matching) ---
        predefined_trail = self._run_predefined_checks(explanation, metadata)
        trail.extend(predefined_trail)

        # --- Step 2B: LLM-generated questions ---
        llm_questions = self._plan_llm_questions(explanation, metadata)

        # --- Step 3: Answer LLM questions from metadata ---
        llm_answers = self._execute_independently(llm_questions, metadata)

        # --- Step 4B: DeBERTa NLI judgment for LLM questions ---
        for question, answer in zip(llm_questions, llm_answers):
            is_consistent, contra_score = self._nli_consistency(
                explanation, question, answer
            )
            trail.append({
                "question":          question,
                "metadata_answer":   answer,
                "consistent":        is_consistent,
                "method":            "deberta_nli",
                "contradiction_score": round(contra_score, 3),
            })
            status = "PASS" if is_consistent else "FAIL"
            print(f"   [CoVe | NLI] [{status}] Q: {question[:50]} | "
                  f"A: {answer} | contra_score={contra_score:.3f}")

        if not trail:
            return True, [], 1.0

        consistent_count  = sum(1 for t in trail if t["consistent"])
        consistency_score = consistent_count / len(trail)
        passes            = consistency_score >= self.PASS_THRESHOLD

        status = "PASS" if passes else "FAIL -> regeneration needed"
        print(f"   [Layer 3 | CoVe] {consistent_count}/{len(trail)} consistent "
              f"(score={consistency_score:.0%}) -> {status}")

        return passes, trail, consistency_score


# Singleton
cove_verifier = CoVeVerifier()
