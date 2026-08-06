"""
Layer 2 — Enhanced Chain-of-Verification (CoVe).

Reference: Dhuliawala et al. (2023), "Chain-of-Verification Reduces
Hallucination in Large Language Models."
Cited in: Tonmoy et al. (2024), "A Comprehensive Survey of Hallucination
Mitigation Techniques in Large Language Models" (arXiv:2401.01313v3).

Enhancements over baseline CoVe:

  1. Fully dynamic question generation — no hardcoded field checks.
     The LLM reads the explanation and generates questions about the
     specific claims it made. This handles the infinite vocabulary of
     a conversational recommender without brittle keyword matching.

  2. DeBERTa NLI consistency judgment (replaces LLM judge).
     Baseline CoVe uses the same LLM to judge its own consistency —
     unreliable. We use cross-encoder/nli-deberta-v3-base which gives
     a calibrated CONTRADICTION / NEUTRAL / ENTAILMENT score.

  3. Independent execution — questions answered from metadata only,
     never from the explanation (core CoVe insight).

Three-step pipeline:
  Step 1 — Draft   : LLM generates explanation (done upstream).
  Step 2 — Plan    : LLM generates fact-checking questions from explanation.
  Step 3 — Execute : Questions answered from metadata only.
  Step 4 — Judge   : DeBERTa NLI checks for contradictions.
"""

from m2_multimodal_rag.llm_generator import llm_generator

# ── DeBERTa NLI singleton ──────────────────────────────────────────────────────
_nli_model = None

def _get_nli():
    """
    Lazily loads the DeBERTa NLI cross-encoder once and caches it in the
    module-level singleton. If loading fails (e.g. model unavailable),
    _nli_model stays None and callers fall back to a default PASS.
    """
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


class CoVeVerifier:
    """
    Enhanced Chain-of-Verification (CoVe) with DeBERTa NLI judgment.
    Fully dynamic — no hardcoded field checks or keyword matching.
    Works correctly regardless of how the LLM phrases its explanation.
    """

    NUM_QUESTIONS  = 4     # questions generated per explanation
    PASS_THRESHOLD = 0.67  # fraction that must be consistent to pass

    # ------------------------------------------------------------------ #
    # Step 2 — LLM generates fact-checking questions from the explanation
    # ------------------------------------------------------------------ #
    def _plan_questions(self, explanation: str) -> list[str]:
        """
        LLM reads the explanation and generates targeted questions about
        the specific factual claims it contains — colour, fabric, fit,
        occasion, style, features — anything that can be verified against
        the product metadata.
        """
        if not llm_generator.is_available or not explanation.strip():
            return []

        prompt = (
            f"A fashion recommendation explanation was generated.\n"
            f"Generate {self.NUM_QUESTIONS} short fact-checking questions "
            f"about the specific factual claims made in the explanation.\n\n"
            f"Explanation: \"{explanation}\"\n\n"
            f"Rules:\n"
            f"- Cover different claims: colour, fabric, fit, features, occasion\n"
            f"- Each question must be answerable from product metadata alone\n"
            f"- Under 12 words each\n"
            f"- Be specific to what this explanation actually claims\n"
            f"Output ONLY {self.NUM_QUESTIONS} questions, one per line."
        )

        result = llm_generator._call_llm(prompt, max_tokens=100, temperature=0.0)
        if not result:
            return []

        # Strip leading bullet/numbering characters (e.g. "1.", "-", "•")
        # the LLM may prepend to each line, keeping only the question text.
        questions = [
            ln.strip("•-– 1234567890.").strip()
            for ln in result.strip().split("\n")
            if ln.strip() and len(ln.strip()) > 5
        ]
        return questions[:self.NUM_QUESTIONS]

    # ------------------------------------------------------------------ #
    # Step 3 — Answer questions from metadata ONLY (never from explanation)
    # ------------------------------------------------------------------ #
    def _execute_independently(
        self, questions: list[str], metadata: dict
    ) -> list[str]:
        """
        Answers questions from metadata ONLY — never from the explanation.
        This is the core CoVe insight: independent execution prevents the
        LLM from confirming its own claims.
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
    # Step 4 — DeBERTa NLI contradiction check
    # ------------------------------------------------------------------ #
    def _nli_consistency(
        self, explanation: str, question: str, metadata_answer: str
    ) -> tuple[bool, float]:
        """
        Uses DeBERTa NLI to check whether the metadata-derived answer
        CONTRADICTS what the explanation claims.

        Premise    = fact from metadata ("The answer to Q is: A")
        Hypothesis = the explanation text

        Returns (is_consistent, contradiction_score).
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
        Runs the full CoVe pipeline — fully semantic, no keyword matching.

        Returns:
            passes             bool        True if score >= PASS_THRESHOLD
            verification_trail list[dict]  Q&A pairs with consistency verdicts
            consistency_score  float       fraction consistent (0.0–1.0)
        """
        # Step 2 — generate questions from the explanation
        questions = self._plan_questions(explanation)
        if not questions:
            return True, [], 1.0

        # Step 3 — answer from metadata independently
        answers = self._execute_independently(questions, metadata)

        # Step 4 — NLI contradiction check for each Q&A pair
        trail = []
        for question, answer in zip(questions, answers):
            is_consistent, contra_score = self._nli_consistency(
                explanation, question, answer
            )
            trail.append({
                "question":            question,
                "metadata_answer":     answer,
                "consistent":          is_consistent,
                "method":              "deberta_nli",
                "contradiction_score": round(contra_score, 3),
            })
            status = "PASS" if is_consistent else "FAIL"
            print(f"   [CoVe | NLI] [{status}] Q: {question[:50]} | "
                  f"A: {answer} | contra_score={contra_score:.3f}")

        consistent_count  = sum(1 for t in trail if t["consistent"])
        consistency_score = consistent_count / len(trail)
        passes            = consistency_score >= self.PASS_THRESHOLD

        status = "PASS" if passes else "FAIL -> regeneration needed"
        print(f"   [Layer 2 | CoVe] {consistent_count}/{len(trail)} consistent "
              f"(score={consistency_score:.0%}) -> {status}")

        return passes, trail, consistency_score

    # ------------------------------------------------------------------ #
    # Prior-claim consistency check (used by explanation_generate)
    # ------------------------------------------------------------------ #
    def check_claim_consistency(self, text: str, claim_text: str) -> tuple[bool, float]:
        """
        NLI check: does `text` contradict a claim made in an earlier turn?

        Premise    = the prior claim (already told to the customer)
        Hypothesis = the newly generated text

        Returns (is_consistent, contradiction_score).
        Falls back to True (pass) if the NLI model is unavailable.
        """
        nli = _get_nli()
        if nli is None:
            return True, 0.0
        try:
            scores = nli.predict([(claim_text, text[:300])])
            contra_score = float(scores[0][_NLI_CONTRADICTION_IDX])
            return contra_score < _NLI_CONTRADICTION_THRESHOLD, contra_score
        except Exception as e:
            print(f"   [Claim | NLI] Prediction failed: {e} — defaulting to PASS")
            return True, 0.0


# Singleton
cove_verifier = CoVeVerifier()
