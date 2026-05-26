"""
Chain-of-Verification (CoVe) — Post-Generation Hallucination Verifier.

Reference: Dhuliawala et al. (2023), "Chain-of-Verification Reduces Hallucination
in Large Language Models."
Cited in: Tonmoy et al. (2024), "A Comprehensive Survey of Hallucination Mitigation
Techniques in Large Language Models" (arXiv:2401.01313v3), Section 2.1.2.

Four-step CoVe pipeline applied to fashion recommendation explanations:
  Step 1 — Draft    : LLM generates an explanation (done upstream in llm_generator)
  Step 2 — Plan     : LLM plans targeted verification questions about its own claims
  Step 3 — Execute  : Questions are answered INDEPENDENTLY using only verified metadata
                      (not the explanation), so answers cannot be contaminated by
                      hallucinated content in the draft.
  Step 4 — Final    : Inconsistencies are flagged; the verifier passes only if
                      consistency_score >= PASS_THRESHOLD.

This is Layer 3 of the M2 hallucination guard stack:
  Layer 1 — EMNLP 2023 Knowledge-Grounded Self-Reflection  (llm_generator.py)
  Layer 2 — VLM Visual Verification                         (blip_verification.py)
  Layer 3 — Chain-of-Verification / CoVe                    (this file)
"""

from m2_multimodal_rag.llm_generator import llm_generator


class CoVeVerifier:
    """
    Implements the Chain-of-Verification (CoVe) protocol for fashion explanations.

    Questions are generated about the draft explanation, then answered independently
    from verified product metadata — never from the explanation itself — so the
    verification is unbiased and immune to hallucinated content in the draft.
    """

    NUM_QUESTIONS = 3
    PASS_THRESHOLD = 0.67   # ≥2/3 questions must be consistent to pass

    # ------------------------------------------------------------------
    # Step 2 — Plan
    # ------------------------------------------------------------------
    def plan_questions(self, explanation: str, metadata: dict) -> list[str]:
        """
        Generates targeted fact-checking questions about the claims in the explanation.
        Questions must be answerable from product metadata alone.
        """
        if not llm_generator.is_available or not explanation.strip():
            return []

        prompt = (
            f"A fashion recommendation explanation was generated for a product.\n"
            f"Generate exactly {self.NUM_QUESTIONS} short, specific questions "
            f"that fact-check the key claims made in the explanation.\n\n"
            f"Explanation: \"{explanation}\"\n\n"
            f"Rules:\n"
            f"- Each question must target a concrete factual claim (colour, type, style)\n"
            f"- Each question must be answerable from product metadata alone\n"
            f"- Keep each question under 12 words\n"
            f"Output ONLY the {self.NUM_QUESTIONS} questions, one per line. "
            f"No numbering. No extra text."
        )

        result = llm_generator._call_llm(prompt, max_tokens=80, temperature=0.0)
        if not result:
            return []

        questions = [ln.strip("•-– ").strip() for ln in result.strip().split("\n") if ln.strip()]
        return [q for q in questions if len(q) > 5][: self.NUM_QUESTIONS]

    # ------------------------------------------------------------------
    # Step 3 — Execute independently
    # ------------------------------------------------------------------
    def execute_independently(self, questions: list[str], metadata: dict) -> list[str]:
        """
        Answers each verification question using ONLY the verified product metadata,
        not the explanation — ensuring answers are unbiased (core CoVe insight).
        """
        if not questions or not llm_generator.is_available:
            return []

        meta_context = (
            f"Product Name : {metadata.get('prod_name', '?')}\n"
            f"Colour       : {metadata.get('colour_group_name', '?')}\n"
            f"Type         : {metadata.get('product_type_name', '?')}\n"
            f"Department   : {metadata.get('department_name', '?')}\n"
            f"Category     : {metadata.get('product_group_name', '?')}\n"
            f"Appearance   : {metadata.get('graphical_appearance_name', '?')}\n"
            f"Description  : {str(metadata.get('detail_desc', ''))[:250]}"
        )

        answers = []
        for question in questions:
            prompt = (
                f"Answer the question below using ONLY the product metadata provided. "
                f"Give a short, factual answer (under 10 words).\n\n"
                f"PRODUCT METADATA:\n{meta_context}\n\n"
                f"QUESTION: {question}\n"
                f"ANSWER:"
            )
            answer = llm_generator._call_llm(prompt, max_tokens=20, temperature=0.0)
            answers.append(answer.strip() if answer else "Unknown")

        return answers

    # ------------------------------------------------------------------
    # Step 4 — Final consistency check
    # ------------------------------------------------------------------
    def check_consistency(
        self, explanation: str, questions: list[str], answers: list[str]
    ) -> list[dict]:
        """
        Checks whether the metadata-derived answers are consistent with what
        the explanation claimed. Inconsistencies reveal hallucinated content.
        """
        if not questions or not answers or not llm_generator.is_available:
            return [{"question": q, "metadata_answer": a, "consistent": True}
                    for q, a in zip(questions, answers)]

        qa_block = "\n".join(
            f"Q{i+1}: {q}\nA{i+1} (from metadata): {a}"
            for i, (q, a) in enumerate(zip(questions, answers))
        )

        prompt = (
            f"Check whether each metadata-verified answer is CONSISTENT with "
            f"the fashion explanation below.\n\n"
            f"Explanation: \"{explanation}\"\n\n"
            f"Verified Q&A pairs:\n{qa_block}\n\n"
            f"For each pair output: CONSISTENT or INCONSISTENT\n"
            f"One verdict per line (Q1, Q2, Q3 order). Nothing else."
        )

        result = llm_generator._call_llm(prompt, max_tokens=40, temperature=0.0)
        if not result:
            return [{"question": q, "metadata_answer": a, "consistent": True}
                    for q, a in zip(questions, answers)]

        verdicts = [ln.strip().upper() for ln in result.strip().split("\n") if ln.strip()]

        trail = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            verdict = verdicts[i] if i < len(verdicts) else "CONSISTENT"
            trail.append({
                "question": q,
                "metadata_answer": a,
                "consistent": "INCONSISTENT" not in verdict,
            })

        return trail

    # ------------------------------------------------------------------
    # Full CoVe pipeline
    # ------------------------------------------------------------------
    def verify(self, explanation: str, metadata: dict) -> tuple[bool, list[dict], float]:
        """
        Runs the full CoVe pipeline (Steps 2-4) on a single explanation.

        Returns:
            passes            bool         True if consistency_score >= PASS_THRESHOLD
            verification_trail list[dict]  Q&A pairs with consistency verdicts
            consistency_score float        fraction of consistent Q&A pairs (0.0–1.0)
        """
        questions = self.plan_questions(explanation, metadata)
        if not questions:
            return True, [], 1.0

        answers = self.execute_independently(questions, metadata)
        trail = self.check_consistency(explanation, questions, answers)

        if not trail:
            return True, [], 1.0

        consistent_count = sum(1 for t in trail if t["consistent"])
        consistency_score = consistent_count / len(trail)
        passes = consistency_score >= self.PASS_THRESHOLD

        status = "PASS" if passes else "FAIL → regeneration needed"
        print(
            f"   [CoVe] {consistent_count}/{len(trail)} questions consistent "
            f"(score={consistency_score:.0%}) → {status}"
        )
        for t in trail:
            mark = "PASS" if t["consistent"] else "FAIL"
            print(f"   [CoVe]   [{mark}] Q: {t['question']} | A: {t['metadata_answer']}")

        return passes, trail, consistency_score


# Singleton
cove_verifier = CoVeVerifier()
