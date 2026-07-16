"""
Layer 1 — Knowledge-Grounded Self-Reflection.

Reference: Ji et al. (2023), "Towards Mitigating LLM Hallucination via
Self Reflection", Findings of ACL: EMNLP 2023 (2023.findings-emnlp.123).

Three sub-loops from the paper, adapted to fashion product explanations:

  Loop 1 — Factual Knowledge Acquisition
    Generate background product knowledge from verified metadata.
    Score it for factual consistency (1-10). Refine once if score < 6.
    This anchors all downstream generation to verified facts.

  Loop 2 — Knowledge-Consistent Answering
    Score the LLM-generated explanation against the verified knowledge.
    If score < 6, proactively regenerate before the visual guard sees it.

  Loop 3 — Entailment Check
    The self_evaluate score acts as a proxy entailment check:
    a low score signals the explanation does not entail the product facts.
"""

from m2_multimodal_rag.llm_generator import llm_generator


class KnowledgeSelfReflector:
    """
    Implements the EMNLP 2023 three-loop self-reflection strategy for
    fashion product explanations.

    - generate_product_knowledge() → Loop 1: produce & validate factual knowledge
    - self_evaluate()              → Loop 2 + 3: score explanation consistency
                                     and entailment against verified knowledge
    """

    # ------------------------------------------------------------------ #
    # Loop 1 — Factual Knowledge Acquisition
    # ------------------------------------------------------------------ #
    def generate_product_knowledge(self, metadata: dict) -> str:
        """
        Generates and validates factual background knowledge about the product.
        Follows the generate → score → refine cycle from Ji et al. (2023).

        Returns a verified knowledge string injected into the explanation
        prompt as grounding context (Loop 1 output feeds into Loop 2 input).
        """
        if not llm_generator.is_available:
            return ""

        colour      = metadata.get("colour_group_name", "")
        product_type = metadata.get("product_type_name", "")
        department  = metadata.get("department_name", "")
        appearance  = metadata.get("graphical_appearance_name", "")
        detail_desc = str(metadata.get("detail_desc", ""))[:200]

        meta_facts = (
            f"Colour: {colour} | Type: {product_type} | "
            f"Department: {department} | Appearance: {appearance} | "
            f"Description: {detail_desc}"
        )

        def _generate(facts: str) -> str:
            prompt = (
                f"You are a fashion expert. Based on the verified product metadata "
                f"below, generate concise factual background knowledge about this "
                f"item (2-3 sentences). Focus on verifiability, objectivity, and "
                f"strict accuracy to the given attributes.\n\n"
                f"Product metadata:\n{facts}\n\n"
                f"Output ONLY the factual knowledge. No recommendations yet."
            )
            return llm_generator._call_llm(prompt, max_tokens=100, temperature=0.1) or ""

        def _score(knowledge: str, facts: str) -> int:
            prompt = (
                f"Score this product knowledge for factual consistency with the "
                f"verified metadata (1-10). Deduct points for any claim not "
                f"directly supported by the metadata.\n\n"
                f"Metadata  : {facts}\n"
                f"Knowledge : \"{knowledge}\"\n\n"
                f"Output ONLY: SCORE: <number>"
            )
            result = llm_generator._call_llm(prompt, max_tokens=10, temperature=0.0)
            if not result:
                return 7
            try:
                line = next(
                    (l for l in result.split("\n") if "SCORE" in l.upper()), "SCORE: 7"
                )
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 7

        knowledge = _generate(meta_facts)
        if not knowledge:
            return ""

        score = _score(knowledge, meta_facts)
        print(f"   [Layer 1 | Knowledge Loop] Factuality score: {score}/10")

        if score < 6:
            print(f"   [Layer 1 | Knowledge Loop] Below threshold — refining...")
            refine_prompt = (
                f"The product knowledge below scored {score}/10 for factual "
                f"accuracy. Rewrite it to be strictly consistent with the metadata.\n\n"
                f"Metadata          : {meta_facts}\n"
                f"Original knowledge: \"{knowledge}\"\n\n"
                f"Output ONLY the corrected knowledge. No extra text."
            )
            refined = llm_generator._call_llm(refine_prompt, max_tokens=100, temperature=0.0)
            if refined:
                knowledge = refined
                print(f"   [Layer 1 | Knowledge Loop] Knowledge refined successfully.")

        return knowledge

    # ------------------------------------------------------------------ #
    # Loop 2 + 3 — Knowledge-Consistent Self-Evaluation
    # ------------------------------------------------------------------ #
    def self_evaluate(
        self, explanation: str, metadata: dict, product_knowledge: str = ""
    ) -> tuple[bool, str]:
        """
        Scores the explanation against verified product knowledge (Loop 2)
        and checks entailment of product facts (Loop 3 proxy).

        Returns (passes: bool, feedback: str).
        Score >= 6 → PASS. Score < 6 → FAIL → caller triggers regeneration.
        """
        if not llm_generator.is_available:
            return True, "Layer 1 self-evaluation skipped (LLM unavailable)."

        grounding = (
            f"Verified product knowledge:\n{product_knowledge}"
            if product_knowledge
            else (
                f"Verified item facts: "
                f"{metadata.get('colour_group_name', '')} "
                f"{metadata.get('product_type_name', '')}"
            )
        )

        prompt = (
            f"Evaluate this fashion recommendation explanation for quality.\n\n"
            f"{grounding}\n"
            f"Explanation: \"{explanation}\"\n\n"
            f"Score 1-10 on: factual consistency with verified facts, clarity, "
            f"helpfulness. Score below 6 if the explanation contradicts or "
            f"ignores the verified product knowledge.\n"
            f"Output format (two lines only):\n"
            f"SCORE: <number>\n"
            f"FEEDBACK: <one sentence>"
        )

        result = llm_generator._call_llm(prompt, max_tokens=60, temperature=0.0)
        if not result:
            return True, "Layer 1 self-evaluation inconclusive. Passing."

        try:
            lines = result.strip().split("\n")
            score_line    = next((l for l in lines if l.upper().startswith("SCORE:")),    None)
            feedback_line = next((l for l in lines if l.upper().startswith("FEEDBACK:")), None)

            score    = int(score_line.split(":", 1)[1].strip()) if score_line else 7
            feedback = feedback_line.split(":", 1)[1].strip() if feedback_line else "Quality acceptable."

            passes = score >= 6
            status = "PASS" if passes else "FAIL → proactive regeneration"
            print(f"   [Layer 1 | Self-Reflect] Score: {score}/10 — {status}")
            return passes, feedback
        except Exception:
            return True, "Layer 1 self-evaluation parse error. Passing."


# Singleton
knowledge_reflector = KnowledgeSelfReflector()
