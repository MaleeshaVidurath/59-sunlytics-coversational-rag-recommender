import re

from m2_multimodal_rag.llm_generator import llm_generator

_SCORE_RE = re.compile(r"SCORE\s*:\s*(\d+)", re.IGNORECASE)


# Implements Ji et al. (2023), "Towards Mitigating LLM Hallucination via
# Self Reflection" (ACL Findings EMNLP 2023, 2023.findings-emnlp.123).
class KnowledgeSelfReflector:
    # ------------------------------------------------------------------ #
    # Loop 1 — Factual Knowledge Acquisition
    # ------------------------------------------------------------------ #
    def generate_product_knowledge(self, metadata: dict) -> str:
        # No LLM configured -> nothing to verify with. Fail open: the caller
        # falls back to the pre-filter draft untouched.
        if not llm_generator.is_available:
            return ""

        # Step 1: pull only catalog-verified fields (never model output) —
        # this is the ground truth every later score in this file is
        # measured against.
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
        print(f"   [Layer 1 | Knowledge Loop] Verified facts: {meta_facts}")

        # Step 2: generate — ask the LLM to restate the facts as prose.
        # temperature=0.1 and "No recommendations yet" deliberately keep
        # this factual/boring rather than persuasive, since persuasive
        # writing is where models tend to invent unsupported detail.
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

        # Step 3: score — a SEPARATE LLM call fact-checks the generated
        # knowledge against meta_facts (not the same call that wrote it).
        # Any parse/API failure defaults to 7 (a pass) but logs it as
        # UNVERIFIED so a genuine 7/10 isn't confused with "couldn't check".
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
                print("   [Layer 1 | Knowledge Loop] Score UNVERIFIED (empty LLM "
                      "response) — defaulting to pass (7/10).")
                return 7
            match = _SCORE_RE.search(result)
            if match:
                return int(match.group(1))
            print(f"   [Layer 1 | Knowledge Loop] Score UNVERIFIED (could not parse "
                  f"\"{result}\") — defaulting to pass (7/10).")
            return 7

        knowledge = _generate(meta_facts)
        if not knowledge:
            return ""
        print(f"   [Layer 1 | Knowledge Loop] Generated knowledge: \"{knowledge}\"")

        score = _score(knowledge, meta_facts)
        print(f"   [Layer 1 | Knowledge Loop] Factuality score: {score}/10")

        # Step 4: refine — one corrective rewrite if the score fails the
        # bar (6/10, same threshold used in self_evaluate() below). The
        # rewrite is NOT re-scored, so this is a best-effort correction,
        # not a guaranteed fix.
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

        # This "knowledge" string is the deliverable of Loop 1: it becomes
        # the grounding context the real explanation is generated from,
        # and the self_evaluate() checks that explanation against.
        return knowledge

    # ------------------------------------------------------------------ #
    # Loop 2 + 3 — Knowledge-Consistent Self-Evaluation
    # ------------------------------------------------------------------ #
    def self_evaluate(
        self, explanation: str, metadata: dict, product_knowledge: str = ""
    ) -> tuple[bool, str]:
        """
        Scores the explanation for factual consistency with verified knowledge
        (Loop 2) as a proxy entailment check (Loop 3).

        Grounds against `product_knowledge` (Loop 1 output) when available,
        else falls back to raw colour/type metadata.
        """
        # Fail open: no LLM available means nothing can be checked, so this
        # is treated as a pass rather than blocking the pipeline.
        if not llm_generator.is_available:
            return True, "Layer 1 self-evaluation skipped (LLM unavailable)."

        # Ground truth for this check: prefer the Loop 1 knowledge string
        # (richer), fall back to raw colour/type metadata if Loop 1 never
        # ran (e.g. it returned "").
        grounding = (
            f"Verified product knowledge:\n{product_knowledge}"
            if product_knowledge
            else (
                f"Verified item facts: "
                f"{metadata.get('colour_group_name', '')} "
                f"{metadata.get('product_type_name', '')}"
            )
        )

        # Loop 2 — Knowledge-Consistent Answering: score the explanation
        # against `grounding` (the verified knowledge/facts above).
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

        # This is the actual gate: score the EXPLANATION (not the knowledge
        # string) against grounding. Same fail-open pattern as _score()
        # above — any failure here defaults to a pass, never a crash.
        result = llm_generator._call_llm(prompt, max_tokens=60, temperature=0.0)
        if not result:
            return True, "Layer 1 self-evaluation inconclusive. Passing."

        try:
            lines = result.strip().split("\n")
            score_line    = next((l for l in lines if l.upper().startswith("SCORE:")),    None)
            feedback_line = next((l for l in lines if l.upper().startswith("FEEDBACK:")), None)

            score    = int(score_line.split(":", 1)[1].strip()) if score_line else 7
            feedback = feedback_line.split(":", 1)[1].strip() if feedback_line else "Quality acceptable."

            # Loop 3 — Entailment Check (proxy): the same score doubles as
            # the entailment signal — low score means the explanation does
            # not entail the verified facts, not just that it scored poorly.
            # >= 6 passes -> continues to Layer 2 (CoVe). < 6 fails -> the
            # orchestrator (_stage_layer1 in regeneration_loop.py) does ONE
            # regenerate call with `feedback`, and does not re-run this check.
            passes = score >= 6
            status = "PASS" if passes else "FAIL → proactive regeneration"
            print(f"   [Layer 1 | Self-Reflect] Score: {score}/10 — {status}")
            return passes, feedback
        except Exception:
            return True, "Layer 1 self-evaluation parse error. Passing."


# Singleton
knowledge_reflector = KnowledgeSelfReflector()
