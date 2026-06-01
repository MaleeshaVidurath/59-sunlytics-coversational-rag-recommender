"""
M2 Hallucination Guard — Orchestrator / Regeneration Loop.

Pipeline (in execution order):
  Pre-filter  — Direct attribute check (deterministic, zero LLM cost)
  Layer 1     — Knowledge-Grounded Self-Reflection (Ji et al., EMNLP 2023)
  Layer 3     — Enhanced CoVe with DeBERTa NLI (Dhuliawala et al., 2023)
  Layer 2     — CLIPScore faithfulness (Hessel et al., 2021)
              + ViLT VQA visual verification (cross-modal ground truth)

Return value: (explanation: str, verification_trail: dict)
"""

from shared.data_loader import data_loader
from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.hallucination_guard.layer_1_knowledge_self_reflection import knowledge_reflector
from m2_multimodal_rag.hallucination_guard.layer_2_vlm_visual_verification    import blip_verifier
from m2_multimodal_rag.hallucination_guard.layer_3_cove_verification           import cove_verifier
from m2_multimodal_rag.hallucination_guard.clip_faithfulness_scorer            import clip_faithfulness_scorer


class GenerationLoop:
    """
    Orchestrates the full hallucination guard pipeline for explanation generation.
    Each sub-method handles one pipeline stage to keep complexity low.
    """

    def __init__(self, max_vlm_attempts: int = 2):
        self.max_vlm_attempts = max_vlm_attempts
        print("M2 Guard: Hallucination Guard pipeline initialised.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _direct_attribute_check(self, explanation: str, metadata: dict) -> dict:
        """
        Deterministic pre-filter. Checks colour, type, and appearance directly
        against metadata without any LLM call.
        Returns {attribute: correct_value} for each attribute missing from
        the explanation. Empty dict = all clear.
        """
        exp_lower = explanation.lower()
        checks = [
            ("colour",       metadata.get("colour_group_name", "")),
            ("product_type", metadata.get("product_type_name", "")),
            ("appearance",   metadata.get("graphical_appearance_name", "")),
        ]
        return {
            label: value
            for label, value in checks
            if value and value.strip().lower() not in ("", "unknown")
            and value.lower() not in exp_lower
        }

    # ── Pipeline stages ────────────────────────────────────────────────────────

    def _stage_prefilter(
        self, article_id: str, metadata: dict,
        force_hallucination: bool, trail: dict,
    ) -> str:
        """Pre-filter: generate initial explanation, fix obvious attribute errors."""
        explanation = llm_generator.generate(
            article_id, metadata,
            force_hallucination=force_hallucination,
            product_knowledge="",
        )
        fails = self._direct_attribute_check(explanation, metadata)
        if fails:
            print(f"   [Pre-filter] FAIL on: {fails} — regenerating")
            feedback = (
                "The explanation contains wrong attribute values: "
                + "; ".join(f"{k} should be '{v}'" for k, v in fails.items())
                + ". Correct these facts."
            )
            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=feedback
            )
            trail["layers_passed"].append("pre_filter_corrected")
        else:
            trail["layers_passed"].append("pre_filter_passed")
            print("   [Pre-filter] PASS — core attributes correct")
        return explanation

    def _stage_layer1(
        self, article_id: str, metadata: dict,
        preflight: str, force_hallucination: bool, trail: dict,
    ) -> str:
        """Layer 1: knowledge acquisition + grounded generation + self-reflection."""
        print("   [Layer 1-A] Acquiring verified product knowledge...")
        product_knowledge = knowledge_reflector.generate_product_knowledge(metadata)
        if product_knowledge:
            trail["layers_passed"].append("layer_1a_knowledge_acquisition")
            print(f"   [Layer 1-A] Knowledge: \"{product_knowledge[:80]}...\"")

        explanation = (
            llm_generator.generate(
                article_id, metadata,
                force_hallucination=force_hallucination,
                product_knowledge=product_knowledge,
            )
            if product_knowledge else preflight
        )
        print(f"   [Layer 1-B] Explanation: \"{explanation}\"")

        print("   [Layer 1-C] Self-reflection...")
        passes_self, self_feedback = knowledge_reflector.self_evaluate(
            explanation, metadata, product_knowledge=product_knowledge
        )
        if not passes_self:
            print(f"   [Layer 1-C] FAIL — regenerating. Reason: {self_feedback}")
            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=self_feedback
            )
        else:
            trail["layers_passed"].append("layer_1c_self_reflection")

        return explanation

    def _stage_layer3(
        self, article_id: str, metadata: dict,
        explanation: str, trail: dict,
    ) -> str:
        """Layer 3: Enhanced CoVe with DeBERTa NLI."""
        print("   [Layer 3] CoVe verification...")
        cove_passes, cove_trail, cove_score = cove_verifier.verify(explanation, metadata)
        trail["cove_trail"]             = cove_trail
        trail["cove_consistency_score"] = round(cove_score, 3)

        if not cove_passes:
            failed = [
                f"'{t['question']}' (metadata: {t['metadata_answer']})"
                for t in cove_trail if not t["consistent"]
            ]
            feedback = (
                "CoVe failed — claims inconsistent with metadata: "
                + "; ".join(failed) + ". Correct these."
            )
            print("   [Layer 3] FAIL — regenerating with CoVe feedback...")
            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=feedback
            )
        else:
            trail["layers_passed"].append("layer_3_cove_verification")

        return explanation

    def _stage_layer2(
        self, article_id: str, image_path, metadata: dict,
        explanation: str, trail: dict,
    ) -> tuple[str, bool]:
        """
        Layer 2: CLIPScore faithfulness + ViLT VQA visual verification.

        CLIPScore (Hessel et al., 2021) measures cosine similarity between
        CLIP image and text embeddings — a continuous faithfulness metric.
        ViLT VQA provides a complementary binary visual consistency check.

        Returns (explanation, done) where done=True signals early return.
        """
        # ── CLIPScore (image-text alignment metric) ────────────────────────
        clip_score, clip_passes, clip_feedback = clip_faithfulness_scorer.score(
            str(image_path), explanation
        )
        trail["clip_score"] = round(clip_score, 4)

        if not clip_passes:
            print("   [Layer 2 | CLIPScore] FAIL — regenerating with visual feedback")
            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=clip_feedback
            )
            trail["layers_passed"].append("layer_2_clipscore_corrected")
        else:
            trail["layers_passed"].append("layer_2_clipscore_passed")

        # ── ViLT VQA loop ──────────────────────────────────────────────────
        for attempt in range(1, self.max_vlm_attempts + 1):
            print(f"   [Layer 2 | ViLT] attempt {attempt}/{self.max_vlm_attempts}...")
            is_valid, reason = blip_verifier.verify(str(image_path), explanation)

            if is_valid:
                print("   [Layer 2 | ViLT] PASS")
                trail["vlm_verified"] = True
                trail["layers_passed"].append("layer_2_vilt_verification")
                return explanation, True

            print(f"   [Layer 2 | ViLT] FAIL — {reason}")

            if attempt == self.max_vlm_attempts:
                print("   [Layer 2 | ViLT] Max retries — falling back to metadata template.")
                fallback = (
                    f"This is a {metadata.get('colour_group_name', 'Black')} "
                    f"{metadata.get('product_type_name', 'item')}."
                )
                return fallback, True

            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=reason
            )

        return explanation, False

    # ── Main entry point ───────────────────────────────────────────────────────

    def generate_faithful_explanation(
        self,
        article_id: str,
        force_hallucination_test: bool = False,
        kb_fact: str = "",
    ) -> tuple[str, dict]:
        """
        Runs the full hallucination guard pipeline for one product.

        Returns:
            explanation        str   Verified, faithful explanation text.
            verification_trail dict  Full audit trail exposed in the API response.
        """
        articles_df = data_loader.load_articles()
        rows = articles_df[articles_df["article_id"] == int(article_id)].to_dict("records")
        if not rows:
            return "Item not found in database.", {}
        metadata = rows[0]

        if kb_fact:
            metadata["kb_psychology_fact"] = kb_fact

        image_path = data_loader.get_image(article_id)
        if not image_path or not image_path.exists():
            return "Visual evidence not available for verification.", {}

        print(f"\n=== Hallucination Guard: Article {article_id} ===")

        trail: dict = {
            "knowledge_score":        None,
            "self_reflection_score":  None,
            "clip_score":             None,
            "cove_trail":             [],
            "cove_consistency_score": None,
            "vlm_verified":           False,
            "layers_passed":          [],
        }

        preflight   = self._stage_prefilter(article_id, metadata, force_hallucination_test, trail)
        explanation = self._stage_layer1(article_id, metadata, preflight, force_hallucination_test, trail)
        explanation = self._stage_layer3(article_id, metadata, explanation, trail)
        explanation, _ = self._stage_layer2(article_id, image_path, metadata, explanation, trail)

        return explanation, trail


# Singleton
generator_loop = GenerationLoop()
