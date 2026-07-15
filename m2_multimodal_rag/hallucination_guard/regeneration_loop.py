"""
M2 Hallucination Guard — Orchestrator / Regeneration Loop.

Pipeline (in execution order):
  Pre-filter  — Direct attribute check (deterministic, zero LLM cost)
  Layer 1     — Knowledge-Grounded Self-Reflection (Ji et al., EMNLP 2023)
  Layer 2     — Enhanced CoVe with DeBERTa NLI (Dhuliawala et al., 2023)
  Layer 3     — CLIPScore faithfulness (Hessel et al., 2021)
              + ViLT VQA visual verification (cross-modal ground truth)

Return value: (explanation: str, verification_trail: dict)
"""

from shared.data_loader import data_loader
from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.hallucination_guard.layer_1_knowledge_self_reflection import knowledge_reflector
from m2_multimodal_rag.hallucination_guard.layer_3_vlm_visual_verification    import blip_verifier
from m2_multimodal_rag.hallucination_guard.layer_2_cove_verification           import cove_verifier
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
        Deterministic pre-filter. Only checks colour — the one field that
        must appear verbatim and whose hallucination is always critical
        (saying 'red' for a black item is unambiguously wrong).

        product_type excluded: vision LLM may correctly call a catalog
        'Shirt' a 'blouse' based on image evidence — not a hallucination.
        appearance excluded: 'Solid', 'Striped' are catalog tags that
        never appear in natural language recommendations.
        """
        exp_lower = explanation.lower()
        colour = str(metadata.get("colour_group_name", "")).strip()
        if colour and colour.lower() not in ("", "unknown"):
            if colour.lower() not in exp_lower:
                return {"colour": colour}
        return {}

    # ── Pipeline stages ────────────────────────────────────────────────────────

    def _stage_prefilter(
        self, article_id: str, metadata: dict,
        force_hallucination: bool, trail: dict, image_path: str = "",
    ) -> str:
        """Pre-filter: generate initial explanation, fix obvious attribute errors."""
        explanation = llm_generator.generate(
            article_id, metadata,
            force_hallucination=force_hallucination,
            product_knowledge="",
            image_path=image_path,
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
                article_id, metadata, visual_feedback=feedback, image_path=image_path,
            )
            trail["layers_passed"].append("pre_filter_corrected")
        else:
            trail["layers_passed"].append("pre_filter_passed")
            print("   [Pre-filter] PASS — core attributes correct")
        return explanation

    def _stage_layer1(
        self, article_id: str, metadata: dict,
        preflight: str, force_hallucination: bool, trail: dict, image_path: str = "",
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
                image_path=image_path,
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
                article_id, metadata, visual_feedback=self_feedback, image_path=image_path,
            )
        else:
            trail["layers_passed"].append("layer_1c_self_reflection")

        return explanation

    def _stage_layer2(
        self, article_id: str, metadata: dict,
        explanation: str, trail: dict, image_path: str = "",
    ) -> str:
        """Layer 2: Enhanced CoVe with DeBERTa NLI."""
        print("   [Layer 2] CoVe verification...")
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
            print("   [Layer 2] FAIL — regenerating with CoVe feedback...")
            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=feedback, image_path=image_path,
            )
        else:
            trail["layers_passed"].append("layer_2_cove_verification")

        return explanation

    def _stage_layer3(
        self, article_id: str, image_path, metadata: dict,
        explanation: str, trail: dict,
    ) -> tuple[str, bool]:
        """
        Layer 3: CLIPScore faithfulness + ViLT VQA visual verification.

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

        image_path_str = str(image_path)

        if not clip_passes:
            print("   [Layer 3 | CLIPScore] FAIL — regenerating with visual feedback")
            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=clip_feedback, image_path=image_path_str,
            )
            trail["layers_passed"].append("layer_3_clipscore_corrected")
        else:
            trail["layers_passed"].append("layer_3_clipscore_passed")

        # ── ViLT VQA loop ──────────────────────────────────────────────────
        for attempt in range(1, self.max_vlm_attempts + 1):
            print(f"   [Layer 3 | ViLT] attempt {attempt}/{self.max_vlm_attempts}...")
            is_valid, reason = blip_verifier.verify(image_path_str, explanation)

            if is_valid:
                print("   [Layer 3 | ViLT] PASS")
                trail["vlm_verified"] = True
                trail["layers_passed"].append("layer_3_vilt_verification")
                return explanation, True

            print(f"   [Layer 3 | ViLT] FAIL — {reason}")

            if attempt == self.max_vlm_attempts:
                print("   [Layer 3 | ViLT] Max retries — falling back to metadata template.")
                fallback = (
                    f"This is a {metadata.get('colour_group_name', 'Black')} "
                    f"{metadata.get('product_type_name', 'item')}."
                )
                return fallback, True

            explanation = llm_generator.regenerate(
                article_id, metadata, visual_feedback=reason, image_path=image_path_str,
            )

        return explanation, False

    # ── Prompt-driven entry point (attribute/compare/explain/detail handlers) ──

    def _reprompt(self, prompt: str, feedback: str, current: str) -> str:
        """Re-issues the handler's original prompt with corrective feedback appended."""
        regenerated = llm_generator._call_llm(
            prompt + f"\n\n[Your previous answer failed verification: {feedback}. "
                     f"Rewrite it, correcting these issues.]",
            max_tokens=250,
        )
        return regenerated or current

    def verify_prompted_response(
        self,
        prompt: str,
        article_id: str,
        metadata: dict,
        skip_visual: bool = False,
    ) -> tuple[str | None, dict]:
        """
        Full 3-layer guard for prompt-driven handler responses (attribute
        lookup, item compare, explanation-why, item detail lookup).

        Differences from generate_faithful_explanation():
          - The initial text comes from the handler's own prompt, and every
            regeneration re-issues that prompt with corrective feedback, so
            the answer stays on-topic instead of becoming a generic
            recommendation explanation.
          - The colour pre-filter is skipped: a factual answer ("Yes, it's
            machine washable") legitimately may not mention the colour.
          - If the ViLT visual gate never passes, returns None (caller falls
            back to a safe metadata template) instead of unverified text.
          - skip_visual=True skips Layer 3 entirely: answers about non-visual
            topics (material/care, price, availability) are not descriptions
            of the image, so CLIPScore/ViLT reject them spuriously. Layers
            1+2 still verify against metadata.

        Returns:
            response  str|None  Verified response, or None if generation
                                failed / visual verification never passed.
            trail     dict      Audit trail (same shape as catalog search).
        """
        image_path = data_loader.get_image(article_id)
        has_image = image_path is not None and image_path.exists()
        image_path_str = str(image_path) if has_image else ""

        print(f"\n=== Hallucination Guard (prompted): Article {article_id} ===")

        trail: dict = {
            "clip_score":             None,
            "cove_trail":             [],
            "cove_consistency_score": None,
            "vlm_verified":           False,
            "layers_passed":          [],
            "image_grounded":         has_image,
        }

        response = llm_generator._call_llm(prompt, max_tokens=250)
        if not response:
            return None, trail

        # ── Layer 1: knowledge-grounded self-reflection ────────────────────
        print("   [Layer 1] Knowledge-grounded self-reflection...")
        product_knowledge = knowledge_reflector.generate_product_knowledge(metadata)
        if product_knowledge:
            trail["layers_passed"].append("layer_1a_knowledge_acquisition")
        passes_self, self_feedback = knowledge_reflector.self_evaluate(
            response, metadata, product_knowledge=product_knowledge
        )
        if not passes_self:
            print(f"   [Layer 1] FAIL — regenerating. Reason: {self_feedback}")
            response = self._reprompt(prompt, self_feedback, response)
        else:
            trail["layers_passed"].append("layer_1c_self_reflection")

        # ── Layer 2: Enhanced CoVe with DeBERTa NLI ────────────────────────
        print("   [Layer 2] CoVe verification...")
        cove_passes, cove_trail, cove_score = cove_verifier.verify(response, metadata)
        trail["cove_trail"]             = cove_trail
        trail["cove_consistency_score"] = round(cove_score, 3)
        if not cove_passes:
            failed = [
                f"'{t['question']}' (metadata: {t['metadata_answer']})"
                for t in cove_trail if not t["consistent"]
            ]
            feedback = (
                "claims inconsistent with the item's catalog data: "
                + "; ".join(failed)
            )
            print("   [Layer 2] FAIL — regenerating with CoVe feedback...")
            response = self._reprompt(prompt, feedback, response)
        else:
            trail["layers_passed"].append("layer_2_cove_verification")

        # ── Layer 3: CLIPScore + ViLT VQA (image required) ─────────────────
        if skip_visual:
            trail["layers_passed"].append("layer_3_skipped_non_visual")
            print("   [Layer 3] Non-visual topic — skipping visual gate (Layers 1+2 verified).")
            return response, trail
        if not has_image:
            trail["layers_passed"].append("layer_3_skipped_no_image")
            print(f"   [Layer 3] No image for {article_id} — skipping visual gate.")
            return response, trail

        clip_score, clip_passes, clip_feedback = clip_faithfulness_scorer.score(
            image_path_str, response
        )
        trail["clip_score"] = round(clip_score, 4)
        if not clip_passes:
            print("   [Layer 3 | CLIPScore] FAIL — regenerating with visual feedback")
            response = self._reprompt(prompt, clip_feedback, response)
            trail["layers_passed"].append("layer_3_clipscore_corrected")
        else:
            trail["layers_passed"].append("layer_3_clipscore_passed")

        for attempt in range(1, self.max_vlm_attempts + 1):
            print(f"   [Layer 3 | ViLT] attempt {attempt}/{self.max_vlm_attempts}...")
            is_valid, reason = blip_verifier.verify(image_path_str, response)
            if is_valid:
                print("   [Layer 3 | ViLT] PASS")
                trail["vlm_verified"] = True
                trail["layers_passed"].append("layer_3_vilt_verification")
                return response, trail
            print(f"   [Layer 3 | ViLT] FAIL — {reason}")
            if attempt < self.max_vlm_attempts:
                response = self._reprompt(prompt, f"visual check failed: {reason}", response)

        # Visual gate never passed — signal the caller to use its safe fallback
        print("   [Layer 3 | ViLT] Max retries — returning None (caller uses template fallback).")
        return None, trail

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
        has_image = image_path is not None and image_path.exists()
        image_path_str = str(image_path) if has_image else ""

        print(f"\n=== Hallucination Guard: Article {article_id} ===")
        if not has_image:
            print("   [Guard] No image found — running text-only pipeline (Layers 1+2 active, Layer 3 skipped).")

        trail: dict = {
            "knowledge_score":        None,
            "self_reflection_score":  None,
            "clip_score":             None,
            "cove_trail":             [],
            "cove_consistency_score": None,
            "vlm_verified":           False,
            "layers_passed":          [],
            "image_grounded":         has_image,
        }

        preflight   = self._stage_prefilter(article_id, metadata, force_hallucination_test, trail, image_path_str)
        explanation = self._stage_layer1(article_id, metadata, preflight, force_hallucination_test, trail, image_path_str)
        explanation = self._stage_layer2(article_id, metadata, explanation, trail, image_path_str)

        # Layer 2 requires a valid image — skip it gracefully when unavailable
        if has_image:
            explanation, _ = self._stage_layer3(article_id, image_path, metadata, explanation, trail)
        else:
            trail["layers_passed"].append("layer_3_skipped_no_image")

        return explanation, trail


# Singleton
generator_loop = GenerationLoop()
