"""
M2 Hallucination Guard Package — Three-Layer Verified Generation Pipeline.

Layer 1 — layer_1_knowledge_self_reflection.py
    EMNLP 2023 knowledge-grounded self-reflection loop.
    Reference: Ji et al. (2023), "Towards Mitigating LLM Hallucination via
    Self Reflection", EMNLP 2023 Findings (2023.findings-emnlp.123).

Layer 2 — layer_2_vlm_visual_verification.py
    ViLT cross-modal image-text verification.

Layer 3 — layer_3_cove_verification.py
    Chain-of-Verification (CoVe) post-generation hallucination check.
    Reference: Dhuliawala et al. (2023). Cited in Tonmoy et al. (2024),
    arXiv:2401.01313v3, Section 2.1.2.

Orchestrator — regeneration_loop.py
    Coordinates all three layers into a single verified generation pipeline.
"""

from m2_multimodal_rag.hallucination_guard.layer_1_knowledge_self_reflection import knowledge_reflector
from m2_multimodal_rag.hallucination_guard.layer_2_vlm_visual_verification import blip_verifier
from m2_multimodal_rag.hallucination_guard.layer_3_cove_verification import cove_verifier
from m2_multimodal_rag.hallucination_guard.regeneration_loop import generator_loop

__all__ = ["knowledge_reflector", "blip_verifier", "cove_verifier", "generator_loop"]
