"""
4-Stage Hallucination Guard — Live Demo
Shows the full pipeline running on a real product article.

Run from project root:
    python Accuracy/hallucination_guard_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m2_multimodal_rag.hallucination_guard.regeneration_loop import generator_loop

SEP = "=" * 65

# Article ID to demonstrate — Washington Dress (Black, Dress, Stripe)
ARTICLE_ID = "0702938001"

print(f"\n{SEP}")
print("  NOVELTY 4 — 4-Stage Hallucination Guard  |  Live Demo")
print(f"  Article: {ARTICLE_ID}")
print(SEP)
print("\n  Pipeline: Pre-filter → Layer 1 → Layer 3 → Layer 2")
print("  Each FAIL triggers regeneration with corrective feedback.\n")

explanation, trail = generator_loop.generate_faithful_explanation(
    article_id=ARTICLE_ID,
)

print(f"\n{SEP}")
print("  VERIFICATION TRAIL")
print(SEP)
print(f"  Layers passed   : {trail.get('layers_passed', [])}")
print(f"  CLIP score      : {trail.get('clip_score', 'N/A')}")
print(f"  CoVe score      : {trail.get('cove_consistency_score', 'N/A')}")
print(f"  VLM verified    : {trail.get('vlm_verified', False)}")
print(f"\n  Final explanation:")
print(f"  \"{explanation}\"")
print(f"{SEP}\n")
