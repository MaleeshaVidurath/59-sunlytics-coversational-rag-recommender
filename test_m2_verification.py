import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from m2_multimodal_rag.regeneration_loop import generator_loop

def run_tests():
    print("==================================================")
    print("🛡️ M2 VISUAL VERIFICATION & REGENERATION GUARD TEST")
    print("==================================================\n")
    
    # We use our locally cached test image 
    test_article = "0108775015"
    
    print("[SCENARIO 1]: Faithful Generation")
    print("Testing a normal, non-hallucinating LLM sequence...")
    final_output = generator_loop.generate_faithful_explanation(test_article, force_hallucination_test=False)
    print(f"\n✨ APPROVED OUTPUT SENT TO USER: \"{final_output}\"\n")
    
    print("-" * 65)
    
    print("\n[SCENARIO 2]: Force Hallucination (Triggering Regeneration Loop!)")
    print("We intentionally corrupt the LLM to hallucinate a completely wrong color.")
    print("The VLM Guard must catch this mathematically and force a rewrite.")
    final_output_2 = generator_loop.generate_faithful_explanation(test_article, force_hallucination_test=True)
    print(f"\n✨ REGENERATED APPROVED OUTPUT SENT TO USER: \"{final_output_2}\"\n")

if __name__ == "__main__":
    run_tests()
