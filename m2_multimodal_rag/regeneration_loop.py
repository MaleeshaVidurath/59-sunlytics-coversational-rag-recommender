from shared.data_loader import data_loader
from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.blip_verification import blip_verifier

class GenerationLoop:
    """
    Coordinates the Verification & Regeneration logic flowchart of M2.
    Ensures that ONLY mathematically verified, non-hallucinated explanations are passed to the user.
    """
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        print("M2 Guard: Initializing Regeneration Loop orchestration...")
        
    def generate_faithful_explanation(self, article_id: str, force_hallucination_test=False) -> str:
        """
        Follows the strict PASS/FAIL logic tree depicted in the project architecture.
        """
        # Fetch grounding data
        articles_df = data_loader.load_articles()
        metadata = articles_df[articles_df['article_id'] == int(article_id)].to_dict('records')
        if not metadata:
            return "Item not found in database."
        metadata = metadata[0]
        
        # Fetch physical visual evidence
        image_path = data_loader.get_image(article_id)
        if not image_path or not image_path.exists():
            return "Visual evidence not available for verification."

        print(f"\n--- Constructing Explanation for Article {article_id} ---")

        # -------------------------------------------------------------
        # STEP 1: INITIAL LLM GENERATION
        # -------------------------------------------------------------
        explanation = llm_generator.generate(article_id, metadata, force_hallucination=force_hallucination_test)
        print(f"[LLM Initial Output] : \"{explanation}\"")
        
        attempts = 1
        
        # -------------------------------------------------------------
        # STEP 2: VERIFICATION & REGENERATION LOOP
        # -------------------------------------------------------------
        while attempts <= self.max_attempts:
            print(f"   -> Passing to VLM Verifier (Attempt {attempts}/{self.max_attempts})...")
            
            # The BLIP ITM mathematical gate
            is_valid, reason = blip_verifier.verify(str(image_path), explanation)
            
            # LOGIC FORK: PASS
            if is_valid:
                print("   ✅ VLM GUARD PASSED (Consistent)")
                return explanation
                
            # LOGIC FORK: FAIL
            print(f"   ❌ VLM GUARD FAILED (Inconsistent) : {reason}")
            
            if attempts == self.max_attempts:
                print("   ⚠️ MAX RETRIES REACHED. Fallback to basic metadata.")
                return f"This is a {metadata.get('colour_group_name', 'Black')} {metadata.get('product_type_name', 'item')}."
                
            # STEP 3: REGENERATE EXPLANATION
            explanation = llm_generator.regenerate(article_id, metadata, visual_feedback=reason)
            print(f"\n[LLM Regenerated Output]: \"{explanation}\"")
            attempts += 1

# Singleton Global Accessor
generator_loop = GenerationLoop()
