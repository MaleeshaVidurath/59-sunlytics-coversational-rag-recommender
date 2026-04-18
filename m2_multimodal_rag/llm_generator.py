import random

class ExplanationGenerator:
    """
    Simulates the LLaMA-3 Language Model block for M2.
    Translates raw retrieved item metadata into conversational explanations.
    This component includes a 'hallucination' mode strictly to test the BLIP Guard!
    """
    def __init__(self, is_mock=True):
        self.is_mock = is_mock
        print("M2 LLM: Initializing Explanation Generator...")

    def generate(self, article_id: str, metadata: dict, force_hallucination=False) -> str:
        """
        Generates a natural language explanation of why the item was recommended.
        """
        if not self.is_mock:
            # TODO: Integrate actual HuggingFace LLaMA-3 or LangChain logic here
            raise NotImplementedError("Real LLM mode temporarily disabled for offline M2 architectural testing.")
            
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')
        
        if force_hallucination:
            # CAUTION: We intentionally instruct the mock LLM to lie about the color
            # so we can prove our VLM Guard catches errors mathematically!
            wrong_colors = ['neon green', 'hot pink', 'silver', 'striped magenta']
            bad_color = random.choice(wrong_colors)
            return f"I highly recommend this item! As you can see, it features a beautiful {bad_color} design."
            
        # Faithful Generation
        return f"I recommend this item because it is a stylish {color} {product_type} that matches your search."
        
    def regenerate(self, article_id: str, metadata: dict, visual_feedback: str) -> str:
        """
        Triggered ONLY if the BLIP Verification Guard rejects the previous explanation.
        The LLM is prompted with the visual feedback to constrain its next attempt.
        """
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')
        
        # In a real deployed LLM, we inject `visual_feedback` into the prompt here.
        # For mock testing, we simply output a corrected, truthful response.
        print("\n[LLM INTERNAL] Received strict feedback from Visual Guard. Regenerating response...")
        return f"Correcting my previous statement: based on verified visual evidence, this is a {color} {product_type}."

# Singleton
llm_generator = ExplanationGenerator(is_mock=True)
