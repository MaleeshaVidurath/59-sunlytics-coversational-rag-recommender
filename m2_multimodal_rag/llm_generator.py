import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ExplanationGenerator:
    """
    Cloud-Based LLM Explanation Generator for M2 Multimodal RAG.
    Uses Google Gemini API to translate retrieved item metadata into
    conversational fashion recommendation explanations.
    Includes a 'hallucination' mock mode strictly to test the BLIP Guard!
    """
    def __init__(self):
        self.client = None
        self.model_name = "gemini-2.5-flash"
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key and api_key != "your_key_here":
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                print(f"M2 LLM: [SUCCESS] Gemini Cloud LLM initialized (Model: {self.model_name})")
            except Exception as e:
                print(f"M2 LLM: [WARNING] Failed to initialize Gemini client: {e}")
                print("M2 LLM: Falling back to mock mode.")
        else:
            print("M2 LLM: [WARNING] No GEMINI_API_KEY found in .env -- running in MOCK mode.")
            print("M2 LLM: To enable real AI, add your key to the .env file.")

    def _call_gemini(self, prompt: str) -> str:
        """
        Sends a prompt to the Gemini API and returns the generated text.
        Falls back to None if the API call fails.
        """
        if not self.client:
            return None
            
        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                )
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"   [LLM API Error] {e}")
            return None

    def generate(self, article_id: str, metadata: dict, force_hallucination=False) -> str:
        """
        Generates a natural language explanation of why the item was recommended.
        Uses Gemini Cloud API for real generation, with mock fallback.
        """
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')
        
        if force_hallucination:
            # CAUTION: We intentionally instruct the mock LLM to lie about the color
            # so we can prove our VLM Guard catches errors mathematically!
            wrong_colors = ['neon green', 'hot pink', 'silver', 'striped magenta']
            bad_color = random.choice(wrong_colors)
            return f"I highly recommend this item! As you can see, it features a beautiful {bad_color} design."
        
        # Try Gemini Cloud API first
        if self.client:
            department = metadata.get('department_name', 'Fashion')
            category = metadata.get('product_group_name', 'Clothing')
            detail_desc = metadata.get('detail_desc', '')
            
            prompt = (
                f"You are a friendly fashion recommendation assistant. "
                f"Generate a warm, conversational 1-2 sentence explanation of why "
                f"this item is a great recommendation for the customer.\n\n"
                f"Item details:\n"
                f"- Product Type: {product_type}\n"
                f"- Color: {color}\n"
                f"- Department: {department}\n"
                f"- Category: {category}\n"
                f"- Description: {detail_desc}\n\n"
                f"Respond with ONLY the recommendation explanation, nothing else. "
                f"Do not start with 'I recommend' — be creative and natural."
            )
            
            result = self._call_gemini(prompt)
            if result:
                return result
        
        # Fallback to mock template if API unavailable
        return f"I recommend this item because it is a stylish {color} {product_type} that matches your search."
        
    def regenerate(self, article_id: str, metadata: dict, visual_feedback: str) -> str:
        """
        Triggered ONLY if the BLIP Verification Guard rejects the previous explanation.
        The LLM is prompted with the visual feedback to constrain its next attempt.
        """
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')
        
        print("\n[LLM INTERNAL] Received strict feedback from Visual Guard. Regenerating response...")
        
        # Try Gemini Cloud API with corrective feedback
        if self.client:
            prompt = (
                f"Your previous fashion recommendation was REJECTED by our visual verification system "
                f"because: \"{visual_feedback}\"\n\n"
                f"Generate a corrected 1-2 sentence recommendation that ACCURATELY describes "
                f"this {color} {product_type}. Only describe features that are visually confirmed.\n\n"
                f"Respond with ONLY the corrected recommendation, nothing else."
            )
            
            result = self._call_gemini(prompt)
            if result:
                return result
        
        # Fallback to mock template
        return f"Correcting my previous statement: based on verified visual evidence, this is a {color} {product_type}."

# Singleton
llm_generator = ExplanationGenerator()
