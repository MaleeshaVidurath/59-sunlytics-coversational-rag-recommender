import random
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ExplanationGenerator:
    """
    Cloud LLM Explanation Generator for M2 Multimodal RAG.
    Uses Groq API (free tier) to run Llama 3.1 in the cloud instead of locally.
    This eliminates all local RAM/GPU requirements for text generation.
    Includes a 'hallucination' mock mode strictly to test the BLIP Guard!
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.is_available = False
        self.client = None
        
        # Initialize the Groq client
        if not self.api_key:
            print("M2 LLM: [WARNING] GROQ_API_KEY not found in .env file.")
            print("M2 LLM: Get a free key at https://console.groq.com/keys")
            print("M2 LLM: Falling back to mock mode.")
            return
            
        try:
            self.client = Groq(api_key=self.api_key)
            # Quick test call to verify the key works
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            if test_response.choices:
                self.is_available = True
                print(f"M2 LLM: [SUCCESS] Groq Cloud LLM initialized (Model: {self.model_name})")
            else:
                print("M2 LLM: [WARNING] Groq returned empty response. Falling back to mock mode.")
        except Exception as e:
            print(f"M2 LLM: [WARNING] Failed to initialize Groq: {e}")
            print("M2 LLM: Falling back to mock mode.")

    def _call_llm(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Sends a prompt to the Groq Cloud API and returns the generated text.
        Falls back to None if the API call fails.
        """
        if not self.is_available:
            return None
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = response.choices[0].message.content.strip()
            return result if result else None
        except Exception as e:
            print(f"   [LLM API Error] {e}")
            return None

    def generate(self, article_id: str, metadata: dict, force_hallucination=False) -> str:
        """
        Generates a natural language explanation of why the item was recommended.
        Uses Groq Cloud for real generation, with mock fallback.
        """
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')
        
        if force_hallucination:
            # CAUTION: We intentionally instruct the mock LLM to lie about the color
            # so we can prove our VLM Guard catches errors mathematically!
            wrong_colors = ['neon green', 'hot pink', 'silver', 'striped magenta']
            bad_color = random.choice(wrong_colors)
            return f"I highly recommend this item! As you can see, it features a beautiful {bad_color} design."
        
        # Try Groq Cloud API first
        if self.is_available:
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
            
            result = self._call_llm(prompt)
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
        
        # Try Groq Cloud with corrective feedback
        if self.is_available:
            prompt = (
                f"Your previous fashion recommendation was REJECTED by our visual verification system "
                f"because: \"{visual_feedback}\"\n\n"
                f"Generate a corrected 1-2 sentence recommendation that ACCURATELY describes "
                f"this {color} {product_type}. Only describe features that are visually confirmed.\n\n"
                f"Respond with ONLY the corrected recommendation, nothing else."
            )
            
            result = self._call_llm(prompt)
            if result:
                return result
        
        # Fallback to mock template
        return f"Correcting my previous statement: based on verified visual evidence, this is a {color} {product_type}."

# Singleton
llm_generator = ExplanationGenerator()
