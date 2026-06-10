import base64
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
        self.vision_model_name = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
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
                print(f"M2 LLM: [SUCCESS] Vision model configured: {self.vision_model_name}")
            else:
                print("M2 LLM: [WARNING] Groq returned empty response. Falling back to mock mode.")
        except Exception as e:
            print(f"M2 LLM: [WARNING] Failed to initialize Groq: {e}")
            print("M2 LLM: Falling back to mock mode.")

    def _call_llm(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str | None:
        """
        Sends a prompt to the Groq Cloud API and returns the generated text.
        Returns None if unavailable or the API call fails.
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

    def _encode_image_base64(self, image_path: str) -> str | None:
        """Reads an image file and returns its base64-encoded string, or None on failure."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"   [Vision] Could not encode image {image_path}: {e}")
            return None

    def _call_vision_llm(self, prompt: str, image_path: str,
                         max_tokens: int = 150, temperature: float = 0.7) -> str | None:
        """
        Sends a prompt + product image to the vision-capable Groq model.
        Falls back to text-only _call_llm if encoding or API call fails.
        """
        if not self.is_available:
            return None

        b64 = self._encode_image_base64(image_path)
        if not b64:
            print("   [Vision] Image encoding failed — falling back to text-only LLM.")
            return self._call_llm(prompt, max_tokens, temperature)

        try:
            response = self.client.chat.completions.create(
                model=self.vision_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = response.choices[0].message.content.strip()
            print(f"   [Vision] Image-grounded generation succeeded.")
            return result if result else None
        except Exception as e:
            print(f"   [Vision LLM API Error] {e} — falling back to text-only LLM.")
            return self._call_llm(prompt, max_tokens, temperature)

    def generate(self, article_id: str, metadata: dict, force_hallucination: bool = False,
                 product_knowledge: str = "", image_path: str = "") -> str:
        """
        Generates a natural language explanation of why the item was recommended.
        When image_path is provided the vision LLM sees the actual product image,
        making the explanation grounded in visual evidence — not just metadata text.
        Falls back to text-only generation if the vision call fails.
        """
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')

        if force_hallucination:
            wrong_colors = ['neon green', 'hot pink', 'silver', 'striped magenta']
            bad_color = random.choice(wrong_colors)
            return f"I highly recommend this item! As you can see, it features a beautiful {bad_color} design."

        if self.is_available:
            department = metadata.get('department_name', 'Fashion')
            category = metadata.get('product_group_name', 'Clothing')
            detail_desc = metadata.get('detail_desc', '')
            kb_fact = metadata.get('kb_psychology_fact', '')
            kb_instruction = (
                f"\n- Psychology insight: {kb_fact}" if kb_fact else ""
            )
            knowledge_instruction = (
                f"\n\nVerified product knowledge (use this to stay factually grounded):\n{product_knowledge}"
                if product_knowledge else ""
            )

            if image_path:
                # Vision-grounded prompt: LLM sees the actual product image
                prompt = (
                    f"You are a friendly fashion recommendation assistant. "
                    f"The image above shows the actual product being recommended. "
                    f"Generate a warm, conversational 1-2 sentence explanation grounded in "
                    f"what you can SEE in the image. Describe visible details like colour, "
                    f"texture, pattern, fit, or styling that make this item appealing.\n\n"
                    f"Catalog metadata (for context only):\n"
                    f"- Product Type: {product_type}\n"
                    f"- Color: {color}\n"
                    f"- Department: {department}\n"
                    f"- Category: {category}\n"
                    f"- Description: {detail_desc}"
                    f"{kb_instruction}"
                    f"{knowledge_instruction}\n\n"
                    f"Respond with ONLY the recommendation explanation, nothing else. "
                    f"Do not start with 'I recommend' — be creative and natural. "
                    f"Prioritise what you observe in the image over the metadata."
                )
                print(f"   [Vision] Generating image-grounded explanation for {article_id}...")
                result = self._call_vision_llm(prompt, image_path)
            else:
                # Text-only fallback (no image available)
                prompt = (
                    f"You are a friendly fashion recommendation assistant. "
                    f"Generate a warm, conversational 1-2 sentence explanation of why "
                    f"this item is a great recommendation for the customer.\n\n"
                    f"Item details:\n"
                    f"- Product Type: {product_type}\n"
                    f"- Color: {color}\n"
                    f"- Department: {department}\n"
                    f"- Category: {category}\n"
                    f"- Description: {detail_desc}"
                    f"{kb_instruction}"
                    f"{knowledge_instruction}\n\n"
                    f"Respond with ONLY the recommendation explanation, nothing else. "
                    f"Do not start with 'I recommend' — be creative and natural. "
                    f"If a psychology insight is provided, weave it naturally into your explanation. "
                    "Only describe features that are supported by the verified product knowledge."
                )
                result = self._call_llm(prompt)

            if result:
                return result

        # Fallback to mock template if API unavailable
        return f"I recommend this item because it is a stylish {color} {product_type} that matches your search."
        
    def regenerate(self, article_id: str, metadata: dict, visual_feedback: str,
                   image_path: str = "") -> str:
        """
        Triggered when a guard layer rejects the previous explanation.
        When image_path is provided the vision LLM sees the product image directly,
        so corrections are grounded in visual evidence rather than feedback text alone.
        """
        color = metadata.get('colour_group_name', 'Black')
        product_type = metadata.get('product_type_name', 'Garment')

        print("\n[LLM INTERNAL] Received strict feedback from Visual Guard. Regenerating response...")

        if self.is_available:
            if image_path:
                prompt = (
                    f"The image above shows the actual product. Your previous fashion recommendation "
                    f"was REJECTED because: \"{visual_feedback}\"\n\n"
                    f"Look at the image carefully and generate a corrected 1-2 sentence recommendation "
                    f"that ACCURATELY reflects what you see. "
                    f"Catalog metadata for reference: {color} {product_type}.\n\n"
                    f"Respond with ONLY the corrected recommendation, nothing else."
                )
                print(f"   [Vision] Regenerating with image context for {article_id}...")
                result = self._call_vision_llm(prompt, image_path)
            else:
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

        return f"Correcting my previous statement: based on verified visual evidence, this is a {color} {product_type}."

    # ------------------------------------------------------------------
    # NOVELTY 1: LLM Query Expansion
    # ------------------------------------------------------------------
    def expand_query(self, query: str) -> list:
        """
        Generates 3 semantic variants of the search query for multi-vector CLIP retrieval.
        Returns a list containing the original query plus up to 3 LLM-generated variants.
        Paper: RAG-VisualRec — enriching sparse signals into richer textual representations.
        """
        if not self.is_available or not query:
            return [query]

        prompt = (
            f"You are a fashion search expert. Generate exactly 3 alternative search phrases "
            f"for the same fashion item described below.\n"
            f"Original: '{query}'\n"
            f"Rules:\n"
            f"- Each phrase must describe the same item from a different vocabulary angle\n"
            f"- Keep each phrase under 8 words\n"
            f"- Use varied fashion terminology (fabric, occasion, style, silhouette)\n"
            f"Output ONLY the 3 phrases, one per line, no numbers, no explanation."
        )

        result = self._call_llm(prompt, max_tokens=80, temperature=0.3)
        if not result:
            return [query]

        variants = [line.strip() for line in result.strip().split('\n') if line.strip()][:3]
        print(f"   [Query Expansion] '{query}' → {len(variants)} variants: {variants}")
        return [query] + variants

    # ------------------------------------------------------------------
    # NOVELTY 2: LLM Cross-Encoder Re-ranking
    # ------------------------------------------------------------------
    def rerank_candidates(self, user_message: str, candidates: list,
                          soft_constraints: dict = None, purchase_hints: dict = None) -> list:
        """
        Two-stage re-ranking: LLM acts as a cross-encoder to score each candidate
        against the full user context (query + style preferences + purchase history).
        Paper: RAG-VisualRec — LLM-based re-ranking improves nDCG.
        """
        if not self.is_available or len(candidates) <= 2:
            return candidates

        pool = candidates[:8]

        # Build context string
        ctx_parts = [f"Customer query: '{user_message}'"]
        if soft_constraints:
            style_parts = [f"{k}: {v}" for k, v in soft_constraints.items() if v]
            if style_parts:
                ctx_parts.append(f"Style preference: {', '.join(style_parts)}")
        if purchase_hints:
            dc = purchase_hints.get('dominant_colour')
            dt = purchase_hints.get('dominant_type')
            bt = purchase_hints.get('budget_tier')
            if dc or dt:
                ctx_parts.append(f"Typically buys: {(dc or '')} {(dt or '')}".strip())
            if bt:
                ctx_parts.append(f"Budget tier: {bt}")

        context = "\n".join(ctx_parts)

        item_lines = []
        for i, c in enumerate(pool, 1):
            m = c.get('metadata', {})
            item_lines.append(
                f"{i}. {m.get('prod_name', '?')} | "
                f"Colour: {m.get('colour_group_name', '?')} | "
                f"Type: {m.get('product_type_name', '?')} | "
                f"Dept: {m.get('department_name', '?')}"
            )

        prompt = (
            f"You are a fashion recommendation expert. Rank these items for the customer.\n\n"
            f"Customer context:\n{context}\n\n"
            f"Candidates:\n" + "\n".join(item_lines) + "\n\n"
            f"Output ONLY a comma-separated list of item numbers ranked best to worst.\n"
            f"Example output: 3,1,5,2,4"
        )

        result = self._call_llm(prompt, max_tokens=25, temperature=0.0)
        if not result:
            return candidates

        try:
            ranked_idx = [
                int(x.strip()) - 1
                for x in result.strip().split(',')
                if x.strip().isdigit()
            ]
            ranked_idx = [i for i in ranked_idx if 0 <= i < len(pool)]
            seen = set(ranked_idx)
            reranked = [pool[i] for i in ranked_idx]
            reranked += [pool[i] for i in range(len(pool)) if i not in seen]
            reranked += candidates[8:]
            print(f"   [LLM Re-rank] Cross-encoder reordered {len(pool)} candidates")
            return reranked
        except Exception as e:
            print(f"   [LLM Re-rank] Parse error: {e}. Keeping original order.")
            return candidates

    # generate_product_knowledge() and self_evaluate() live in:
    # hallucination_guard/layer_1_knowledge_self_reflection.py


# Singleton
llm_generator = ExplanationGenerator()
