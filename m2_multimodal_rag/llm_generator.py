import base64
import os
import random
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HALLUCINATION_COLORS = ["neon green", "hot pink", "silver", "striped magenta"]
_RERANK_POOL_SIZE = 8
_QUERY_VARIANTS = 3

# Reasoning models (Qwen3.x, and others Groq may substitute in later) prefix their
# answer with a <think>...</think> block.  reasoning_effort="none" normally
# suppresses it; this strips any that still leaks through so a model swap can
# never silently poison an explanation with the model's internal monologue.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str | None) -> str | None:
    """Removes <think> blocks, including an unclosed trailing one."""
    if not text:
        return text
    cleaned = _THINK_RE.sub("", text)
    # Truncated output can leave an unterminated <think> with no closing tag.
    if "<think>" in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().index("<think>")]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# ExplanationGenerator
# ---------------------------------------------------------------------------
class ExplanationGenerator:
    """
    Cloud LLM wrapper for M2 Multimodal RAG using the Groq API (free tier).
    Runs Llama 3.x in the cloud — no local RAM/GPU required.

    Public surface:
      generate()         — recommendation explanation (novelty 4)
      regenerate()       — hallucination-corrected re-explanation
      expand_query()     — semantic query variants for CLIP retrieval (novelty 1)
      rerank_candidates()— LLM cross-encoder re-ranking (novelty 2)
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        self.model_name        = os.getenv("GROQ_MODEL",        "llama-3.1-8b-instant")
        self.vision_model_name = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.is_available      = False
        self.client            = None
        # Real API failure counter (evaluation harnesses use this to detect
        # quota exhaustion directly, instead of inferring it from guard
        # layers' built-in "pass on failure" fallbacks — which look
        # identical to a genuine pass in their output).
        self.fail_count        = 0
        self._init_client()

    def _init_client(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            print("M2 LLM: [WARNING] GROQ_API_KEY not found — falling back to mock mode.")
            print("M2 LLM: Get a free key at https://console.groq.com/keys")
            return

        try:
            client = Groq(api_key=api_key)
            test   = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            if not test.choices:
                print("M2 LLM: [WARNING] Groq returned empty response — falling back to mock mode.")
                return

            self.client       = client
            self.is_available = True
            print(f"M2 LLM: [SUCCESS] Text model : {self.model_name}")
            print(f"M2 LLM: [SUCCESS] Vision model: {self.vision_model_name}")

        except Exception as exc:
            print(f"M2 LLM: [WARNING] Groq init failed ({exc}) — falling back to mock mode.")

    # ------------------------------------------------------------------
    # Private: low-level API helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 150,
                  temperature: float = 0.7) -> str | None:
        if not self.is_available:
            self.fail_count += 1
            return None
        try:
            resp   = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = resp.choices[0].message.content.strip()
            if not result:
                self.fail_count += 1
            return result or None
        except Exception as exc:
            print(f"   [LLM API Error] {exc}")
            self.fail_count += 1
            return None

    def _call_vision_llm(self, prompt: str, image_path: str,
                         max_tokens: int = 150,
                         temperature: float = 0.7) -> str | None:
        if not self.is_available:
            return None

        b64 = self._encode_image_base64(image_path)
        if not b64:
            print("   [Vision] Encoding failed — falling back to text-only LLM.")
            return self._call_llm(prompt, max_tokens, temperature)

        try:
            resp   = self.client.chat.completions.create(
                model=self.vision_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=temperature,
                # Qwen3.6 is a reasoning model: without this it spends the whole
                # token budget on a <think> block and returns nothing usable.
                reasoning_effort="none",
            )
            result = resp.choices[0].message.content.strip()
            result = _strip_reasoning(result)
            print("   [Vision] Image-grounded generation succeeded.")
            return result or None
        except Exception as exc:
            print(f"   [Vision LLM API Error] {exc} — falling back to text-only LLM.")
            return self._call_llm(prompt, max_tokens, temperature)

    @staticmethod
    def _encode_image_base64(image_path: str) -> str | None:
        try:
            with open(image_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("utf-8")
        except Exception as exc:
            print(f"   [Vision] Could not encode {image_path}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private: metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_metadata(metadata: dict) -> dict:
        return {
            "color":        metadata.get("colour_group_name", "Black"),
            "product_type": metadata.get("product_type_name", "Garment"),
            "department":   metadata.get("department_name",   "Fashion"),
            "category":     metadata.get("product_group_name","Clothing"),
            "description":  metadata.get("detail_desc",       ""),
            "kb_fact":      metadata.get("kb_psychology_fact",""),
        }

    # ------------------------------------------------------------------
    # Private: prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_generate_prompt(m: dict, product_knowledge: str,
                               with_image: bool) -> str:
        kb_line = f"\n- Psychology insight: {m['kb_fact']}" if m["kb_fact"] else ""
        knowledge_block = (
            f"\n\nVerified product knowledge (stay factually grounded):\n{product_knowledge}"
            if product_knowledge else ""
        )
        base_metadata = (
            f"- Product Type: {m['product_type']}\n"
            f"- Color: {m['color']}\n"
            f"- Department: {m['department']}\n"
            f"- Category: {m['category']}\n"
            f"- Description: {m['description']}"
            f"{kb_line}"
            f"{knowledge_block}"
        )
        closing = (
            "Respond with ONLY the recommendation explanation, nothing else. "
            "Do not start with 'I recommend' — be creative and natural."
        )

        if with_image:
            return (
                "You are a friendly fashion recommendation assistant. "
                "The image above shows the actual product being recommended. "
                "Generate a warm, conversational 1-2 sentence explanation grounded in "
                "what you can SEE in the image (colour, texture, pattern, fit, styling).\n\n"
                f"Catalog metadata (for context only):\n{base_metadata}\n\n"
                f"{closing} Prioritise what you observe in the image over the metadata."
            )

        return (
            "You are a friendly fashion recommendation assistant. "
            "Generate a warm, conversational 1-2 sentence explanation of why "
            "this item is a great recommendation for the customer.\n\n"
            f"Item details:\n{base_metadata}\n\n"
            f"{closing} Weave any psychology insight naturally into your explanation. "
            "Only describe features supported by the verified product knowledge."
        )

    @staticmethod
    def _build_regenerate_prompt(m: dict, visual_feedback: str,
                                 with_image: bool) -> str:
        closing = "Respond with ONLY the corrected recommendation, nothing else."
        if with_image:
            return (
                f"The image above shows the actual product. "
                f"Your previous recommendation was REJECTED because: \"{visual_feedback}\"\n\n"
                f"Look at the image carefully and write a corrected 1-2 sentence recommendation "
                f"that ACCURATELY reflects what you see. "
                f"Catalog metadata for reference: {m['color']} {m['product_type']}.\n\n"
                f"{closing}"
            )
        return (
            f"Your previous fashion recommendation was REJECTED by our visual verification "
            f"system because: \"{visual_feedback}\"\n\n"
            f"Write a corrected 1-2 sentence recommendation that ACCURATELY describes "
            f"this {m['color']} {m['product_type']}. "
            f"Only describe visually confirmed features.\n\n"
            f"{closing}"
        )

    @staticmethod
    def _build_rerank_context(user_message: str, soft_constraints: dict,
                              purchase_hints: dict) -> str:
        parts = [f"Customer query: '{user_message}'"]

        if soft_constraints:
            style_parts = [f"{k}: {v}" for k, v in soft_constraints.items() if v]
            if style_parts:
                parts.append(f"Style preference: {', '.join(style_parts)}")

        if purchase_hints:
            dc = purchase_hints.get("dominant_colour", "")
            dt = purchase_hints.get("dominant_type",   "")
            bt = purchase_hints.get("budget_tier",     "")
            if dc or dt:
                parts.append(f"Typically buys: {dc} {dt}".strip())
            if bt:
                parts.append(f"Budget tier: {bt}")

        return "\n".join(parts)

    @staticmethod
    def _build_rerank_prompt(context: str, pool: list) -> str:
        item_lines = [
            f"{i}. {c.get('metadata', {}).get('prod_name', '?')} | "
            f"Colour: {c.get('metadata', {}).get('colour_group_name', '?')} | "
            f"Type: {c.get('metadata', {}).get('product_type_name', '?')} | "
            f"Dept: {c.get('metadata', {}).get('department_name', '?')}"
            for i, c in enumerate(pool, 1)
        ]
        return (
            "You are a fashion recommendation expert. Rank these items for the customer.\n\n"
            f"Customer context:\n{context}\n\n"
            f"Candidates:\n" + "\n".join(item_lines) + "\n\n"
            "Output ONLY a comma-separated list of item numbers ranked best to worst.\n"
            "Example output: 3,1,5,2,4"
        )

    # ------------------------------------------------------------------
    # Public: explanation generation (Novelty 4)
    # ------------------------------------------------------------------

    def generate(self, article_id: str, metadata: dict,
                 force_hallucination: bool = False,
                 product_knowledge: str = "",
                 image_path: str = "") -> str:
        m = self._parse_metadata(metadata)

        if force_hallucination:
            bad_color = random.choice(_HALLUCINATION_COLORS)
            return f"I highly recommend this item! As you can see, it features a beautiful {bad_color} design."

        if self.is_available:
            prompt = self._build_generate_prompt(m, product_knowledge, bool(image_path))
            if image_path:
                print(f"   [Vision] Generating image-grounded explanation for {article_id}...")
                result = self._call_vision_llm(prompt, image_path)
            else:
                result = self._call_llm(prompt)

            if result:
                return result

        return f"I recommend this item because it is a stylish {m['color']} {m['product_type']} that matches your search."

    def regenerate(self, article_id: str, metadata: dict,
                   visual_feedback: str, image_path: str = "") -> str:
        m = self._parse_metadata(metadata)
        print("\n[LLM INTERNAL] Received strict feedback from Visual Guard. Regenerating response...")

        if self.is_available:
            prompt = self._build_regenerate_prompt(m, visual_feedback, bool(image_path))
            if image_path:
                print(f"   [Vision] Regenerating with image context for {article_id}...")
                result = self._call_vision_llm(prompt, image_path)
            else:
                result = self._call_llm(prompt)

            if result:
                return result

        return f"Correcting my previous statement: based on verified visual evidence, this is a {m['color']} {m['product_type']}."

    # ------------------------------------------------------------------
    # Public: LLM query expansion (Novelty 1)
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> list:
        """
        Returns the original query plus up to 3 LLM-generated semantic variants
        for multi-vector CLIP retrieval (RAG-VisualRec novelty 1).
        """
        if not self.is_available or not query:
            return [query]

        prompt = (
            f"You are a fashion search expert. Generate exactly {_QUERY_VARIANTS} alternative "
            f"search phrases for the same fashion item described below.\n"
            f"Original: '{query}'\n"
            f"Rules:\n"
            f"- Each phrase must describe the same item from a different vocabulary angle\n"
            f"- Keep each phrase under 8 words\n"
            f"- Use varied fashion terminology (fabric, occasion, style, silhouette)\n"
            f"Output ONLY the {_QUERY_VARIANTS} phrases, one per line, no numbers, no explanation."
        )

        result = self._call_llm(prompt, max_tokens=80, temperature=0.3)
        if not result:
            return [query]

        variants = [ln.strip() for ln in result.strip().splitlines() if ln.strip()][:_QUERY_VARIANTS]
        print(f"   [Query Expansion] '{query}' → {len(variants)} variants: {variants}")
        return [query] + variants

    # ------------------------------------------------------------------
    # Public: LLM cross-encoder re-ranking (Novelty 2)
    # ------------------------------------------------------------------

    def rerank_candidates(self, user_message: str, candidates: list,
                          soft_constraints: dict = None,
                          purchase_hints: dict = None) -> list:
        """
        Scores each candidate against the full user context (query + style prefs +
        purchase history) and returns candidates in ranked order
        (RAG-VisualRec novelty 2).
        """
        if not self.is_available or len(candidates) <= 2:
            return candidates

        pool    = candidates[:_RERANK_POOL_SIZE]
        context = self._build_rerank_context(user_message, soft_constraints or {}, purchase_hints or {})
        prompt  = self._build_rerank_prompt(context, pool)

        result = self._call_llm(prompt, max_tokens=25, temperature=0.0)
        if not result:
            return candidates

        try:
            ranked_idx = [
                int(x.strip()) - 1
                for x in result.strip().split(",")
                if x.strip().isdigit()
            ]
            ranked_idx = [i for i in ranked_idx if 0 <= i < len(pool)]
            seen       = set(ranked_idx)
            reranked   = [pool[i] for i in ranked_idx]
            reranked  += [pool[i] for i in range(len(pool)) if i not in seen]
            reranked  += candidates[_RERANK_POOL_SIZE:]
            print(f"   [LLM Re-rank] Cross-encoder reordered {len(pool)} candidates")
            return reranked
        except Exception as exc:
            print(f"   [LLM Re-rank] Parse error: {exc}. Keeping original order.")
            return candidates

    # generate_product_knowledge() and self_evaluate() live in:
    # hallucination_guard/layer_1_knowledge_self_reflection.py


# Singleton used across the m2 pipeline
llm_generator = ExplanationGenerator()
