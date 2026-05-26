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

    def generate(self, article_id: str, metadata: dict, force_hallucination=False,
                 product_knowledge: str = "") -> str:
        """
        Generates a natural language explanation of why the item was recommended.
        Uses Groq Cloud for real generation, with mock fallback.
        If product_knowledge is provided (from the EMNLP 2023 knowledge loop),
        it is injected as a grounding context so the LLM stays factually anchored.
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
            # NOVELTY 5: psychology-grounded fact from Fashion KB
            kb_fact = metadata.get('kb_psychology_fact', '')
            kb_instruction = (
                f"\n- Psychology insight: {kb_fact}"
                if kb_fact else ""
            )
            # EMNLP 2023: inject verified product knowledge as grounding context
            knowledge_instruction = (
                f"\n\nVerified product knowledge (use this to stay factually grounded):\n{product_knowledge}"
                if product_knowledge else ""
            )

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
                f"Only describe features that are supported by the verified product knowledge."
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

    # ------------------------------------------------------------------
    # EMNLP 2023 Loop 1: Factual Knowledge Acquisition
    # Paper: Ji et al. (2023), "Towards Mitigating LLM Hallucination via
    # Self Reflection", EMNLP 2023 Findings (2023.findings-emnlp.123).
    # Generate-score-refine loop to produce verified product knowledge
    # before explanation generation — grounding the LLM in facts first.
    # ------------------------------------------------------------------
    def generate_product_knowledge(self, metadata: dict) -> str:
        """
        Loop 1 of the EMNLP 2023 self-reflection method: generate factual
        background knowledge about the product, score it for factual
        consistency with the metadata, and refine once if the score is low.

        Returns a verified knowledge string used to ground explanation generation.
        """
        if not self.is_available:
            return ""

        colour = metadata.get('colour_group_name', '')
        product_type = metadata.get('product_type_name', '')
        department = metadata.get('department_name', '')
        appearance = metadata.get('graphical_appearance_name', '')
        detail_desc = str(metadata.get('detail_desc', ''))[:200]

        meta_facts = (
            f"Colour: {colour} | Type: {product_type} | "
            f"Department: {department} | Appearance: {appearance} | "
            f"Description: {detail_desc}"
        )

        def _generate_knowledge(facts: str) -> str:
            prompt = (
                f"You are a fashion expert. Based on the product metadata below, "
                f"generate concise factual background knowledge about this item "
                f"(2-3 sentences). Consider: verifiability, objectivity, and "
                f"accuracy to the given attributes.\n\n"
                f"Product metadata:\n{facts}\n\n"
                f"Output ONLY the factual knowledge. No recommendations yet."
            )
            return self._call_llm(prompt, max_tokens=100, temperature=0.1) or ""

        def _score_knowledge(knowledge: str, facts: str) -> int:
            prompt = (
                f"Score this product knowledge for factual consistency with "
                f"the verified metadata (1-10). Deduct points for any claim "
                f"not supported by the metadata.\n\n"
                f"Metadata: {facts}\n"
                f"Knowledge: \"{knowledge}\"\n\n"
                f"Output ONLY: SCORE: <number>"
            )
            result = self._call_llm(prompt, max_tokens=10, temperature=0.0)
            if not result:
                return 7
            try:
                line = next((l for l in result.split('\n') if 'SCORE' in l.upper()), "SCORE: 7")
                return int(line.split(':', 1)[1].strip())
            except Exception:
                return 7

        knowledge = _generate_knowledge(meta_facts)
        if not knowledge:
            return ""

        score = _score_knowledge(knowledge, meta_facts)
        print(f"   [Knowledge Loop] Factuality score: {score}/10")

        if score < 6:
            print(f"   [Knowledge Loop] Score below threshold — refining knowledge...")
            refine_prompt = (
                f"The following product knowledge scored {score}/10 for factual "
                f"consistency with the metadata. Rewrite it to be strictly accurate.\n\n"
                f"Metadata: {meta_facts}\n"
                f"Original knowledge: \"{knowledge}\"\n\n"
                f"Output ONLY the corrected knowledge. No extra text."
            )
            refined = self._call_llm(refine_prompt, max_tokens=100, temperature=0.0)
            if refined:
                knowledge = refined
                print(f"   [Knowledge Loop] Knowledge refined successfully.")

        return knowledge

    # ------------------------------------------------------------------
    # NOVELTY 4 (upgraded): Knowledge-Grounded Self-Reflection Gate
    # Paper: Ji et al. (2023) EMNLP — Loop 2: Knowledge-Consistent
    # Answering. Upgraded from simple 1-10 scoring to knowledge-grounded
    # consistency check: the LLM now compares the explanation against the
    # verified product knowledge produced in Loop 1, not just raw metadata.
    # ------------------------------------------------------------------
    def self_evaluate(self, explanation: str, metadata: dict, product_knowledge: str = "") -> tuple:
        """
        Knowledge-grounded self-reflection (EMNLP 2023, Loop 2).
        Scores the explanation against verified product knowledge (if available)
        OR against raw metadata facts — whichever provides stronger grounding.
        Returns (passes: bool, feedback: str).
        """
        if not self.is_available:
            return True, "Self-evaluation skipped (LLM unavailable)."

        colour = metadata.get('colour_group_name', '')
        product_type = metadata.get('product_type_name', '')

        grounding = (
            f"Verified product knowledge:\n{product_knowledge}"
            if product_knowledge
            else f"Verified item facts: {colour} {product_type}"
        )

        prompt = (
            f"Evaluate this fashion recommendation explanation for quality.\n\n"
            f"{grounding}\n"
            f"Explanation to evaluate: \"{explanation}\"\n\n"
            f"Score 1-10 based on: factual consistency with verified facts, "
            f"clarity, and helpfulness. Be strict — score below 6 if the "
            f"explanation contradicts or ignores the verified product knowledge.\n"
            f"Output format (two lines only):\n"
            f"SCORE: <number>\n"
            f"FEEDBACK: <one sentence>"
        )

        result = self._call_llm(prompt, max_tokens=60, temperature=0.0)
        if not result:
            return True, "Self-evaluation inconclusive. Passing."

        try:
            lines = result.strip().split('\n')
            score_line = next((l for l in lines if l.upper().startswith('SCORE:')), None)
            feedback_line = next((l for l in lines if l.upper().startswith('FEEDBACK:')), None)

            score = int(score_line.split(':', 1)[1].strip()) if score_line else 7
            feedback = feedback_line.split(':', 1)[1].strip() if feedback_line else "Quality acceptable."

            passes = score >= 6
            print(f"   [Self-Reflect] Score: {score}/10 — {'PASS' if passes else 'FAIL → proactive regeneration'}")
            return passes, feedback
        except Exception:
            return True, "Self-evaluation parse error. Passing."


# Singleton
llm_generator = ExplanationGenerator()
