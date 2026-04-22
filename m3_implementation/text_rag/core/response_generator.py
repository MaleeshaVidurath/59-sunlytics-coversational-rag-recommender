# m3_implementation/text_rag/core/response_generator.py
#
# Generates user-friendly responses from evidence bundles using Ollama LLM.
#
# KEY DESIGN PRINCIPLES:
#   1. Evidence-grounded: LLM is explicitly told to use ONLY the evidence
#   2. Action-specific prompts: each action type has its own prompt template
#   3. Concise: responses should be clear but not lengthy
#   4. Justification-first: for recommendations, always explain WHY
#   5. Three prompt tiers: normal → strict → strictest (for regeneration)

import json
import os
import sys
import httpx
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from text_rag.config import OLLAMA_HOST, OLLAMA_RAG_MODEL

# Fashion context prepended to every prompt.
# This prevents the model from going off-topic (e.g. confusing dress names
# with food items, or discussing non-fashion topics).
FASHION_CONTEXT = (
    "You are a fashion shopping assistant for H&M clothing store. "
    "You ONLY discuss clothing, fashion items, styles, and shopping. "
    "Never discuss food, restaurants, desserts, or any non-fashion topics. "
    "Always refer to items by their full name including the word 'dress', "
    "'top', 'jacket' etc. so it is clear you are discussing clothing.\n\n"
)


# ── Prompt templates ───────────────────────────────────────────────────────────

def _build_catalog_search_prompt(evidence: dict, strictness: int = 0) -> str:
    """Prompt for recommending 2 items with justification."""
    items = evidence.get("items", [])
    prefs = evidence.get("preference_boosts", [])
    soft  = evidence.get("soft_constraints", {})
    hints = evidence.get("purchase_hints", {})
    user_msg = evidence.get("user_message", "")

    item_texts = []
    for i, item in enumerate(items, 1):
        price = item.get("price", "N/A")
        desc  = item.get("material_description", "")[:200]
        item_texts.append(
            f"Option {i}: {item.get('name','')} | "
            f"Type: {item.get('type','')} | "
            f"Colour: {item.get('colour','')} | "
            f"Price: {price} | "
            f"Pattern: {item.get('pattern','')} | "
            f"Description: {desc}"
        )

    pref_text = ""
    if prefs:
        pref_text = "User preferences: " + ", ".join(
            f"{p['attribute']}={p['value']} (weight {p['weight']:.2f})"
            for p in prefs if p.get('weight', 0) > 0.3
        )

    style_text = ""
    if soft:
        style_text = f"Style context: {', '.join(f'{k}={v}' for k,v in soft.items() if v)}"

    history_text = ""
    dom_colour = hints.get("dominant_colour") if hints else None
    dom_type   = hints.get("dominant_type")   if hints else None
    if dom_colour or dom_type:
        history_text = f"Purchase history shows this customer often buys: " \
                       f"{dom_colour or ''} {dom_type or ''} items."

    strictness_instruction = {
        0: "Write a friendly, natural recommendation.",
        1: "STRICT MODE: Only state facts present in the evidence below. Do not infer or add details.",
        2: "STRICTEST MODE: Write ONLY bullet points. Each bullet states ONLY: item name, colour, type, and price. No explanations, no reasoning, no opinions.",
    }[strictness]

    return FASHION_CONTEXT + f"""You are a fashion shopping assistant for H&M.
{strictness_instruction}

User asked: "{user_msg}"

EVIDENCE — use ONLY these facts:
{chr(10).join(item_texts)}
{pref_text}
{style_text}
{history_text}

Write a recommendation response:
- Introduce both options briefly
- For each item state: name, colour, price, and ONE sentence why it suits this user
- Keep total response under 100 words
- Do not mention internal IDs or article numbers
- Do not invent details not in the evidence above
- Do not say items "suit the user" or "match preferences" unless explicitly listed in evidence above"""


def _build_attribute_prompt(evidence: dict, strictness: int = 0) -> str:
    """Prompt for answering attribute questions."""
    article = evidence.get("article") or {}
    facts   = evidence.get("extracted_facts", {})
    topic   = evidence.get("attribute_topic", "general_details")
    user_msg= evidence.get("user_message", "")

    facts_text = "\n".join(f"  {k}: {v}" for k, v in facts.items() if v)
    if not facts_text:
        facts_text = f"  Description: {article.get('material_description','No description available')[:300]}"

    strictness_instruction = {
        0: "Answer naturally and helpfully.",
        1: "STRICT MODE: Answer ONLY from the facts listed below.",
        2: "STRICTEST MODE: Quote the relevant fact directly and add nothing else.",
    }[strictness]

    return FASHION_CONTEXT + f"""You are a fashion shopping assistant.
{strictness_instruction}

User asked: "{user_msg}"
About item: {article.get('name','')} ({article.get('colour','')} {article.get('type','')})

EVIDENCE — use ONLY these facts:
{facts_text}

Answer the question in 1-2 sentences. Do not invent information not in the facts above."""


def _build_comparison_prompt(evidence: dict, strictness: int = 0) -> str:
    """Prompt for comparing two items."""
    item_a  = evidence.get("item_a") or {}
    item_b  = evidence.get("item_b") or {}
    facts   = evidence.get("comparison_facts", {})
    dim     = evidence.get("comparison_dimension", "overall")
    user_msg= evidence.get("user_message", "")

    facts_text = "\n".join(f"  {k}: {v}" for k, v in facts.items() if v)

    strictness_instruction = {
        0: "Compare them helpfully and clearly.",
        1: "STRICT MODE: State ONLY facts present in the evidence below.",
        2: "STRICTEST MODE: Use bullet points, each citing a specific fact from evidence.",
    }[strictness]

    return FASHION_CONTEXT + f"""You are a fashion shopping assistant.
{strictness_instruction}

User asked: "{user_msg}"
Comparing two clothing items: {item_a.get('name','')} (Option 1) vs {item_b.get('name','')} (Option 2)
Dimension to compare: {dim}

EVIDENCE — use ONLY these facts:
{facts_text}

IMPORTANT: Never use the words "item_a" or "item_b" in your response.
Always refer to items by their actual names: {item_a.get('name','Option 1')} and {item_b.get('name','Option 2')}.

Write a clear comparison in 2-3 sentences. State which clothing item is better for {dim} and why, using only the facts above."""


def _build_explanation_prompt(evidence: dict, strictness: int = 0) -> str:
    """Prompt for explaining why an item was recommended."""
    article = evidence.get("article") or {}
    matches = evidence.get("confirmed_matches", [])
    prior   = evidence.get("prior_claims", [])
    user_msg= evidence.get("user_message", "")

    match_text = "\n".join(
        f"  - {m['attribute']} matches: {m['value']} (preference strength {m['weight']:.2f})"
        for m in matches
    ) or "  - No strong preference matches confirmed"

    prior_text = ""
    active_claims = [c for c in prior if c.get("status") == "active"]
    if active_claims:
        prior_text = "IMPORTANT — Already told the user:\n" + "\n".join(
            f"  - {c['claim_text']}" for c in active_claims
        ) + "\nDo NOT contradict any of the above."

    strictness_instruction = {
        0: "Explain naturally and concisely.",
        1: "STRICT MODE: Justify ONLY using the confirmed matches below.",
        2: "STRICTEST MODE: List each match as a bullet. No additional claims.",
    }[strictness]

    return FASHION_CONTEXT + f"""You are a fashion shopping assistant.
{strictness_instruction}

User asked: "{user_msg}"
Clothing item: {article.get('name','')} ({article.get('colour','')} {article.get('type','')}, {article.get('price','')})
Description: {article.get('material_description','')[:200]}

CONFIRMED preference matches (evidence):
{match_text}

{prior_text}

Explain why this item was recommended in 2-3 sentences. 
Base your explanation ONLY on the confirmed matches above."""


def _build_detail_prompt(evidence: dict, strictness: int = 0) -> str:
    """Prompt for presenting full item details."""
    article = evidence.get("article") or {}
    user_msg= evidence.get("user_message", "")

    desc = article.get("material_description", "")[:300]

    return FASHION_CONTEXT + f"""You are a fashion shopping assistant.
User asked: "{user_msg}"

Clothing item details (use ONLY these facts):
  Name:    {article.get('name','')}
  Type:    {article.get('type','')}
  Colour:  {article.get('colour','')}
  Pattern: {article.get('pattern','')}
  Price:   {article.get('price','')}
  Section: {article.get('section','')}
  Description: {desc}

Present these details in a friendly, readable format.
Do not add any information not listed above. Keep it under 80 words."""


def _build_feedback_prompt(evidence: dict) -> str:
    """Prompt for acknowledging user feedback about a clothing item."""
    feedback = evidence.get("feedback") or {}
    is_pos   = feedback.get("is_positive", True)
    item     = feedback.get("item_reacted_to") or {}
    item_name = item.get("prod_name", "the item")
    item_type = item.get("product_type_name", "clothing item")
    item_colour = item.get("colour_group_name", "")

    if is_pos:
        return (
            FASHION_CONTEXT +
            f"The user selected a {item_colour} {item_type} called '{item_name}'.\n"
            f"Write exactly 1 short sentence acknowledging their choice.\n"
            f"Then ask if they want to see more items or are done shopping.\n"
            f"RULES: Only state facts you know. Do NOT invent descriptions or praise.\n"
            f"Do NOT say it is elegant, sophisticated, stunning, or any quality you cannot verify.\n"
            f"Do NOT mention food, desserts, or anything unrelated to fashion.\n"
            f"Keep total response under 40 words."
        )
    else:
        return (
            FASHION_CONTEXT +
            f"The user does not want the {item_colour} {item_type} called '{item_name}'.\n"
            f"Write exactly 1 sentence acknowledging this and offer to search for something different.\n"
            f"Keep total response under 25 words."
        )


def _build_chitchat_prompt(evidence: dict) -> str:
    """Prompt for conversational responses."""
    refusal = evidence.get("refusal_message")
    if refusal:
        return FASHION_CONTEXT + f"Politely tell the user: {refusal} Keep it to 1 sentence about fashion."
    # Get the actual user message from evidence
    user_msg = (
        evidence.get("user_message") or
        evidence.get("memory_context", {}).get("user_message") or
        ""
    )
    msg_lower = user_msg.lower()

    # Detect farewell vs greeting based on actual message content
    farewell_words = ["thanks", "thank you", "helpful", "bye", "goodbye",
                      "cheers", "great", "awesome", "perfect", "wonderful"]
    is_farewell = any(w in msg_lower for w in farewell_words)

    if is_farewell:
        return (
            FASHION_CONTEXT +
            f"The user said: '{user_msg}'\n"
            "This is a farewell or thank-you message. "
            "Respond with a warm goodbye. Say something like: "
            "'You are welcome! Enjoy your new look!' or "
            "'Happy to help! Come back anytime for fashion advice.'\n"
            "Keep it under 20 words. Do NOT ask what they are looking for."
        )
    else:
        return (
            FASHION_CONTEXT +
            f"The user said: '{user_msg}'\n"
            "This is a greeting. Welcome them warmly and ask what "
            "clothing they are looking for today. Keep it under 25 words."
        )


# ── LLM caller ────────────────────────────────────────────────────────────────

async def call_ollama(prompt: str, max_tokens: int = 300) -> str:
    """Calls Ollama LLM and returns the generated text."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":  OLLAMA_RAG_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature":  0.3,
                        "num_predict":  max_tokens,
                        "top_p":        0.9,
                        "repeat_penalty": 1.1,
                    }
                }
            )
        if response.status_code != 200:
            return ""
        return response.json().get("response", "").strip()
    except Exception as e:
        err_str = str(e)
        if not err_str or err_str.strip() == "":
            # Empty error usually means connection timeout or Ollama busy
            print(f"[ResponseGenerator] Ollama timeout/no response — using fallback")
        elif "Connection" in err_str or "refused" in err_str:
            print(f"[ResponseGenerator] Ollama not running — run: ollama serve")
        elif "model" in err_str.lower():
            print(f"[ResponseGenerator] Ollama model error: {err_str[:100]}")
        else:
            print(f"[ResponseGenerator] Ollama error: {err_str[:150]}")
        return ""


# ── Main response generator ────────────────────────────────────────────────────

class ResponseGenerator:
    """
    Generates user-friendly responses from evidence bundles.
    Selects the appropriate prompt template for each action.
    """

    async def generate(
        self,
        evidence: dict,
        strictness: int = 0
    ) -> str:
        """
        Generates a response for the given evidence bundle.

        Args:
            evidence:   Evidence bundle from EvidenceAssembler
            strictness: 0=normal, 1=strict, 2=strictest
                        Higher strictness used in regeneration attempts

        Returns generated response text.
        """
        action = evidence.get("action", "no_retrieval")

        if action == "catalog_search":
            prompt = _build_catalog_search_prompt(evidence, strictness)
        elif action == "item_attribute_lookup":
            prompt = _build_attribute_prompt(evidence, strictness)
        elif action == "item_compare":
            prompt = _build_comparison_prompt(evidence, strictness)
        elif action == "explanation_generate":
            prompt = _build_explanation_prompt(evidence, strictness)
        elif action == "item_detail_lookup":
            prompt = _build_detail_prompt(evidence, strictness)
        elif action == "no_retrieval":
            feedback = evidence.get("feedback")
            if feedback:
                prompt = _build_feedback_prompt(evidence)
            else:
                prompt = _build_chitchat_prompt(evidence)
        else:
            prompt = _build_chitchat_prompt(evidence)

        response = await call_ollama(prompt)

        # Basic cleanup
        response = response.strip()
        if not response:
            return self._fallback_response(evidence)

        return response

    def _fallback_response(self, evidence: dict) -> str:
        """
        Returns a meaningful fallback when Ollama fails or is not running.
        Uses evidence directly to build a structured response without LLM.
        """
        action = evidence.get("action", "")

        if action == "catalog_search":
            items = evidence.get("items", [])
            if len(items) >= 2:
                a, b = items[0], items[1]
                # Build description snippet — truncate at word boundary
                desc_a = a.get('material_description', '')
                desc_a = (desc_a[:100].rsplit(' ', 1)[0] + '.') if len(desc_a) > 100 else desc_a
                return (
                    f"I found two options for you. "
                    f"Option 1: {a.get('name','')} — {a.get('colour','')} "
                    f"{a.get('type','')}, {a.get('price','')}. "
                    f"{desc_a} "
                    f"Option 2: {b.get('name','')} — {b.get('colour','')} "
                    f"{b.get('type','')}, {b.get('price','')}."
                )
            elif len(items) == 1:
                a = items[0]
                return (
                    f"I found: {a.get('name','')} — {a.get('colour','')} "
                    f"{a.get('type','')}, {a.get('price','')}."
                )
            return "I could not find items matching your criteria. Try adjusting your preferences."

        elif action == "item_attribute_lookup":
            article = evidence.get("article") or {}
            facts   = evidence.get("extracted_facts", {})
            topic   = evidence.get("attribute_topic", "")
            name    = article.get("name", "the item")
            if "detail_desc" in facts:
                return f"About {name}: {facts['detail_desc'][:200]}"
            if "avg_price" in facts:
                return f"{name} is priced at {facts['avg_price']}."
            desc = article.get("material_description", "")
            if desc:
                return f"About {name}: {desc[:200]}"
            return f"Here are the details for {name}: {article.get('colour','')} {article.get('type','')}."

        elif action == "item_compare":
            facts = evidence.get("comparison_facts", {})
            dim   = evidence.get("comparison_dimension", "overall")
            a_name = facts.get("item_a_name", "Option 1")
            b_name = facts.get("item_b_name", "Option 2")
            if dim == "price" and facts.get("cheaper_item"):
                diff = facts.get("price_difference", "")
                return (
                    f"{facts['cheaper_item']} is cheaper"
                    f"{f' by {diff}' if diff else ''}. "
                    f"{a_name} costs {facts.get('item_a_price','?')} and "
                    f"{b_name} costs {facts.get('item_b_price','?')}."
                )
            return (
                f"Comparing {a_name} and {b_name}: "
                f"{a_name} costs {facts.get('item_a_price','?')}, "
                f"{b_name} costs {facts.get('item_b_price','?')}."
            )

        elif action == "explanation_generate":
            article = evidence.get("article") or {}
            matches = evidence.get("confirmed_matches", [])
            name    = article.get("name", "this item")
            if matches:
                reasons = [
                    f"{m['attribute'].replace('_',' ')} is {m['value']}"
                    for m in matches[:2]
                ]
                return (
                    f"I recommended {name} because it matches your preference — "
                    f"{' and '.join(reasons)}. "
                    f"It is priced at {article.get('price','?')}."
                )
            return f"I recommended {name} based on your style preferences."

        elif action == "item_detail_lookup":
            article = evidence.get("article") or {}
            name    = article.get("name", "the item")
            desc    = article.get("material_description", "")[:200]
            return (
                f"{name}: {article.get('colour','')} {article.get('type','')}, "
                f"{article.get('price','?')}. {desc}"
            )

        elif action == "no_retrieval":
            feedback = evidence.get("feedback") or {}
            if feedback.get("is_positive"):
                item = feedback.get("item_reacted_to") or {}
                name = item.get("prod_name", "your choice")
                return f"Great choice! {name} is a wonderful pick. Would you like to explore more options?"
            elif feedback.get("is_positive") is False:
                return "I understand — let me find something different for you. What would you prefer?"
            return "You are welcome! Feel free to ask if you need anything else."

        return "How can I help you with your fashion choices today?"
