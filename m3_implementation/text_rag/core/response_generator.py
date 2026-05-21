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
from text_rag.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL,
)
import httpx
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Groq+Ollama imports
from text_rag.config import OLLAMA_HOST, OLLAMA_RAG_MODEL

def _clean_response(text: str) -> str:
    """
    Cleans LLM response text before sending to user.
    Removes markdown artifacts and normalises formatting.
    """
    import re
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Replace em-dash with colon
    text = text.replace(' — ', ': ').replace('—', ', ')
    # Remove double spaces
    text = re.sub(r'  +', ' ', text)
    # Remove trailing/leading whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    return text.strip()


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
- Do not say items "suit the user" or "match preferences" unless explicitly listed in evidence above
- Format each item clearly on its own line starting with Option number
- Use plain text only, no markdown, no asterisks, no dashes"""


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
    """Prompt for comparing two or more items."""
    item_a    = evidence.get("item_a") or {}
    item_b    = evidence.get("item_b") or {}
    facts     = evidence.get("comparison_facts", {})
    dim       = evidence.get("comparison_dimension", "overall")
    user_msg  = evidence.get("user_message", "")
    items_all = evidence.get("items_all")

    strictness_instruction = {
        0: "Compare them helpfully and clearly.",
        1: "STRICT MODE: State ONLY facts present in the evidence below.",
        2: "STRICTEST MODE: Use bullet points, each citing a specific fact from evidence.",
    }[strictness]

    # General multi-item comparison (non-price, >2 items — generic or named group)
    if items_all and len(items_all) > 2 and dim != "price":
        item_lines = []
        for i, a in enumerate(items_all, 1):
            name   = a.get("prod_name") or a.get("name", f"Option {i}")
            colour = a.get("colour_group_name") or a.get("colour", "")
            price  = a.get("avg_price")
            price_str = f"£{float(price):.2f}" if price is not None else ""
            desc   = (a.get("detail_desc") or "")[:100]
            line   = f"  Option {i}: {name} | {colour}" + (f" | {price_str}" if price_str else "")
            if desc:
                line += f" | {desc}"
            item_lines.append(line)
        items_text = "\n".join(item_lines)
        return (
            FASHION_CONTEXT
            + f'\nUser asked: "{user_msg}"\n\n'
            + f"Compare all {len(items_all)} recommended items ({dim}):\n"
            + items_text
            + f"\n\n{strictness_instruction}\n"
            + "Give a helpful overall comparison and your top recommendation. 3-4 sentences."
        )

    # Multi-item price comparison — list all prices, identify cheapest
    if items_all and len(items_all) > 2 and dim == "price":
        ranked = facts.get("all_items_ranked", [])
        if ranked:
            price_lines = "\n".join(
                f"  {i+1}. {r['name']} ({r['colour']}) — {r['price']}"
                for i, r in enumerate(ranked)
            )
            cheapest = ranked[0]
            return (
                FASHION_CONTEXT
                + f'\nUser asked: "{user_msg}"\n\n'
                + f"All {len(ranked)} recommended items from cheapest to most expensive:\n"
                + price_lines
                + f"\n\n{strictness_instruction}\n"
                + f"State that {cheapest['name']} ({cheapest['colour']}) at {cheapest['price']} is the cheapest, "
                + "then briefly list all other prices. Write 2-3 sentences."
            )

    facts_text = "\n".join(f"  {k}: {v}" for k, v in facts.items() if v)

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


def _build_explanation_all_prompt(evidence: dict, strictness: int = 0) -> str:
    """Prompt when user asks 'why' with no specific product name — summarise all recommended items."""
    articles  = evidence.get("articles", [])
    all_prefs = evidence.get("matched_prefs", [])
    user_msg  = evidence.get("user_message", "")

    item_lines = []
    for i, a in enumerate(articles, 1):
        item_lines.append(
            f"  Option {i}: {a.get('name','')} | "
            f"{a.get('colour','')} {a.get('type','')} | {a.get('price','')}"
        )

    top_prefs = sorted(all_prefs, key=lambda x: x.get("weight", 0), reverse=True)[:3]
    pref_text = ", ".join(
        f"{p.get('attribute_value','')} {p.get('attribute_name','').replace('_group_name','').replace('_name','').replace('_','  ')}"
        for p in top_prefs if p.get("attribute_value")
    ) or "general fashion preferences"

    strictness_instruction = {
        0: "Give a natural, friendly group explanation in 2-3 sentences.",
        1: "STRICT MODE: Only reference items and preferences listed below.",
        2: "STRICTEST MODE: One sentence per item. No additional claims.",
    }.get(strictness, "Give a natural, friendly group explanation in 2-3 sentences.")

    return FASHION_CONTEXT + f"""You are a fashion shopping assistant.
{strictness_instruction}

USER QUESTION: "{user_msg}"

RECOMMENDED ITEMS:
{chr(10).join(item_lines)}

USER PREFERENCES: {pref_text}

TASK: Explain briefly why these items were recommended as a group.
- Mention how the overall selection matches the user's preferences
- You may reference individual items by name
- Keep total response under 100 words
- Do not invent details not listed above"""


def _build_explanation_prompt(evidence: dict, strictness: int = 0) -> str:
    """
    Builds a rich explanation prompt combining:
    - Item details (name, colour, type, price, description)
    - Confirmed preference matches (verified against article)
    - All user preferences ranked by weight
    - Prior claims (must not contradict)
    """
    # All-items summary path (user asked "why" with no specific product name)
    if evidence.get("articles"):
        return _build_explanation_all_prompt(evidence, strictness)

    article       = evidence.get("article") or {}
    matches       = evidence.get("confirmed_matches", [])
    all_prefs     = evidence.get("matched_prefs", [])
    prior         = evidence.get("prior_claims", [])
    user_msg      = evidence.get("user_message", "")

    item_name   = article.get("name", "this item")
    item_colour = article.get("colour", "")
    item_type   = article.get("type", "")
    item_price  = article.get("price", "")
    item_desc   = article.get("material_description", "")[:250]
    item_pattern= article.get("pattern", "")
    item_group  = article.get("garment_group", "")

    _ATTR_MAP = {
        "colour_group_name":         "colour",
        "product_type_name":         "product type",
        "graphical_appearance_name": "pattern",
        "garment_group_name":        "style category",
        "index_group_name":          "category",
        "avg_price":                 "price range",
        "occasion":                  "occasion",
        "style":                     "style",
    }

    def _human(attr):
        return _ATTR_MAP.get(attr, attr.replace("_", " ").replace(" name", "").strip())

    def _strength(w):
        if w >= 0.7: return "strongly"
        if w >= 0.4: return "moderately"
        return "slightly"

    pref_lines = []

    # Confirmed matches: article attribute literally equals preference value
    for m in matches[:4]:
        attr = _human(m.get("attribute", ""))
        val  = m.get("value", "")
        wt   = m.get("weight", 0)
        pref_lines.append(
            f"  CONFIRMED MATCH: Your {attr} preference is '{val}' "
            f"({_strength(wt)} preferred, weight={wt:.2f})"
        )

    # Top user preferences (from matched_prefs in payload)
    # These use attribute_name/attribute_value keys from enrichment layer
    top_prefs = sorted(all_prefs, key=lambda x: x.get("weight", 0), reverse=True)[:5]
    for p in top_prefs:
        attr = _human(p.get("attribute_name", p.get("attribute", "")))
        val  = p.get("attribute_value", p.get("value", ""))
        wt   = p.get("weight", 0)
        src  = p.get("source", "")
        if attr and val:
            pref_lines.append(
                f"  USER PREFERENCE: {_strength(wt)} preference for {attr}: '{val}' "
                f"(weight={wt:.2f}, source={src})"
            )

    if not pref_lines:
        pref_lines = ["  User general fashion preferences align with this item"]

    pref_section = "\n".join(pref_lines)

    prior_text = ""
    active = [cl for cl in prior if cl.get("status") == "active"]
    if active:
        prior_text = "\nAlready told user (do NOT contradict):\n" + "\n".join(
            f"  - {cl.get('claim_text', '')}" for cl in active
        )

    strictness_map = {
        0: "Give a natural, friendly explanation in 3-4 sentences.",
        1: "Be precise: cite only confirmed preference matches.",
        2: "One sentence per confirmed match. No additional claims.",
    }
    instruction = strictness_map.get(strictness, strictness_map[0])

    return FASHION_CONTEXT + f"""You are a knowledgeable fashion shopping assistant.

USER QUESTION: "{user_msg}"

ITEM BEING EXPLAINED:
  Name:        {item_name}
  Type:        {item_type}
  Colour:      {item_colour}
  Price:       {item_price}
  Pattern:     {item_pattern}
  Category:    {item_group}
  Description: {item_desc}

WHY THIS ITEM MATCHES THIS USER:
{pref_section}
{prior_text}

TASK: {instruction}

WRITE YOUR EXPLANATION:
- Start with "I recommended {item_name} because..."
- Mention the {item_colour} colour and {item_type} type specifically
- Explain the 2-3 strongest preference matches in plain English
- Mention the price {item_price} and value
- Be specific and informative

STRICT RULES:
- Only talk about {item_name}
- Never use raw field names like colour_group_name or product_type_name
- Only use facts listed above — do not invent details"""


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
    farewell_words = [
        "thanks", "thank you", "thnak you", "thankyou", "ty",
        "helpful", "bye", "goodbye", "good bye", "cheers",
        "great", "awesome", "perfect", "wonderful", "appreciate",
        "that's all", "thats all", "no more", "done", "finished",
        "see you", "cya", "take care",
    ]
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
    """
    Unified LLM caller. Routes to Groq or Ollama based on LLM_PROVIDER setting.
    LLM_PROVIDER=groq  → Groq cloud API (recommended: fast, free, no local RAM)
    LLM_PROVIDER=ollama → Local Ollama (fallback)
    """
    if LLM_PROVIDER == "groq":
        return await _call_groq(prompt, max_tokens)
    else:
        return await _call_ollama_local(prompt, max_tokens)


async def _call_groq(prompt: str, max_tokens: int = 300) -> str:
    """Calls Groq cloud API — same Llama 3.1 8B model, 10-20x faster than local."""
    import time
    _t0 = time.time()
    print(f"[GROQ] ─── calling Groq API: model={GROQ_MODEL} max_tokens={max_tokens}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       GROQ_MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  max_tokens,
                    "temperature": 0.3,
                    "top_p":       0.9,
                }
            )
        elapsed = time.time() - _t0
        if response.status_code != 200:
            print(f"[GROQ] Error {response.status_code}: {response.text[:200]}")
            return ""
        text = response.json()["choices"][0]["message"]["content"].strip()
        print(f"[GROQ] response in {elapsed:.1f}s len={len(text)} chars")
        return text
    except Exception as e:
        print(f"[GROQ] Exception: {str(e)[:150]}")
        return ""


async def _call_ollama_local(prompt: str, max_tokens: int = 300) -> str:
    """Calls local Ollama (fallback when LLM_PROVIDER=ollama)."""
    import time
    _t0 = time.time()
    print(f"[OLLAMA] ─── calling local Ollama: model={OLLAMA_RAG_MODEL}")
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
        elapsed = time.time() - _t0
        if response.status_code != 200:
            return ""
        text = response.json().get("response", "").strip()
        print(f"[OLLAMA] response in {elapsed:.1f}s len={len(text)} chars")
        return text
    except Exception as e:
        err_str = str(e)
        if not err_str or err_str.strip() == "":
            print(f"[ResponseGenerator] Ollama timeout/no response — using fallback")
        elif "Connection" in err_str or "refused" in err_str:
            print(f"[ResponseGenerator] Ollama not running — run: ollama serve")
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
        print(f"\n[RESPONSE-GEN] ━━━ generate() called ━━━")
        action = evidence.get("action", "no_retrieval")
        print(f"[RESPONSE-GEN] action={action} strictness={strictness}")
        print(f"[RESPONSE-GEN] evidence_items={len(evidence.get('items',[]))}")

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

        print(f"[DBG-5a] PROMPT action={action} strictness={strictness} prompt_len={len(prompt)}")
        print(f"[DBG-5a] PROMPT PREVIEW: {prompt[:200].replace(chr(10), ' ')!r}")
        response = await call_ollama(prompt)

        # Basic cleanup and markdown removal
        response = response.strip()
        if not response:
            print("[DBG-5b] OLLAMA: no response → using fallback")
            return _clean_response(self._fallback_response(evidence))

        _cleaned = _clean_response(response)
        print(f"[RESPONSE-GEN] Ollama: RESPONSE len={len(response)}")
        print(f"[RESPONSE-GEN] Cleaned: {repr(_cleaned[:200])}")
        return _cleaned

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
            facts     = evidence.get("comparison_facts", {})
            dim       = evidence.get("comparison_dimension", "overall")
            items_all = evidence.get("items_all")
            a_name    = facts.get("item_a_name", "Option 1")
            b_name    = facts.get("item_b_name", "Option 2")
            # Multi-item price fallback
            if dim == "price" and items_all and len(items_all) > 2:
                ranked = facts.get("all_items_ranked", [])
                if ranked:
                    cheapest = ranked[0]
                    others = ", ".join(
                        f"{r['name']} ({r['colour']}) {r['price']}"
                        for r in ranked[1:]
                    )
                    return (
                        f"The cheapest is {cheapest['name']} ({cheapest['colour']}) "
                        f"at {cheapest['price']}. "
                        f"Other prices: {others}."
                    )
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
            article   = evidence.get("article") or {}
            matches   = evidence.get("confirmed_matches", [])
            all_prefs = evidence.get("matched_prefs", [])
            name      = article.get("name", "this item")
            colour    = article.get("colour", "")
            ptype     = article.get("type", "")
            price     = article.get("price", "?")
            desc      = article.get("material_description", "")[:120]
            _AM = {"colour_group_name":"colour","product_type_name":"type",
                   "occasion":"occasion","style":"style"}
            def _h(a): return _AM.get(a, a.replace("_"," ").replace(" name",""))
            reasons = []
            for m in matches[:3]:
                reasons.append(f"its {_h(m.get('attribute',''))} is {m.get('value','')}")
            if len(reasons) < 2:
                top = sorted(all_prefs, key=lambda x: x.get("weight",0), reverse=True)
                for p in top[:3]:
                    attr = _h(p.get("attribute_name", p.get("attribute","")))
                    val  = p.get("attribute_value", p.get("value",""))
                    if attr and val:
                        reasons.append(f"you prefer {attr}: {val}")
                    if len(reasons) >= 3:
                        break
            reason_text = ", and ".join(reasons[:3]) if reasons else "it matches your preferences"
            result = (
                f"I recommended {name} because {reason_text}. "
                f"It is a {colour} {ptype} priced at {price}."
            )
            if desc:
                result += f" {desc}"
            return result

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
