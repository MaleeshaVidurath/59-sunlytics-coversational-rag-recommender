"""
generate_training_data.py
=========================
Training corpus generation script for the DistilBERT 8-class intent classifier.
Sunlytics Conversational Recommender System (CRS) — M3 Adaptive RAG Module.

PURPOSE
-------
Generates the synthetic training dataset for the retrieval trigger classifier
using LLM-assisted generation (Anthropic Claude claude-haiku-4-5).

All samples follow the production input format:
    USER: <prev_message> [SEP] BOT: <bot_response> [SEP] CURRENT: <current_message>
This matches exactly how the classifier receives inputs during live inference
via predict.py:format_input_text().

TARGET DATASET SIZE
-------------------
Total generated : 55,696 samples
After balancing : 52,028 samples (INITIAL_REQUEST capped to match REFINEMENT)

Label               Raw count    After balance
-----------         ---------    -------------
INITIAL_REQUEST     11,440       7,772
REFINEMENT           7,772       7,772
ATTRIBUTE_QUESTION   5,944       5,944
EXPLANATION_WHY      5,880       5,880
COMPARISON           5,882       5,882
SELECTION_REFERENCE  5,951       5,951
FEEDBACK             6,461       6,461
CHITCHAT             6,366       6,366
TOTAL               55,696      52,028

DESIGN DECISIONS
----------------
1. INITIAL_REQUEST is deliberately over-generated (11,440 raw) because 47% of
   samples represent mid-session requests (user switches product category
   mid-conversation). A separate generation pass produces these mid-session
   samples to fix the production misclassification of "I want a coat" (after
   an existing dress conversation) being predicted as REFINEMENT.

2. Class balancing at training time: INITIAL_REQUEST is downsampled to 7,772
   (matching REFINEMENT, the largest other class) to prevent classifier bias.

3. All follow-up classes (REFINEMENT, ATTRIBUTE_QUESTION, EXPLANATION_WHY,
   COMPARISON, SELECTION_REFERENCE, FEEDBACK) always include prior conversation
   context — they are definitionally follow-up turns and cannot exist without
   a prior recommendation from the bot.

4. CHITCHAT is 35% context-bearing (mid-conversation remarks) and 65% standalone
   (greetings, farewells, acknowledgements with no prior exchange).
"""

import os
import json
import time
import random
import pandas as pd
import anthropic
from typing import List, Dict, Tuple

# ── API CLIENT ────────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL  = "claude-haiku-4-5-20251001"

SEED = 42
random.seed(SEED)


# =============================================================================
# STEP 1 — SIX USER PERSONAS
# =============================================================================
# Each persona defines a distinct writing style that represents a real segment
# of e-commerce users. Applied in rotation across all 8 intent classes to
# ensure the classifier handles linguistic diversity, not just canonical phrasing.
# =============================================================================

PERSONAS = [
    {
        "name": "casual_texter",
        "description": "Young adult who texts casually. Uses abbreviations, "
                       "no capitalisation, drops punctuation, very short messages.",
        "style_rules": [
            "Use lowercase throughout",
            "Abbreviate where natural: 'u' for 'you', 'thx' for 'thanks', 'pls' for 'please', 'ngl' for 'not gonna lie'",
            "Drop end punctuation",
            "Keep messages under 8 words where possible",
            "Use fragments: 'cheaper?', 'first one?', 'why tho'",
        ],
        "examples": ["need smth cheaper", "which would u pick", "thx", "first one ngl"],
    },
    {
        "name": "busy_professional",
        "description": "Adult in a hurry. Direct, minimal words, task-focused, "
                       "no small talk. Still grammatically correct but very brief.",
        "style_rules": [
            "Short, direct sentences",
            "No pleasantries",
            "Imperative mood: 'Show me', 'Find', 'Give me'",
            "May include specific constraints: price, colour, occasion",
            "No filler words",
        ],
        "examples": ["Show me black options under £40", "Need something for a meeting", "Which is better quality"],
    },
    {
        "name": "teenager",
        "description": "Teenager shopping for fashion. Uses slang, expressive, "
                       "asks short rhetorical questions, dramatic reactions.",
        "style_rules": [
            "Use teen slang: 'omg', 'literally', 'lowkey', 'slay', 'vibe'",
            "Expressive punctuation: '!!' but also no punctuation at all",
            "Short questions: 'wait why tho?', 'is it cute tho'",
            "Mix capitalisation inconsistently",
        ],
        "examples": ["omg which one tho", "lowkey need smth cheaper", "why is it so expensive lol", "first one slay"],
    },
    {
        "name": "non_native_speaker",
        "description": "User whose first language is not English. Simplified "
                       "grammar, dropped articles, direct word-for-word phrasing.",
        "style_rules": [
            "Drop articles: 'show me black dress' not 'show me a black dress'",
            "Simple verb forms: 'I want' not 'I would like'",
            "May repeat question with 'yes?' or 'no?'",
            "Avoid idioms, use literal phrasing",
            "Short sentences with simple vocabulary",
        ],
        "examples": ["show me black dress", "I want more cheap", "is good quality yes?", "first item please"],
    },
    {
        "name": "polite_shopper",
        "description": "Considerate adult who uses full sentences, please/thank you, "
                       "and polite question forms. Formal register.",
        "style_rules": [
            "Full sentences with proper capitalisation and punctuation",
            "Use 'Could you', 'Would you mind', 'Please'",
            "Express appreciation: 'Thank you', 'That would be lovely'",
            "Explain reasoning: 'I was thinking something more...'",
        ],
        "examples": [
            "Could you show me something a little less expensive, please?",
            "Would you mind telling me more about the first option?",
            "I was wondering whether it comes in other colours.",
        ],
    },
    {
        "name": "indecisive_browser",
        "description": "Uncertain shopper who hedges, asks comparative questions, "
                       "and expresses doubt. Uses 'maybe', 'not sure', 'hmm'.",
        "style_rules": [
            "Hedging language: 'maybe', 'not sure', 'I think', 'hmm'",
            "Comparative questions: 'which is better?', 'what do you think?'",
            "Backtracking: 'actually', 'on second thought'",
            "Vague preferences: 'something nicer', 'a bit different'",
        ],
        "examples": [
            "hmm not sure, maybe something cheaper?",
            "I think I prefer the first one but not sure",
            "which would you say is better quality?",
            "actually, can you show me something different",
        ],
    },
]


# =============================================================================
# STEP 2 — PRODUCT CATALOG SAMPLES
# =============================================================================
# Bot responses always reference real H&M product names, colours, and prices
# drawn from the 41,794-item catalogue. This grounds the conversation in
# authentic domain content rather than invented placeholders.
# =============================================================================

BOT_RESPONSES = [
    "Option 1 is the Spring Wrap dress (black, dress): Wrap dress in woven fabric with a V-neck, £34.99. "
    "Option 2 is the Carnival Shift dress (red, dress): Knee-length shift dress, £29.99.",

    "Take a look at these two: the Slim Tapered chinos (dark blue, trousers): Slim-fit chinos in stretch cotton, £39.99. "
    "Also the Relaxed Worker trousers (beige, trousers): Wide-leg trousers in cotton twill, £44.99.",

    "Option 1 is the Fav Regular polo (white, polo shirt): Cotton piqué polo, £24.99. "
    "Option 2 is the Oxford Button-down shirt (light blue, shirt): Regular-fit Oxford shirt, £34.99.",

    "Here are two options. First, the OL South PQ sneaker (white, shoes): Clean canvas trainers with rubber sole, £29.99. "
    "Second, the Divided chunky loafer (black, shoes): Chunky platform loafer, £39.99.",

    "Option 1 is the Ribbed turtleneck (grey, jumper): Fine-knit turtleneck in soft viscose blend, £44.99. "
    "Option 2 is the Oversized hoodie (dark green, hoodie): Relaxed-fit hoodie in brushed fabric, £34.99.",

    "Take a look at these: the Tailored blazer (dark grey, blazer): Single-breasted blazer in stretch fabric, £69.99. "
    "Also the Cotton denim jacket (light blue, jacket): Classic denim jacket in washed cotton, £49.99.",

    "Here are two bags: the Tote shopper (beige, bag): Large tote in faux leather with internal pockets, £29.99. "
    "Also the Mini crossbody (black, bag): Small crossbody bag with adjustable strap, £19.99.",

    "Option 1 is the Divided wide-leg jeans (light denim, jeans): High-waisted wide-leg jeans, £39.99. "
    "Option 2 is the Skinny Ankle jeans (dark blue, jeans): Classic skinny jeans with ankle length, £34.99.",
]


# =============================================================================
# STEP 3 — PROMPT TEMPLATES PER INTENT CLASS
# =============================================================================
# Each class uses a structured prompt that specifies:
#   (a) the intent label and its retrieval meaning
#   (b) the required output format
#   (c) the active persona's style rules
#   (d) any boundary constraints (especially for REFINEMENT vs FEEDBACK)
# =============================================================================

def build_prompt(label: str, persona: Dict, bot_response: str, n: int = 10,
                 mid_session: bool = False) -> str:
    """
    Constructs the generation prompt for a given label + persona combination.

    Parameters
    ----------
    label       : one of the 8 intent class names
    persona     : persona dict from PERSONAS list
    bot_response: a real product recommendation from BOT_RESPONSES
    n           : number of samples to generate per call
    mid_session : for INITIAL_REQUEST only — whether to generate mid-session
                  examples (user switches product category mid-conversation)
    """

    persona_instruction = (
        f"Writing style — {persona['name'].replace('_', ' ').title()} persona:\n"
        + "\n".join(f"  - {rule}" for rule in persona["style_rules"])
    )

    format_instruction = (
        "Output format — return ONLY a JSON array of objects. Each object must have:\n"
        '  "current_message": the user message you generate\n'
        '  "input_text": the full context string in format '
        '"USER: <prev_user_msg> [SEP] BOT: <bot_response> [SEP] CURRENT: <current_message>"\n'
        "Do not include any explanation outside the JSON array."
    )

    # ── Class-specific instructions ───────────────────────────────────────────

    if label == "INITIAL_REQUEST":
        if mid_session:
            instruction = (
                f"Generate {n} INITIAL_REQUEST examples where the user is MID-SESSION "
                f"(they have already had a conversation about one product, then suddenly "
                f"request a completely different product category).\n\n"
                f"The prior conversation shows the bot recommended: {bot_response}\n\n"
                f"The user now asks for a DIFFERENT product type entirely "
                f"(e.g. if the prior context was about dresses, the new request should be "
                f"for shoes, bags, jackets, jeans, etc.).\n\n"
                f"CRITICAL: The current_message must be a NEW product request, not a "
                f"refinement of the previous one. The retrieval system must do a FULL "
                f"fresh search.\n\n"
                f"{persona_instruction}\n\n"
                f"{format_instruction}"
            )
        else:
            instruction = (
                f"Generate {n} INITIAL_REQUEST examples where the user opens a "
                f"conversation with their first product request (no prior context).\n\n"
                f"The input_text should be: CURRENT: <current_message> "
                f"(no USER/BOT history — this is the very first message).\n\n"
                f"Requests should cover a variety of fashion categories: dresses, "
                f"tops, trousers, shoes, bags, jackets, jumpers, accessories.\n\n"
                f"{persona_instruction}\n\n"
                f"IMPORTANT: For first-turn examples, set input_text to exactly "
                f'"CURRENT: <current_message>" with no USER/BOT prefix.\n\n'
                f"{format_instruction}"
            )

    elif label == "REFINEMENT":
        instruction = (
            f"Generate {n} REFINEMENT examples. The user has just seen two product "
            f"recommendations and wants to REFINE the search — they are asking for "
            f"something different or more specific, but still in the same product category.\n\n"
            f"Prior bot recommendation: {bot_response}\n\n"
            f"CRITICAL BOUNDARY — REFINEMENT vs FEEDBACK:\n"
            f"  REFINEMENT: implies a new/different suggestion is wanted "
            f"(e.g. 'something cheaper', 'in black please', 'more casual ones')\n"
            f"  FEEDBACK: just expresses opinion with no request for alternatives "
            f"(e.g. 'too expensive', 'not my style') — DO NOT generate this.\n\n"
            f"Cover these refinement types: price (cheaper/more affordable), "
            f"colour (different colour), style (more casual/formal/sporty), "
            f"size constraint, occasion-based ('for a wedding', 'for work').\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    elif label == "ATTRIBUTE_QUESTION":
        instruction = (
            f"Generate {n} ATTRIBUTE_QUESTION examples. The user is asking a specific "
            f"factual question about one of the products already presented by the bot.\n\n"
            f"Prior bot recommendation: {bot_response}\n\n"
            f"Questions should cover: materials (fabric, composition), care instructions "
            f"(machine washable, dry clean), sizing (does it run large?), availability "
            f"(other colours, sizes in stock), sustainability (eco materials), "
            f"dimensions, weight.\n\n"
            f"The question must refer to something already shown — it cannot be a new "
            f"product request.\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    elif label == "EXPLANATION_WHY":
        instruction = (
            f"Generate {n} EXPLANATION_WHY examples. The user wants to know WHY the bot "
            f"chose these particular recommendations — asking for reasoning, justification, "
            f"or explanation of the suggestion.\n\n"
            f"Prior bot recommendation: {bot_response}\n\n"
            f"Vary the phrasing: 'why this one?', 'what makes it suitable?', "
            f"'why do you think this fits me?', 'what's your reasoning?', 'why recommend this?'\n\n"
            f"Include very short forms ('why?', 'why tho', 'reason?') and longer forms "
            f"('I'm not convinced — why do you think this is right for me?').\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    elif label == "COMPARISON":
        instruction = (
            f"Generate {n} COMPARISON examples. The user wants to compare the TWO options "
            f"the bot has presented and asks which is better, what the differences are, "
            f"or which one the bot recommends.\n\n"
            f"Prior bot recommendation: {bot_response}\n\n"
            f"Vary the phrasing: 'which is better?', 'what's the difference?', "
            f"'first or second?', 'which would you recommend?', 'how do they compare?', "
            f"'pros and cons of each?'\n\n"
            f"Include very short forms ('first or second?', 'which one?') and longer forms.\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    elif label == "SELECTION_REFERENCE":
        instruction = (
            f"Generate {n} SELECTION_REFERENCE examples. The user is referring to one "
            f"of the two presented options and wants more detail, wants to select it, "
            f"or is asking something specifically about it by reference.\n\n"
            f"Prior bot recommendation: {bot_response}\n\n"
            f"Reference patterns: 'tell me more about the first one', 'the second one — "
            f"more details?', 'option 1 please', 'expand on the black one', "
            f"'the cheaper one — what else can you tell me?'\n\n"
            f"The key feature: the user references 'first', 'second', 'option 1/2', "
            f"or a specific product name/colour from the bot response.\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    elif label == "FEEDBACK":
        instruction = (
            f"Generate {n} FEEDBACK examples. The user is giving a reaction to the "
            f"presented options — positive acceptance, negative rejection, or neutral "
            f"acknowledgement — WITHOUT asking for anything new.\n\n"
            f"Prior bot recommendation: {bot_response}\n\n"
            f"CRITICAL BOUNDARY — FEEDBACK vs REFINEMENT:\n"
            f"  FEEDBACK: pure reaction, no new request "
            f"(e.g. 'too expensive', 'I'll take it', 'not my style', 'perfect!')\n"
            f"  REFINEMENT: reaction PLUS implicit request for alternatives — DO NOT generate this.\n\n"
            f"Cover: positive ('love it', 'perfect', 'I'll take that'), "
            f"negative ('too expensive', 'not my style', 'too formal', 'don't like the colour'), "
            f"neutral ('okay', 'I see', 'fair enough').\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    elif label == "CHITCHAT":
        instruction = (
            f"Generate {n} CHITCHAT examples. The user is making a social comment "
            f"that is NOT related to product search — greetings, farewells, "
            f"expressions of thanks, or casual conversation.\n\n"
            f"35% of examples should be mid-conversation (include prior context): "
            f"the user makes a remark after seeing: {bot_response}\n"
            f"65% should be standalone (no prior context): greetings, farewells, "
            f"random remarks with no product context.\n\n"
            f"For standalone examples: set input_text to 'CURRENT: <current_message>'.\n\n"
            f"Examples: 'good morning', 'thanks!', 'lol ok', 'that was helpful', "
            f"'you're great', 'bye', 'haha no worries', 'fair enough'.\n\n"
            f"{persona_instruction}\n\n"
            f"{format_instruction}"
        )

    else:
        raise ValueError(f"Unknown label: {label}")

    return instruction


# =============================================================================
# STEP 4 — GENERATION LOOP
# =============================================================================

LABEL_NAMES = [
    "INITIAL_REQUEST", "REFINEMENT", "ATTRIBUTE_QUESTION", "EXPLANATION_WHY",
    "COMPARISON", "SELECTION_REFERENCE", "FEEDBACK", "CHITCHAT",
]

LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

RETRIEVAL_STRATEGY = {
    "INITIAL_REQUEST": "FULL", "REFINEMENT": "FULL",
    "ATTRIBUTE_QUESTION": "PARTIAL", "EXPLANATION_WHY": "PARTIAL",
    "COMPARISON": "PARTIAL", "SELECTION_REFERENCE": "PARTIAL",
    "FEEDBACK": "NO", "CHITCHAT": "NO",
}

# Target raw counts per class (total = 55,696)
# INITIAL_REQUEST is split: 6,067 first-turn + 5,373 mid-session = 11,440
TARGET_COUNTS = {
    "INITIAL_REQUEST":    11440,   # 6,067 first-turn + 5,373 mid-session
    "REFINEMENT":          7772,
    "ATTRIBUTE_QUESTION":  5944,
    "EXPLANATION_WHY":     5880,
    "COMPARISON":          5882,
    "SELECTION_REFERENCE": 5951,
    "FEEDBACK":            6461,
    "CHITCHAT":            6366,
}

# INITIAL_REQUEST split: 53% first-turn, 47% mid-session
INITIAL_REQUEST_FIRST_TURN  = 6067
INITIAL_REQUEST_MID_SESSION = 5373

BATCH_SIZE = 10   # samples per API call


def call_api(prompt: str) -> List[Dict]:
    """Call Claude claude-haiku-4-5 and parse the returned JSON array."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Extract JSON array from response
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    return json.loads(text[start:end])


def generate_class(label: str, target: int, mid_session_split: Tuple[int, int] = None) -> List[Dict]:
    """
    Generate `target` samples for a given intent class.

    For INITIAL_REQUEST, mid_session_split = (n_first_turn, n_mid_session).
    Personas are rotated uniformly across all generated samples.
    """
    rows = []
    label_id = LABEL_TO_ID[label]
    strategy = RETRIEVAL_STRATEGY[label]

    if label == "INITIAL_REQUEST" and mid_session_split:
        n_first, n_mid = mid_session_split
        # ── First-turn pass ───────────────────────────────────────────────────
        print(f"  Generating {n_first} first-turn INITIAL_REQUEST samples...")
        rows += _generate_pass(label, n_first, label_id, strategy, mid_session=False)
        # ── Mid-session pass ──────────────────────────────────────────────────
        print(f"  Generating {n_mid} mid-session INITIAL_REQUEST samples...")
        rows += _generate_pass(label, n_mid, label_id, strategy, mid_session=True)
    else:
        rows += _generate_pass(label, target, label_id, strategy)

    return rows


def _generate_pass(label: str, target: int, label_id: int, strategy: str,
                   mid_session: bool = False) -> List[Dict]:
    rows = []
    persona_idx = 0
    generated = 0

    while generated < target:
        persona     = PERSONAS[persona_idx % len(PERSONAS)]
        bot_resp    = random.choice(BOT_RESPONSES)
        batch_n     = min(BATCH_SIZE, target - generated)
        prompt      = build_prompt(label, persona, bot_resp, n=batch_n, mid_session=mid_session)

        try:
            samples = call_api(prompt)
        except Exception as e:
            print(f"    API error: {e} — retrying in 5s")
            time.sleep(5)
            continue

        for s in samples[:batch_n]:
            if "current_message" not in s or "input_text" not in s:
                continue
            rows.append({
                "input_text":              s["input_text"],
                "current_message":         s["current_message"],
                "conversation_history_json": json.dumps(
                    _extract_history(s["input_text"])
                ),
                "label":          label_id,
                "label_name":     label,
                "retrieval_strategy": strategy,
                "exchanges":      _count_exchanges(s["input_text"]),
            })
            generated += 1
            if generated >= target:
                break

        persona_idx += 1
        time.sleep(0.3)   # rate-limit courtesy pause

    return rows


def _extract_history(input_text: str) -> List[Dict]:
    """Parse the USER/BOT segments from input_text into a history list."""
    history = []
    parts = input_text.split(" [SEP] ")
    for part in parts:
        if part.startswith("USER:"):
            history.append({"role": "user", "content": part[5:].strip()})
        elif part.startswith("BOT:"):
            history.append({"role": "bot", "content": part[4:].strip()})
    return history


def _count_exchanges(input_text: str) -> float:
    """Count the number of USER/BOT exchange pairs in input_text."""
    user_count = input_text.count("USER:")
    bot_count  = input_text.count("BOT:")
    if user_count == 0:
        return 0.0
    return float(min(user_count, bot_count))


# =============================================================================
# STEP 5 — MAIN
# =============================================================================

def main():
    all_rows = []

    for label in LABEL_NAMES:
        target = TARGET_COUNTS[label]
        print(f"\nGenerating {target:,} samples for: {label}")

        if label == "INITIAL_REQUEST":
            rows = generate_class(
                label, target,
                mid_session_split=(INITIAL_REQUEST_FIRST_TURN, INITIAL_REQUEST_MID_SESSION)
            )
        else:
            rows = generate_class(label, target)

        all_rows.extend(rows)
        print(f"  Done — {len(rows):,} samples generated.")

    # ── Shuffle and save ──────────────────────────────────────────────────────
    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows)
    df = df[["input_text", "current_message", "conversation_history_json",
             "label", "label_name", "retrieval_strategy", "exchanges"]]

    output_path = "data/v4_train_midSession.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("GENERATION COMPLETE")
    print("=" * 55)
    print(f"Total rows saved : {len(df):,}")
    print(f"Output file      : {output_path}")
    print()
    print("Label distribution:")
    for label in LABEL_NAMES:
        n = (df["label_name"] == label).sum()
        print(f"  {label:<25} {n:>6}")
    print("=" * 55)

    # ── Balanced version ──────────────────────────────────────────────────────
    # Downsample INITIAL_REQUEST to match REFINEMENT (largest other class)
    max_other = max(
        (df["label_name"] == l).sum()
        for l in LABEL_NAMES if l != "INITIAL_REQUEST"
    )
    ir_rows    = df[df["label_name"] == "INITIAL_REQUEST"].sample(n=max_other, random_state=SEED)
    other_rows = df[df["label_name"] != "INITIAL_REQUEST"]
    df_balanced = pd.concat([ir_rows, other_rows]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    balanced_path = "data/v4_train_balanced.csv"
    df_balanced.to_csv(balanced_path, index=False, encoding="utf-8")
    print(f"\nBalanced dataset : {len(df_balanced):,} rows → {balanced_path}")
    print(f"(INITIAL_REQUEST capped at {max_other:,} to match REFINEMENT)")


if __name__ == "__main__":
    main()
