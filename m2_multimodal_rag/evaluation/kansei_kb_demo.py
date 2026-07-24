"""
Kansei Psychology KB — Live Demo
Shows how emotional style words are detected and translated
into psychology-grounded product attribute boosts.

Run from project root:
    python Accuracy/kansei_kb_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m2_multimodal_rag.knowledge_base.kb_retriever import kb_retriever

SEP  = "=" * 65
THIN = "-" * 65

# Demo queries with style/occasion words
DEMO_QUERIES = [
    ("I want an elegant black dress for a formal evening",  "elegant",      "formal"),
    ("Show me something casual for the weekend",            "casual",       "casual"),
    ("I need a sporty outfit for the gym",                  "sporty",       "gym"),
    ("Find me a professional look for an interview",        "professional", "interview"),
]

# Candidate items to score
CANDIDATES = [
    ("Black",     "Dress",    "Solid",              "Ladieswear"),
    ("White",     "Top",      "Solid",              "Ladieswear"),
    ("Dark Blue", "Trousers", "Solid",              "Menswear"),
    ("Grey",      "Sweater",  "All over pattern",   "Ladieswear"),
    ("Red",       "Dress",    "Stripe",             "Divided"),
]

print(f"\n{SEP}")
print("  NOVELTY 5 — Kansei Psychology KB  |  Live Demo")
print("  Emotional vocabulary → psychology-grounded scoring boost")
print(SEP)

for query, style_word, occasion in DEMO_QUERIES:
    print(f"\n  Query   : '{query}'")
    print(f"  Detected: style='{style_word}'  occasion='{occasion}'")

    # KB context injected into CLIP search and LLM reranking
    context = kb_retriever.get_context(occasion=occasion, style_word=style_word)
    clip_terms = kb_retriever.get_clip_terms(occasion=occasion, style_word=style_word)

    if context:
        print(f"  KB context  : {context[:80]}...")
    if clip_terms:
        print(f"  CLIP terms  : {clip_terms[:80]}...")

    print(f"\n  {'Item':<40} {'KB Score':>10}  Result")
    print(f"  {THIN}")

    for colour, ptype, appearance, index_group in CANDIDATES:
        score = kb_retriever.score(
            item_colour=colour,
            item_type=ptype,
            item_appearance=appearance,
            occasion=occasion,
            style_word=style_word,
            index_group_name=index_group,
        )
        label = f"{colour} {ptype} ({appearance})"
        result = "BOOSTED" if score > 0.1 else ("penalised" if score < 0 else "neutral")
        print(f"  {label:<40} {score:>+10.3f}  {result}")

print(f"\n{SEP}")
print("  KEY INSIGHT")
print(THIN)
print("  Standard CLIP and NCF ignore words like 'elegant', 'casual',")
print("  'sporty'. The Kansei KB translates these emotional signals into")
print("  concrete scoring boosts — psychology-grounded retrieval.")
print(f"{SEP}\n")
