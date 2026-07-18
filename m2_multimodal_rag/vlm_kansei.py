"""
VLM Kansei — automatic visual-psychology capture (NOVELTY 5 evolution).

Replaces the manual Kansei knowledge-base lookup at explanation time: instead
of hand-written style tables (fashion_kb.KANSEI_MAPPING), the Groq vision
model looks at the actual product image and produces a one-sentence
visual-psychology assessment ("the relaxed drape and muted tones convey a
casual, approachable feel"), optionally judged against the customer's
detected style word.

Design: VLM-first, manual-KB fallback —
  - product image cached locally AND Groq available → VLM assessment
    (memoised per article+style, so repeated turns cost zero extra calls)
  - image missing or VLM unavailable → returns "" and the caller falls back
    to kb_retriever.get_explanation() (the manual KB)

The manual KB remains authoritative for Phase-2 candidate *scoring* — this
module only affects the explanation/lookup grounding facts, where a wrong
sentence is caught by the hallucination guard anyway.
"""

from collections import OrderedDict
from pathlib import Path

from m2_multimodal_rag.llm_generator import llm_generator

_CACHE_MAX = 500
_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_put(key: str, value: str) -> None:
    _cache.pop(key, None)
    _cache[key] = value
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def visual_psychology_fact(
    article_id: str,
    image_path,
    metadata: dict,
    style_word: str | None = None,
) -> str:
    """
    Returns a one-sentence VLM-generated visual-psychology fact for the item,
    or "" when the image isn't locally available / the VLM is unavailable —
    the caller then falls back to the manual Kansei KB.
    """
    key = f"{article_id}|{(style_word or '').lower()}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    # Only ever call the VLM with a real local image — the point is visual
    # capture; a text-only guess would defeat the purpose.
    if not image_path or not Path(str(image_path)).exists():
        return ""
    if not llm_generator.is_available:
        return ""

    style_clause = (
        f" The customer's style preference is '{style_word}' — mention whether "
        f"the item's visual character suits that style."
        if style_word else ""
    )
    prompt = (
        f"You are a fashion psychology expert. Look at this product image "
        f"({metadata.get('prod_name', 'item')} — "
        f"{metadata.get('colour_group_name', '')} "
        f"{metadata.get('product_type_name', '')}).\n"
        f"In ONE sentence (max 25 words), describe the visual psychology this "
        f"item conveys: the mood, formality and emotional impression created "
        f"by its colour, texture and cut.{style_clause} "
        f"Reply with the sentence only, no preamble."
    )

    result = llm_generator._call_vision_llm(
        prompt, str(image_path), max_tokens=60, temperature=0.4
    )
    fact = (result or "").strip().strip('"')
    if fact:
        print(f"   [VLM-Kansei] {article_id}: \"{fact[:80]}...\"" if len(fact) > 80
              else f"   [VLM-Kansei] {article_id}: \"{fact}\"")
        _cache_put(key, fact)
    return fact
