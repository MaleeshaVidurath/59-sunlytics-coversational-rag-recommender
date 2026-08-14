"""
M2 Action Handlers — Implements each action type from the retrieval_input spec.

Each handler receives the full retrieval_input dict and returns a standardized
response dict with: action, success, response_text, items, error.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

from shared.data_loader import data_loader
from m2_multimodal_rag.llm_generator import llm_generator
from m2_multimodal_rag.clip_embeddings import clip_encoder
from m2_multimodal_rag.faiss_index import faiss_db
from m2_multimodal_rag.hallucination_guard.regeneration_loop import generator_loop
from m2_multimodal_rag.hallucination_guard.layer_3_vlm_visual_verification import blip_verifier
from m2_multimodal_rag.hallucination_guard.layer_2_cove_verification import cove_verifier
from m2_multimodal_rag.cross_encoder_reranker import cross_encoder_reranker
from m2_multimodal_rag.diversity_bandit import diversity_bandit
from m2_multimodal_rag.knowledge_base.kb_retriever import kb_retriever
from m2_multimodal_rag.collaborative_filtering.cf_scorer import cf_scorer
from m2_multimodal_rag.vlm_kansei import visual_psychology_fact

# Attempt to load CF model at import time (silent if files not present)
cf_scorer.load()

# =====================================================================
# Ablation switches (evaluation only — all default OFF)
# =====================================================================
# Environment variables that disable one novelty at a time so evaluation
# scripts (m2_multimodal_rag/evaluation/) can measure ON-vs-OFF baselines:
#   M2_ABLATE_ENSEMBLE=1        → single query vector (no LLM expansion)
#   M2_ABLATE_CF=1              → rule-based purchase score instead of NCF
#   M2_ABLATE_KB=1              → Kansei KB off (score, CLIP terms, facts)
#   M2_ABLATE_BANDIT=<λ>       → fixed MMR λ (read in diversity_bandit.py)
#   M2_ABLATE_GUARD=none|l1|l12 → guard stage gating (read in regeneration_loop.py)
_ABLATE_ENSEMBLE = os.getenv("M2_ABLATE_ENSEMBLE") == "1"
_ABLATE_CF       = os.getenv("M2_ABLATE_CF") == "1"
_ABLATE_KB       = os.getenv("M2_ABLATE_KB") == "1"
if _ABLATE_ENSEMBLE or _ABLATE_CF or _ABLATE_KB:
    print(f"M2 ABLATION ACTIVE: ensemble_off={_ABLATE_ENSEMBLE} "
          f"cf_off={_ABLATE_CF} kb_off={_ABLATE_KB}")

# =====================================================================
# Article prices (M2-local, in-memory only — shared/ files untouched)
# =====================================================================

from shared.config import SAMPLE_DATA_DIR

_PRICE_SCALE = 595.08  # normalised price → £ (same value M3 uses in text_rag/config.py)


def _load_articles_priced():
    """Returns articles_df with an avg £ `price` column merged in from
    sample_transactions.csv. Merged once, in-memory only; articles never
    purchased in the sample keep price=NaN."""
    import pandas as pd
    df = data_loader.load_articles()
    if 'price' in df.columns:
        return df
    tx_path = SAMPLE_DATA_DIR / 'sample_transactions.csv'
    if not tx_path.exists():
        print("  [prices] sample_transactions.csv not found — price filters will be skipped.")
        return df
    tx = pd.read_csv(tx_path, usecols=['article_id', 'price'])
    avg_price = (tx.groupby('article_id')['price'].mean() * _PRICE_SCALE).round(2)
    df = df.merge(avg_price.rename('price'), on='article_id', how='left')
    data_loader.articles_df = df   # cache in the loader so the merge runs only once
    print(f"  [prices] Attached avg £ prices to {df['price'].notna().sum()}/{len(df)} articles.")
    return df


# =====================================================================
# Filtered FAISS helper (hard filters at index level — pool 50 → 15)
# =====================================================================

def _hard_filter_allowed_ids(articles_df, filters: dict, exclude_ids: list) -> list:
    """Vectorised hard-filter pass over the full article catalogue.
    Returns article_ids (zero-padded 10-char strings) satisfying every hard
    filter — same semantics as the per-candidate Phase 2 checks:
    case-insensitive equality for attributes, price_min/price_max range with
    unknown (NaN) prices never excluding an item. An unknown filter column
    yields [] so the caller falls back to unfiltered search."""
    import pandas as pd
    mask = pd.Series(True, index=articles_df.index)
    for key, value in filters.items():
        if key in ("price_max", "price_min"):
            if "price" not in articles_df.columns:
                continue
            price = articles_df["price"]
            if key == "price_max":
                mask &= price.isna() | (price <= value)
            else:
                mask &= price.isna() | (price >= value)
        else:
            if key not in articles_df.columns:
                return []
            mask &= (articles_df[key].astype(str).str.strip().str.lower()
                     == str(value).strip().lower())
    ids = articles_df.loc[mask, "article_id"].astype(str).str.zfill(10)
    if exclude_ids:
        excl = {str(e).zfill(10) for e in exclude_ids}
        ids = ids[~ids.isin(excl)]
    return ids.tolist()


# =====================================================================
# Session candidate-pool cache (follow-up handling via M3 session memory)
# =====================================================================
# After every full catalog_search, the scored candidate pool is cached per
# session. A REFINEMENT follow-up ("cheaper ones", "not that one") is then
# resolved by re-filtering the cached pool — no query-expansion LLM call,
# no CLIP ensemble, no FAISS re-retrieval.

from collections import OrderedDict as _OrderedDict
import time as _time

_SESSION_POOL_TTL_S = 1800   # cached pools expire after 30 minutes
_SESSION_POOL_MAX   = 50     # max concurrent sessions kept in memory

_session_pool_cache: "_OrderedDict[str, dict]" = _OrderedDict()


def _session_pool_put(session_id: str, pool: list) -> None:
    """Caches the session's scored candidate pool as (article_id, faiss_score)
    pairs. Evicts oldest sessions beyond _SESSION_POOL_MAX."""
    if not session_id or not pool:
        return
    _session_pool_cache.pop(session_id, None)
    _session_pool_cache[session_id] = {"pool": pool, "ts": _time.time()}
    while len(_session_pool_cache) > _SESSION_POOL_MAX:
        _session_pool_cache.popitem(last=False)


def _session_pool_get(session_id: str) -> list | None:
    """Returns the cached pool for a session, or None if absent/expired."""
    entry = _session_pool_cache.get(session_id)
    if not entry:
        return None
    if _time.time() - entry["ts"] > _SESSION_POOL_TTL_S:
        _session_pool_cache.pop(session_id, None)
        return None
    return entry["pool"]


def _followup_pool_from_cache(session_id: str, articles_df, filters: dict,
                              exclude_ids: list, min_needed: int) -> list | None:
    """
    Follow-up fast path: re-applies the (merged) hard filters and exclusions
    to the session's cached candidate pool. Returns surviving
    (article_id, faiss_score) pairs in original relevance order, or None when
    the fast path can't be used (no/expired cache, or too few survivors —
    e.g. a refinement that genuinely needs fresh retrieval, like a new colour
    against a colour-filtered pool).
    """
    cached = _session_pool_get(session_id)
    if not cached:
        return None

    if filters:
        allowed = set(_hard_filter_allowed_ids(articles_df, filters, exclude_ids))
        survivors = [(aid, score) for aid, score in cached if aid in allowed]
    else:
        excl = {str(e).zfill(10) for e in (exclude_ids or [])}
        survivors = [(aid, score) for aid, score in cached if aid not in excl]

    if len(survivors) < min_needed:
        print(f"  [follow-up] Cached pool too narrow after re-filtering "
              f"({len(survivors)} < {min_needed}) — full retrieval needed.")
        return None
    return survivors


# =====================================================================
# Session recommended-items cache (ordinal reference resolution)
# =====================================================================
# Remembers the ordered item list of each session's last recommendation,
# so ordinal follow-ups ("the second one") resolve correctly even when
# M3's dialogue state has been narrowed to a single item by intervening
# single-item follow-ups. Same TTL/eviction pattern as the pool cache.

_session_items_cache: "_OrderedDict[str, dict]" = _OrderedDict()


def _session_items_put(session_id: str, items: list) -> None:
    """Caches the ordered (article_id, prod_name) list of a recommendation."""
    if not session_id or not items:
        return
    _session_items_cache.pop(session_id, None)
    _session_items_cache[session_id] = {"items": items, "ts": _time.time()}
    while len(_session_items_cache) > _SESSION_POOL_MAX:
        _session_items_cache.popitem(last=False)


def _session_items_get(session_id: str) -> list | None:
    entry = _session_items_cache.get(session_id)
    if not entry:
        return None
    if _time.time() - entry["ts"] > _SESSION_POOL_TTL_S:
        _session_items_cache.pop(session_id, None)
        return None
    return entry["items"]


_ORDINAL_MAP = [
    (("first", "1st", "option 1", "number one", "number 1", "item 1"), 0),
    (("second", "2nd", "option 2", "number two", "number 2", "item 2"), 1),
    (("third", "3rd", "option 3", "number three", "number 3", "item 3"), 2),
    (("fourth", "4th"), 3),
    (("fifth", "5th"), 4),
    (("sixth", "6th"), 5),
    (("last one", "last item"), -1),
]


def _session_reference_hits(user_message: str, session_id: str) -> list:
    """
    Resolves explicit item references in the message (ordinals like "the
    second one", or product names) against this session's last cached
    recommendation. Returns matching item dicts in mention order (ordinals
    first). Empty list when nothing explicit is referenced or no cache.
    """
    items = _session_items_get(session_id) if session_id else None
    if not items:
        return []
    msg = (user_message or "").lower()
    hits = []
    for words, idx in _ORDINAL_MAP:
        if any(w in msg for w in words):
            i = len(items) - 1 if idx == -1 else idx
            if i < len(items) and items[i] not in hits:
                hits.append(items[i])
    for it in items:
        name = str(it.get("prod_name", "")).lower()
        if len(name) >= 4 and name in msg and it not in hits:
            hits.append(it)
    return hits


def _override_from_session(article_id: str, user_message: str,
                           memory_context: dict | None, tag: str) -> str:
    """
    If the user's message explicitly references a session-recommended item
    (ordinal or name) that differs from the article M3 resolved, trust the
    explicit reference. Otherwise returns article_id unchanged.
    """
    session_id = (memory_context or {}).get("session_id")
    hits = _session_reference_hits(user_message, session_id)
    if not hits:
        return article_id
    hit = hits[0]
    hid = str(hit.get("article_id", ""))
    if hid and hid.lstrip("0") != str(article_id or "").lstrip("0"):
        print(f"  [{tag}] session-memory override: M3 resolved '{article_id or 'None'}' "
              f"but message references '{hit.get('prod_name')}' ({hid}).")
        return hid
    return article_id


# =====================================================================
# Outfit completion via Colour Harmony KB
# =====================================================================
# "show me an outfit with a black top" → primary item by relevance + a
# companion from the complementary garment group, selected by the Colour
# Harmony KB (complementary/analogous colour tables) instead of MMR.

_OUTFIT_KEYWORDS = (
    "outfit", "complete look", "complete the look", "goes with", "go with",
    "wear with", "pair with", "pair it", "to match", "matching set",
    "full look", "whole look",
)

_COMPANION_GROUP = {
    "Garment Upper body": "Garment Lower body",
    "Garment Lower body": "Garment Upper body",
    "Garment Full body":  "Garment Upper body",   # dress → layering piece
    "Shoes":              "Garment Lower body",
}


def _detect_outfit_intent(user_message: str) -> bool:
    msg = (user_message or "").lower()
    return any(kw in msg for kw in _OUTFIT_KEYWORDS)


def _find_outfit_companion(primary: dict, articles_df, exclude_ids: list) -> dict | None:
    """
    Picks a companion item for the primary result: different garment group,
    colour chosen from the primary colour's harmony partners. Candidates are
    ranked by filtered-FAISS relevance + harmony_score. Returns a result dict
    shaped like a Phase-5 selection (article_id/metadata/final_score) plus
    the KB harmony_note, or None when no companion is available.
    """
    meta   = primary["metadata"]
    colour = str(meta.get("colour_group_name", "")).strip()
    group  = str(meta.get("product_group_name", "")).strip()
    target_group   = _COMPANION_GROUP.get(group, "Garment Upper body")
    partners, note = kb_retriever.harmony_partners(colour)

    mask = articles_df["product_group_name"].astype(str).str.strip() == target_group
    if partners:
        pmask = mask & articles_df["colour_group_name"].astype(str).str.strip().isin(partners)
        if pmask.sum() >= 5:   # keep the colour restriction only if enough choice remains
            mask = pmask
    ids  = articles_df.loc[mask, "article_id"].astype(str).str.zfill(10)
    excl = {str(e).zfill(10) for e in (exclude_ids or [])}
    excl.add(str(primary["article_id"]).zfill(10))
    allowed = [i for i in ids.tolist() if i not in excl]
    if not allowed:
        return None

    qtext = (f"{' '.join(partners[:3])} {target_group.replace('Garment ', '').lower()} "
             f"to wear with a {colour} {meta.get('product_type_name', '')}").strip()
    vec      = clip_encoder.encode_text(qtext)
    selector = faiss_db.build_id_selector(allowed)
    cands = (faiss_db.search_multi([vec], top_k=10, selector=selector)
             if (vec is not None and selector is not None) else [])
    if not cands:
        cands = [(allowed[0], 0.0)]

    best = None
    for aid, score in cands:
        cmeta = _fetch_article(aid)
        if not cmeta:
            continue
        h = kb_retriever.harmony_score(colour, str(cmeta.get("colour_group_name", "")))
        total = score + h
        if best is None or total > best[0]:
            best = (total, aid, cmeta)
    if best is None:
        return None
    return {
        "article_id":   best[1],
        "metadata":     best[2],
        "final_score":  round(best[0], 4),
        "harmony_note": note,
    }


# =====================================================================
# Accuracy Reporter (terminal diagnostic after every catalog_search)
# =====================================================================

_FUZZY_GROUPS = {
    "product_type_name": {
        "Dress":           ["dress"],
        "Top":             ["top", "t-shirt", "vest top", "blouse"],
        "Trousers":        ["trousers", "jeans"],
        "Jacket":          ["jacket", "coat", "blazer"],
        "Sweater":         ["sweater", "jumper", "knitwear"],
        "Hoodie":          ["hoodie", "sweatshirt"],
        "Shirt":           ["shirt"],
        "Blouse":          ["blouse"],
        "Leggings/Tights": ["leggings/tights", "leggings", "tights"],
    },
    "colour_group_name": {
        "Dark Blue": ["dark blue", "blue", "navy blue"],
        "White":     ["white", "off white"],
        "Black":     ["black"],
        "Red":       ["red", "dark red"],
        "Grey":      ["grey", "gray", "dark grey"],
    },
    "index_group_name": {
        "Ladieswear": ["ladieswear"],
        "Menswear":   ["menswear"],
        "Divided":    ["divided"],
        "Children":   ["children", "baby"],
    },
}


def _attr_match(item_val: str, expected_val: str, field: str) -> bool:
    iv = str(item_val).strip().lower()
    ev = str(expected_val).strip().lower()
    if iv == ev:
        return True
    for members in _FUZZY_GROUPS.get(field, {}).values():
        if ev in members and iv in members:
            return True
    return False


def _print_accuracy(items: list, filters: dict):
    """Prints a per-item accuracy report to the terminal using the request's hard filters."""
    checkable = {k: v for k, v in filters.items()
                 if k not in ("price_max", "price_min") and v is not None}

    print("\n" + "=" * 60)
    print("  [ACCURACY] Catalog Search Recommendation Report")
    print("=" * 60)

    if not checkable:
        print("  [ACCURACY] No checkable filters in this request — skipping score.")
        print("=" * 60 + "\n")
        return

    print("  Expected attributes from request filters:")
    for k, v in checkable.items():
        print(f"    {k}: {v}")

    if not items:
        print("  [ACCURACY] No items returned — cannot score.")
        print("=" * 60 + "\n")
        return

    item_accs = []
    for idx, item in enumerate(items, 1):
        hits = 0
        print(f"\n  Item {idx}: {item.get('prod_name', '?')}")
        print(f"  {'Attribute':<30} {'Expected':<18} {'Got':<20} Result")
        print(f"  {'-'*75}")
        for field, exp_val in checkable.items():
            got     = str(item.get(field, "")).strip()
            matched = _attr_match(got, str(exp_val), field)
            tick    = "[PASS]" if matched else "[FAIL]"
            print(f"  {field:<30} {str(exp_val):<18} {got:<20} {tick}")
            if matched:
                hits += 1
        acc = hits / len(checkable)
        item_accs.append(acc)
        print(f"  Item accuracy: {acc:.0%}")

    avg_acc  = sum(item_accs) / len(item_accs)
    full_hit = any(a >= 1.0 - 1e-9 for a in item_accs)
    part_hit = any(a >= 0.5 for a in item_accs)
    status   = "FULL HIT" if full_hit else ("PARTIAL HIT" if part_hit else "MISS")

    print(f"\n  {'─'*60}")
    print(f"  Result          : {status}")
    print(f"  Avg Accuracy    : {avg_acc:.0%}  ({avg_acc*100:.1f}% of expected attrs matched)")
    print(f"  Full Hit        : {'YES' if full_hit else 'NO'}  (all expected attrs matched in >=1 item)")
    print(f"  Items returned  : {len(items)}")
    print("=" * 60 + "\n")


# =====================================================================
# Shared Helpers
# =====================================================================

def _fetch_article(article_id: str) -> dict | None:
    """Fetches a single article's metadata from the articles CSV by article_id."""
    articles_df = _load_articles_priced()
    try:
        match = articles_df[articles_df['article_id'] == int(article_id)]
    except (ValueError, TypeError):
        return None
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _format_article_for_response(metadata: dict) -> dict:
    """Formats raw CSV metadata into a clean response dict for the API."""
    price = metadata.get("price")
    try:
        # NaN (never-purchased articles) and non-numeric values become None
        price = round(float(price), 2) if price == price and price is not None else None
    except (TypeError, ValueError):
        price = None
    return {
        "article_id":               str(metadata.get("article_id", "")).zfill(10),
        "prod_name":                metadata.get("prod_name", "Unknown"),
        "product_type_name":        metadata.get("product_type_name", "Unknown"),
        "product_group_name":       metadata.get("product_group_name", "Unknown"),
        "colour_group_name":        metadata.get("colour_group_name", "Unknown"),
        "department_name":          metadata.get("department_name", "Unknown"),
        "index_group_name":         metadata.get("index_group_name", "Unknown"),
        "detail_desc":              metadata.get("detail_desc", ""),
        "graphical_appearance_name":metadata.get("graphical_appearance_name", "Unknown"),
        "price":                    price,
    }


def _resolve_article_metadata(
    article_id: str,
    retrieval_input: dict,
    ctx_key: str = "context_article",
) -> dict | None:
    """
    Resolves item metadata for lookup handlers, in priority order:
      1. Local articles catalog (richest data: department, prices, …)
      2. payload[ctx_key] — full item data M3's session memory captured at
         recommendation time
      3. items_in_context — matching item from the current dialogue state
    2 and 3 let M2 answer conversational follow-ups even when the article
    isn't in the local catalog sample.
    """
    metadata = _fetch_article(article_id) if article_id else None
    if metadata:
        return metadata

    ctx = (retrieval_input.get("payload") or {}).get(ctx_key)
    if not ctx:
        for item in (retrieval_input.get("items_in_context") or {}).values():
            if item and str(item.get("article_id", "")).lstrip("0") == str(article_id).lstrip("0"):
                ctx = item
                break
    if not ctx:
        return None

    print(f"  [memory] Article {article_id} not in local catalog — using M3 session-memory data.")
    return {k: v for k, v in ctx.items() if v is not None}


def _call_llm(prompt: str) -> str | None:
    """Calls the cloud LLM (Groq) for a natural language response."""
    return llm_generator._call_llm(prompt, max_tokens=250)


def _clarification_response(action: str, retrieval_input: dict, situation: str) -> dict:
    """
    Graceful reply when a lookup request can't be fulfilled (e.g. comparing
    when only one item was shown, or a reference to an unknown item).
    Returns success=True: M3 replaces response_text with the raw `error`
    string on success=False, which would surface a technical message in chat.
    """
    items_ctx = retrieval_input.get("items_in_context") or {}
    shown = [v.get("prod_name") for v in items_ctx.values() if v and v.get("prod_name")]
    shown_str = ", ".join(f"'{n}'" for n in shown) if shown else "none"

    print(f"  [{action}] CLARIFICATION: {situation} (items shown: {shown_str})")

    prompt = (
        f"You are a friendly fashion assistant. The customer {situation}. "
        f"Items shown to them so far: {shown_str}.\n"
        f"Write a brief, warm response (1-2 sentences) that explains the situation "
        f"in plain language and asks what they'd like to do next. "
        f"Never mention technical terms like payloads, IDs or errors."
    )
    if shown:
        fallback = (
            f"So far I've only shown you {shown_str} — would you like me to "
            f"find more items first?"
        )
    else:
        fallback = (
            "I haven't shown you any items yet. Tell me what you're looking "
            "for and I'll find some options!"
        )

    return {
        "action":              action,
        "success":             True,
        "response_text":       _call_llm(prompt) or fallback,
        "items":               [],
        "needs_clarification": True,
        "error":               None,
    }


def _vlm_verified_response(
    prompt: str,
    metadata: dict,
    article_id: str,
    skip_visual: bool = False,
) -> tuple[str | None, dict]:
    """
    Runs the full 3-layer hallucination guard (same pipeline as catalog
    search Phase 6) on a prompt-driven response:
        Layer 1 — knowledge-grounded self-reflection
        Layer 2 — CoVe fact-check questions + DeBERTa NLI
        Layer 3 — CLIPScore + ViLT VQA against the product image
                  (skipped when skip_visual — non-visual answers like care
                  instructions are not image descriptions)

    Returns (response, trail). response is None when generation failed or
    the visual gate never passed — callers fall back to a safe template.
    """
    return generator_loop.verify_prompted_response(
        prompt, article_id, metadata, skip_visual=skip_visual
    )


# Attribute topics whose answers don't describe the product's appearance —
# ViLT/CLIPScore would spuriously reject them (Layers 1+2 still verify).
_NON_VISUAL_TOPICS = {"material_and_care", "price", "availability", "sizing_and_fit"}


def _cached_image_path(article_id: str):
    """Local image path if already downloaded, else None (never downloads)."""
    aid = str(article_id).zfill(10)
    p = data_loader.image_cache_dir / aid[:3] / f"{aid}.jpg"
    return p if p.exists() else None


def _kb_fact(metadata: dict, user_message: str) -> str:
    """
    Grounding fact for lookup prompts. VLM-first (NOVELTY 5 evolution): when
    the product image is cached, the vision model captures the item's visual
    psychology automatically; otherwise falls back to the manual Fashion KB
    tables (colour psychology / occasion fit / Kansei).
    Empty string when neither source applies.
    """
    if _ABLATE_KB:
        return ""
    kansei = kb_retriever.detect_kansei_from_message(user_message or "")
    style  = kansei[0] if kansei else None
    aid    = str(metadata.get("article_id", "")).zfill(10)

    fact = visual_psychology_fact(
        aid, _cached_image_path(aid), metadata, style_word=style
    )
    if fact:
        return fact
    return kb_retriever.get_explanation(
        item_colour=metadata.get("colour_group_name", ""),
        item_type=metadata.get("product_type_name", ""),
        style_word=style,
    )


def _preference_context(memory_context: dict | None) -> str:
    """
    Short summary of the customer's strongest long-term preferences (from
    M3's memory) for personalising lookup answers. Empty string if none.
    """
    if not memory_context:
        return ""
    prefs = memory_context.get("long_term_preferences") or []
    strong = [
        f"{p['attribute_value']} ({p['attribute_name'].replace('_', ' ')})"
        for p in prefs
        if isinstance(p, dict) and p.get("weight", 0) > 0.5
           and p.get("attribute_value") and p.get("attribute_name")
    ][:3]
    if not strong:
        return ""
    return (
        f"\n\nThe customer's known preferences: {', '.join(strong)}. "
        f"Mention a preference only if it is directly relevant and consistent "
        f"with the item facts above."
    )


# =====================================================================
# HANDLER 1: catalog_search
# Triggered by: INITIAL_REQUEST, REFINEMENT  |  Strategy: FULL
# =====================================================================

def handle_catalog_search(retrieval_input: dict, memory_context: dict | None = None) -> dict:
    """
    6-phase search pipeline:

    Phase 1 — LLM Query Expansion + Multi-Vector CLIP Ensemble (NOVELTY 1)
    Phase 2 — Hard filter + boost / penalty / purchase_history scoring
    Phase 3 — Cross-Encoder Neural Re-ranking (MiniLM-BERT)
    Phase 4 — LLM Semantic Re-ranking (NOVELTY 2)
    Phase 5 — Thompson Sampling Diversity Bandit + MMR (NOVELTY 3)
    Phase 6 — Verified explanation generation with self-reflection gate (NOVELTY 4)
    """
    # --- Unpack request ---
    payload          = retrieval_input.get("payload", {})
    user_message     = retrieval_input.get("user_message", "")
    exclude_ids      = retrieval_input.get("exclude_ids", [])
    filters          = payload.get("filters", {})
    boosts           = payload.get("preference_boosts", [])
    penalties        = payload.get("penalties", {})
    soft_constraints = payload.get("soft_constraints", {})
    purchase_hints   = payload.get("purchase_history_hints", {})

    # --- Resolve requested number of items ---
    # Priority: payload.num_items > payload.quantity (M3 LLM entity extraction)
    #           > parsed from user_message > default 2
    # Default is 2 (not 1): keeps comparison follow-ups, choice-based feedback
    # and the diversity bandit's reject/keep signal possible, while only
    # doubling per-item guard latency.
    # Capped at 6 to keep latency reasonable (each item runs VLM + LLM)
    _MAX_ITEMS = 6
    _DEFAULT_ITEMS = 2
    raw_qty = payload.get("num_items") or payload.get("quantity")
    try:
        num_items = int(raw_qty) if raw_qty else 0
    except (TypeError, ValueError):
        num_items = 0
    if num_items <= 0:
        # Parse user message for explicit quantity ("show me 4 dresses", "give 3 options")
        import re as _re
        _qty_match = _re.search(r'\b([1-6])\s*(items?|options?|dresses?|outfits?|products?|tops?|pairs?)\b', user_message.lower())
        num_items = int(_qty_match.group(1)) if _qty_match else _DEFAULT_ITEMS
    num_items = max(1, min(num_items, _MAX_ITEMS))
    print(f"  [catalog_search] Requested items: {num_items}")

    print(f"  [catalog_search] Filters: {filters}")
    print(f"  [catalog_search] Soft constraints: {soft_constraints}")
    print(f"  [catalog_search] Purchase hints — dominant: "
          f"{purchase_hints.get('dominant_colour')}/{purchase_hints.get('dominant_type')}, "
          f"budget: {purchase_hints.get('budget_tier')}")
    print(f"  [catalog_search] Exclude IDs: {exclude_ids}")

    # --- Kansei style detection (NOVELTY 5 — Improvement 3) ---
    # Scans raw user message for emotional vocabulary (Nagamachi, 1995) so KB
    # scoring fires even when soft_constraints["style"] is absent.
    detected_kansei = kb_retriever.detect_kansei_from_message(user_message)
    inferred_style  = soft_constraints.get("style") or (detected_kansei[0] if detected_kansei else None)
    if detected_kansei and not soft_constraints.get("style"):
        print(f"  [KB] Kansei detected from message: {detected_kansei} → style='{inferred_style}'")

    # ---------------------------------------------------------------
    # PHASE 1 — LLM Query Expansion + Multi-Vector CLIP Ensemble
    # ---------------------------------------------------------------
    filter_terms = " ".join(str(v) for v in filters.values() if not isinstance(v, (int, float)))
    soft_terms   = " ".join(str(v) for v in soft_constraints.values() if v)

    # NOVELTY 5: inject psychology KB context into CLIP search text
    if _ABLATE_KB:
        kb_query_context = ""
        clip_terms = ""
    else:
        kb_query_context = kb_retriever.get_context(
            occasion=soft_constraints.get("occasion"),
            style_word=inferred_style,
        )
        clip_terms = kb_retriever.get_clip_terms(
            occasion=soft_constraints.get("occasion"),
            style_word=inferred_style,
        )
    if kb_query_context:
        print(f"  [KB] Query context injected: {kb_query_context[:80]}...")
    if clip_terms:
        print(f"  [KB] CLIP visual terms: {clip_terms[:80]}...")

    base_search_text = (
        f"{user_message} {filter_terms} {soft_terms} {kb_query_context} {clip_terms}".strip()
        or " ".join(str(v) for v in filters.values())
    )
    print(f"  [catalog_search] Base search text: '{base_search_text}'")

    articles_df = _load_articles_priced()

    # --- Follow-up fast path (M3 session memory) ---
    # M3 marks REFINEMENT turns by putting "new_changes" into memory_context
    # and sends session_id alongside. When the session's cached candidate
    # pool still satisfies the merged filters, the follow-up is resolved from
    # it directly — skipping query expansion (LLM), CLIP ensemble and FAISS.
    session_id  = (memory_context or {}).get("session_id")
    is_followup = bool((memory_context or {}).get("new_changes")) or bool(
        (memory_context or {}).get("previous_constraints"))
    prefiltered = False
    from_cache  = False
    candidates  = []

    if session_id and is_followup:
        survivors = _followup_pool_from_cache(
            session_id, articles_df, filters, exclude_ids,
            min_needed=max(num_items, 3),
        )
        if survivors is not None:
            candidates  = survivors
            from_cache  = True
            prefiltered = bool(filters)   # filters re-applied inside the helper
            print(f"  [follow-up] Resolved from session cache: {len(candidates)} "
                  f"candidates — skipping query expansion + FAISS re-retrieval.")

    if from_cache:
        # One local CLIP vector is still needed for MMR similarity in Phase 5
        _vec = clip_encoder.encode_text(base_search_text)
        query_vectors = [_vec] if _vec is not None else []
    else:
        if _ABLATE_ENSEMBLE:
            expanded_queries = [base_search_text]   # ablation: single-vector baseline
        else:
            expanded_queries = llm_generator.expand_query(base_search_text)
        query_vectors    = [clip_encoder.encode_text(q) for q in expanded_queries if q]

        if not query_vectors:
            return {"action": "catalog_search", "success": False,
                    "response_text": "I couldn't process your search request.",
                    "items": [], "error": "CLIP encoding failed"}

        # --- Filtered FAISS: hard filters applied at index level (pool 50 → 15) ---
        # Allowed ids are computed vectorised over the full catalogue, then FAISS
        # only scans those rows via an IDSelector. Every candidate is already
        # filter-valid, so the pool shrinks to 15 without losing valid items.
        if filters:
            allowed_ids = _hard_filter_allowed_ids(articles_df, filters, exclude_ids)
            selector    = faiss_db.build_id_selector(allowed_ids)
            if selector is not None:
                candidates  = faiss_db.search_multi(query_vectors, top_k=15, selector=selector)
                prefiltered = bool(candidates)
                print(f"  [catalog_search] Filtered FAISS: {len(allowed_ids):,} "
                      f"filter-valid articles → {len(candidates)} candidates")

        if not candidates:
            if filters:
                print("  [catalog_search] Filtered FAISS empty/unavailable — "
                      "falling back to unfiltered top-50 search.")
            candidates = faiss_db.search_multi(query_vectors, top_k=50)
        if not candidates:
            return {"action": "catalog_search", "success": False,
                    "response_text": "I couldn't find any items matching your search.",
                    "items": [], "error": "No FAISS results"}

    # ---------------------------------------------------------------
    # PHASE 2 — Hard filter + boost / penalty / purchase history scoring
    # (hard-filter checks skipped when the pool was pre-filtered at index level)
    # ---------------------------------------------------------------
    filtered_results = []

    for article_id, faiss_score in candidates:
        if article_id in exclude_ids or article_id.lstrip('0') in exclude_ids:
            continue

        try:
            article_row = articles_df[articles_df['article_id'] == int(article_id)]
        except (ValueError, TypeError):
            continue

        if article_row.empty:
            continue

        metadata = article_row.iloc[0].to_dict()

        # Hard filters — all must pass (already guaranteed when the FAISS
        # pool was pre-filtered at index level)
        if not prefiltered:
            passes_filters = True
            for filter_key, filter_value in filters.items():
                if filter_key in ("price_max", "price_min"):
                    item_price = metadata.get("price")
                    if item_price is None or item_price != item_price:  # NaN check
                        continue  # unknown price — don't exclude the item
                    if filter_key == "price_max" and item_price > filter_value:
                        passes_filters = False; break
                    if filter_key == "price_min" and item_price < filter_value:
                        passes_filters = False; break
                else:
                    if str(metadata.get(filter_key, "")).strip().lower() != str(filter_value).strip().lower():
                        passes_filters = False; break

            if not passes_filters:
                continue

        # Penalty score
        penalty_score = sum(
            0.3
            for penalty_key, penalty_values in penalties.items()
            for pv in penalty_values
            if str(metadata.get(penalty_key, "")).strip().lower() == str(pv).strip().lower()
        )

        # Preference boost score
        boost_score = sum(
            boost.get("weight", 0.0)
            for boost in boosts
            if str(metadata.get(boost.get("attribute", ""), "")).strip().lower()
               == str(boost.get("value", "")).strip().lower()
        )

        # Purchase history collaborative score
        # CF model (trained on 185,037 transactions) scores at item level.
        # Falls back to rule-based scoring if model files are not yet loaded.
        if cf_scorer._loaded and not _ABLATE_CF:
            history_score = cf_scorer.score(article_id, purchase_hints, articles_df)
        else:
            history_score = 0.0
            if purchase_hints:
                item_colour = str(metadata.get('colour_group_name', '')).strip()
                item_type   = str(metadata.get('product_type_name', '')).strip()
                item_price  = float(metadata.get('price') or 0)

                top_colours = purchase_hints.get('top_colours') or []
                if item_colour in top_colours:
                    history_score += 0.12 * (1 - top_colours.index(item_colour) / max(len(top_colours), 1))

                if item_type in (purchase_hints.get('top_product_types') or []):
                    history_score += 0.08

                price_range = purchase_hints.get('preferred_price_range')
                if price_range and len(price_range) == 2 and price_range[0] <= item_price <= price_range[1]:
                    history_score += 0.08

        # Psychology KB score (NOVELTY 5)
        kb_score = 0.0 if _ABLATE_KB else kb_retriever.score(
            item_colour=metadata.get("colour_group_name", ""),
            item_type=metadata.get("product_type_name", ""),
            item_appearance=metadata.get("graphical_appearance_name", ""),
            occasion=soft_constraints.get("occasion"),
            style_word=inferred_style,
            index_group_name=metadata.get("index_group_name", ""),
        )

        filtered_results.append({
            "article_id":  article_id,
            "metadata":    metadata,
            "faiss_score": faiss_score,
            "final_score": faiss_score + boost_score - penalty_score + history_score + kb_score,
        })

    filtered_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Fallback: if all candidates were filtered out, use top raw FAISS results
    if not filtered_results:
        print("  [catalog_search] Hard filters eliminated all candidates. Falling back to top FAISS results.")
        for article_id, faiss_score in candidates[:10]:
            if article_id not in exclude_ids:
                meta = _fetch_article(article_id)
                if meta:
                    filtered_results.append({
                        "article_id":  article_id,
                        "metadata":    meta,
                        "faiss_score": faiss_score,
                        "final_score": faiss_score,
                    })

    # Cache the scored pool for follow-up turns. Full runs only — a cache-
    # answered follow-up keeps the original pool's breadth so consecutive
    # refinements don't progressively shrink it.
    if session_id and not from_cache and filtered_results:
        _session_pool_put(
            session_id,
            [(r["article_id"], r["faiss_score"]) for r in filtered_results],
        )
        print(f"  [follow-up] Cached {len(filtered_results)} scored candidates "
              f"for session {str(session_id)[:12]}")

    # ---------------------------------------------------------------
    # PHASE 3 — Cross-Encoder Neural Re-ranking (MiniLM-BERT)
    # ---------------------------------------------------------------
    print(f"  [catalog_search] Neural cross-encoder scoring top-{min(len(filtered_results), 20)} candidates...")
    neural_reranked = cross_encoder_reranker.rerank(
        query=base_search_text,
        candidates=filtered_results,
        top_k=20,
    )

    # Prefetch product images for the top candidates in the background.
    # The Kaggle per-file download costs seconds each; starting it now
    # overlaps the download with Phases 4-5 so it's off the guard's
    # critical path. get_image() is a no-op for already-cached images.
    # 8 workers so every candidate starts immediately — otherwise the two
    # items MMR eventually picks can sit queued behind unneeded downloads.
    _img_pool = ThreadPoolExecutor(max_workers=8)
    _img_futures = {
        r["article_id"]: _img_pool.submit(data_loader.get_image, r["article_id"])
        for r in neural_reranked[:8]
    }

    # ---------------------------------------------------------------
    # PHASE 4 — LLM Semantic Re-ranking (NOVELTY 2)
    # ---------------------------------------------------------------
    # NOVELTY 5: inject KB psychology context into LLM reranking
    kb_rerank_context = kb_retriever.get_context(
        occasion=soft_constraints.get("occasion"),
        style_word=inferred_style,
    )
    enriched_soft_constraints = dict(soft_constraints)
    if kb_rerank_context:
        enriched_soft_constraints["kb_psychology"] = kb_rerank_context

    print("  [catalog_search] LLM semantic re-ranking top-8 from neural stage...")
    reranked_results = llm_generator.rerank_candidates(
        user_message=user_message,
        candidates=neural_reranked,
        soft_constraints=enriched_soft_constraints,
        purchase_hints=purchase_hints,
    )

    # Ensure prefetch covers every candidate MMR can pick from — the LLM
    # rerank may promote items outside the Phase-3 top-8.
    for r in reranked_results[:8]:
        if r["article_id"] not in _img_futures:
            _img_futures[r["article_id"]] = _img_pool.submit(
                data_loader.get_image, r["article_id"]
            )

    # ---------------------------------------------------------------
    # PHASE 5 — Thompson Sampling Diversity Bandit + MMR (NOVELTY 3)
    # ---------------------------------------------------------------
    # Derive implicit feedback signals from session context:
    #   exclude_ids      → rejected items → more diversity → β increases
    #   items_in_context → kept items     → more relevance → α increases
    items_ctx      = retrieval_input.get("items_in_context") or {}
    retained_count = sum(1 for k in ("item_a", "item_b") if items_ctx.get(k))

    adaptive_lambda = diversity_bandit.sample_lambda(
        exclude_count=len(exclude_ids),
        retained_count=retained_count,
    )
    print(f"  [catalog_search] MMR with Thompson Sampling λ={adaptive_lambda:.3f}...")

    # --- Outfit completion (Colour Harmony KB) ---
    # Outfit intent → primary item by relevance + a colour-harmonious
    # companion from the complementary garment group, instead of two
    # same-category alternatives via MMR.
    outfit_mode = _detect_outfit_intent(user_message)
    companion   = None
    if outfit_mode and reranked_results:
        primary   = reranked_results[0]
        companion = _find_outfit_companion(primary, articles_df, exclude_ids)
        if companion:
            top_results = [primary, companion]
            print(f"  [outfit] Pairing '{primary['metadata'].get('prod_name')}' "
                  f"({primary['metadata'].get('colour_group_name')}) with "
                  f"'{companion['metadata'].get('prod_name')}' "
                  f"({companion['metadata'].get('colour_group_name')}) — colour-harmony KB")
        else:
            print("  [outfit] No harmonious companion found — standard selection.")

    if not (outfit_mode and companion):
        top_results = faiss_db.mmr_select(
            candidates=reranked_results,
            query_vector=query_vectors[0] if query_vectors else None,
            top_k=num_items,
            lambda_param=adaptive_lambda,
        ) or reranked_results[:num_items]

    # Selected items are known now — cancel any not-yet-started prefetch
    # downloads for candidates that didn't make the final cut.
    _selected_ids = {r["article_id"] for r in top_results}
    for _aid, _f in _img_futures.items():
        if _aid not in _selected_ids:
            _f.cancel()

    # Colour harmony diagnostic for the selected pair (NOVELTY 5 — Improvement 4)
    if len(top_results) >= 2:
        c1 = top_results[0]["metadata"].get("colour_group_name", "")
        c2 = top_results[1]["metadata"].get("colour_group_name", "")
        h  = kb_retriever.harmony_score(c1, c2)
        label = "complementary" if h > 0.10 else ("analogous" if h > 0 else ("clashing" if h < 0 else "neutral"))
        print(f"  [KB] Colour harmony: {c1} × {c2} = {h:+.2f} ({label})")

    # ---------------------------------------------------------------
    # PHASE 6 — Verified explanation generation (parallel per item)
    # ---------------------------------------------------------------
    # Each item's guard is a chain of sequential LLM/model calls, so items
    # are independent of each other — verifying them in parallel threads
    # divides Phase 6 latency by roughly the item count. Output preserves
    # MMR order. Note: guard log lines from different items may interleave.
    def _verify_item(result: dict) -> dict:
        aid  = result["article_id"]
        meta = result["metadata"]

        # Wait for this item's background image prefetch (started after
        # Phase 3) so the guard never downloads the same file concurrently.
        _fut = _img_futures.get(aid)
        if _fut is not None:
            try:
                _fut.result(timeout=180)
            except Exception:
                pass  # guard handles a missing image gracefully (Layer 3 skip)

        # NOVELTY 5 evolution: VLM-first explanation fact — the vision model
        # captures the item's visual psychology from the (just-prefetched)
        # image; the manual Kansei KB remains the fallback.
        kb_explanation_fact = "" if _ABLATE_KB else visual_psychology_fact(
            str(aid).zfill(10), _cached_image_path(aid), meta,
            style_word=inferred_style,
        ) or kb_retriever.get_explanation(
            item_colour=meta.get("colour_group_name", ""),
            item_type=meta.get("product_type_name", ""),
            occasion=soft_constraints.get("occasion"),
            style_word=inferred_style,
        )
        if kb_explanation_fact:
            print(f"  [KB] Explanation fact: {kb_explanation_fact[:80]}...")

        explanation, verification_trail = generator_loop.generate_faithful_explanation(
            article_id=aid,
            kb_fact=kb_explanation_fact,
        )

        item_response = _format_article_for_response(meta)
        item_response["explanation"]         = explanation
        item_response["score"]               = result["final_score"]
        item_response["verification_trail"]  = verification_trail
        return item_response

    if len(top_results) > 1:
        print(f"  [catalog_search] Phase 6: verifying {len(top_results)} items in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(top_results), 3)) as pool:
            response_items = list(pool.map(_verify_item, top_results))
    else:
        response_items = [_verify_item(r) for r in top_results]
    _img_pool.shutdown(wait=False)

    # Natural language summary — handles any number of items
    n = len(response_items)
    if outfit_mode and companion and n == 2:
        _a, _b = response_items
        _pair_note = companion.get("harmony_note") or (
            f"{_a['colour_group_name']} and {_b['colour_group_name']} pair well together"
        )
        summary = (
            f"Here's a complete outfit for you: the {_a['prod_name']} in "
            f"{_a['colour_group_name']}, paired with the {_b['prod_name']} in "
            f"{_b['colour_group_name']}. {_pair_note}."
        )
    elif n == 0:
        summary = "I couldn't find any items matching all your criteria."
    elif n == 1:
        summary = (
            f"I found a great match: the {response_items[0]['prod_name']} "
            f"in {response_items[0]['colour_group_name']}."
        )
    elif n == 2:
        summary = (
            f"Based on your search, I found two great options: "
            f"the {response_items[0]['prod_name']} in {response_items[0]['colour_group_name']} "
            f"and the {response_items[1]['prod_name']} in {response_items[1]['colour_group_name']}."
        )
    else:
        item_list = ", ".join(
            f"the {item['prod_name']} in {item['colour_group_name']}"
            for item in response_items[:-1]
        )
        last = f"the {response_items[-1]['prod_name']} in {response_items[-1]['colour_group_name']}"
        summary = f"Based on your search, I found {n} great options: {item_list}, and {last}."

    _print_accuracy(response_items, filters)

    # Remember this recommendation's ordered items so later ordinal/name
    # follow-ups resolve correctly even if M3's dialogue state narrows.
    if session_id and response_items:
        _session_items_put(session_id, [
            {"article_id": it["article_id"], "prod_name": it["prod_name"]}
            for it in response_items
        ])

    return {
        "action":        "catalog_search",
        "success":       len(response_items) > 0,
        "response_text": summary,
        "items":         response_items,
        "error":         None,
    }


# =====================================================================
# HANDLER 2: item_attribute_lookup
# Triggered by: ATTRIBUTE_QUESTION  |  Strategy: PARTIAL
# =====================================================================

def handle_attribute_lookup(retrieval_input: dict, memory_context: dict | None = None) -> dict:
    """Fetches an item and answers a specific attribute question about it."""
    payload          = retrieval_input.get("payload", {})
    user_message     = retrieval_input.get("user_message", "")
    article_id       = payload.get("article_id", "")
    attribute_topic  = payload.get("attribute_topic", "general_details")

    article_id = _override_from_session(article_id, user_message, memory_context, "attribute_lookup")
    print(f"  [attribute_lookup] Article: {article_id}, Topic: {attribute_topic}")

    metadata = _resolve_article_metadata(article_id, retrieval_input)
    if not metadata:
        return _clarification_response(
            "item_attribute_lookup", retrieval_input,
            "asked about an item's details, but the item they mean couldn't be identified",
        )

    item_info   = _format_article_for_response(metadata)
    detail_desc = metadata.get("detail_desc", "No detailed description available.")

    kb_fact  = _kb_fact(metadata, user_message)
    kb_line  = f"- Fashion knowledge (verified): {kb_fact}\n" if kb_fact else ""
    pref_ctx = _preference_context(memory_context)

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Here are the item details:\n"
        f"- Product: {item_info['prod_name']}\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Appearance: {item_info['graphical_appearance_name']}\n"
        f"- Description: {detail_desc}\n"
        f"{kb_line}{pref_ctx}\n\n"
        f"The customer is specifically asking about: {attribute_topic.replace('_', ' ')}.\n"
        f"Answer their question in 1-3 sentences using ONLY the information above. "
        f"If the information isn't available in the details, say so honestly."
    )

    # Non-visual topics (care, price, stock, sizing): the answer isn't a
    # description of the image, so the visual gate would reject it spuriously.
    skip_visual = attribute_topic in _NON_VISUAL_TOPICS
    print(f"  [attribute_lookup] Running hallucination guard "
          f"(visual gate: {'skipped — non-visual topic' if skip_visual else 'on'})...")
    response_text, verification = _vlm_verified_response(
        prompt, metadata, article_id, skip_visual=skip_visual
    )

    if not response_text:
        response_text = (
            f"The {item_info['prod_name']} is a {item_info['colour_group_name']} "
            f"{item_info['product_type_name']} from the {item_info['department_name']} department."
        )
        if attribute_topic == "material_and_care" and detail_desc:
            response_text = f"Here are the details: {detail_desc}"

    return {
        "action":        "item_attribute_lookup",
        "success":       True,
        "response_text": response_text,
        "items":         [item_info],
        "verification":  verification,
        "error":         None,
    }


# =====================================================================
# HANDLER 3: item_compare
# Triggered by: COMPARISON  |  Strategy: PARTIAL
# =====================================================================

def handle_item_compare(retrieval_input: dict, memory_context: dict | None = None) -> dict:
    """
    Compares two items on a specified dimension. Runs VLM verification in
    two phases — once anchored on item_a, once on item_b — to catch
    hallucinations about either item independently.
    """
    payload              = retrieval_input.get("payload", {})
    user_message         = retrieval_input.get("user_message", "")
    article_id_a         = payload.get("article_id_a") or ""
    article_id_b         = payload.get("article_id_b") or ""
    comparison_dimension = payload.get("comparison_dimension", "overall")
    preference_weights   = payload.get("preference_weights", {})

    # Session-memory resolution: explicit ordinals/names in the message win
    # over (or fill in for) M3's resolution — fixes "compare the first and
    # second one" after M3's dialogue state narrowed to a single item.
    _hits = _session_reference_hits(user_message, (memory_context or {}).get("session_id"))
    if len(_hits) >= 2:
        _ha, _hb = str(_hits[0]["article_id"]), str(_hits[1]["article_id"])
        if {_ha.lstrip("0"), _hb.lstrip("0")} != {str(article_id_a).lstrip("0"),
                                                  str(article_id_b).lstrip("0")}:
            print(f"  [item_compare] session-memory resolution: comparing "
                  f"'{_hits[0]['prod_name']}' vs '{_hits[1]['prod_name']}'")
            article_id_a, article_id_b = _ha, _hb
    elif len(_hits) == 1 and article_id_a and not article_id_b:
        _hb = str(_hits[0]["article_id"])
        if _hb.lstrip("0") != str(article_id_a).lstrip("0"):
            print(f"  [item_compare] session-memory resolution: item_b filled "
                  f"with '{_hits[0]['prod_name']}'")
            article_id_b = _hb

    if not article_id_a or not article_id_b:
        return _clarification_response(
            "item_compare", retrieval_input,
            "asked to compare two items, but fewer than two items have been shown so far",
        )

    print(f"  [item_compare] Comparing {article_id_a} vs {article_id_b} on '{comparison_dimension}'")

    meta_a = _resolve_article_metadata(article_id_a, retrieval_input, ctx_key="context_article_a")
    meta_b = _resolve_article_metadata(article_id_b, retrieval_input, ctx_key="context_article_b")

    if not meta_a or not meta_b:
        return _clarification_response(
            "item_compare", retrieval_input,
            "asked to compare two items, but one of them couldn't be identified",
        )

    item_a = _format_article_for_response(meta_a)
    item_b = _format_article_for_response(meta_b)

    pref_str = ""
    if preference_weights:
        pref_parts = [f"{k.replace('_', ' ')}: {v:.0%} importance" for k, v in preference_weights.items()]
        pref_str = f"\n\nThe customer's preferences: {', '.join(pref_parts)}."
    pref_str += _preference_context(memory_context)

    # KB grounding: psychology facts per item + colour pairing note
    kb_a = _kb_fact(meta_a, user_message)
    kb_b = _kb_fact(meta_b, user_message)
    kb_a_line = f"  - Fashion knowledge (verified): {kb_a}\n" if kb_a else ""
    kb_b_line = f"  - Fashion knowledge (verified): {kb_b}\n" if kb_b else ""
    harmony = kb_retriever.harmony_score(
        item_a["colour_group_name"], item_b["colour_group_name"]
    )
    harmony_str = ""
    if harmony > 0:
        harmony_str = (f"\nNote: {item_a['colour_group_name']} and "
                       f"{item_b['colour_group_name']} pair well together visually.")
    elif harmony < 0:
        harmony_str = (f"\nNote: {item_a['colour_group_name']} and "
                       f"{item_b['colour_group_name']} tend to clash visually.")

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Compare these two items on the dimension of '{comparison_dimension}':\n\n"
        f"ITEM A — {item_a['prod_name']}:\n"
        f"  - Type: {item_a['product_type_name']}\n"
        f"  - Colour: {item_a['colour_group_name']}\n"
        f"  - Department: {item_a['department_name']}\n"
        f"  - Description: {meta_a.get('detail_desc', 'N/A')}\n"
        f"{kb_a_line}\n"
        f"ITEM B — {item_b['prod_name']}:\n"
        f"  - Type: {item_b['product_type_name']}\n"
        f"  - Colour: {item_b['colour_group_name']}\n"
        f"  - Department: {item_b['department_name']}\n"
        f"  - Description: {meta_b.get('detail_desc', 'N/A')}\n"
        f"{kb_b_line}"
        f"{pref_str}{harmony_str}\n\n"
        f"Give a clear, concise comparison (2-4 sentences). "
        f"State which item is better for the customer and why, based on the comparison dimension."
    )

    # Phase 1: full 3-layer guard anchored on item_a
    print(f"  [item_compare] Phase 1 — 3-layer hallucination guard (anchor: item_a)...")
    response_text, verification = _vlm_verified_response(prompt, meta_a, article_id_a)

    # Phase 2: additional VLM pass anchored on item_b
    if response_text:
        image_path_b = data_loader.get_image(article_id_b)
        if image_path_b and image_path_b.exists():
            print(f"  [item_compare] Phase 2 — additional VLM verification (anchor: item_b)...")
            is_valid_b, reason_b = blip_verifier.verify(str(image_path_b), response_text)
            if is_valid_b:
                print(f"  [item_compare] Item B VLM PASS: {reason_b}")
                verification["item_b_vlm_verified"] = True
            else:
                print(f"  [item_compare] Item B VLM FAIL: {reason_b} — regenerating with item_b feedback")
                corrected = _call_llm(
                    prompt + f"\n\n[Visual check on Item B failed: {reason_b}. "
                             f"Ensure your description of Item B is visually accurate.]"
                )
                # Re-verify the corrected text — don't accept it blindly
                if corrected and blip_verifier.verify(str(image_path_b), corrected)[0]:
                    print(f"  [item_compare] Item B VLM PASS after correction")
                    response_text = corrected
                    verification["item_b_vlm_verified"] = True
                else:
                    print(f"  [item_compare] Item B still unverified — using template fallback")
                    response_text = None
                    verification["item_b_vlm_verified"] = False
        else:
            print(f"  [item_compare] Phase 2 — no image for item_b ({article_id_b}), skipping.")

    if not response_text:
        response_text = (
            f"Comparing the {item_a['prod_name']} ({item_a['colour_group_name']}) "
            f"and {item_b['prod_name']} ({item_b['colour_group_name']}) "
            f"on {comparison_dimension}: both are great options from their respective departments."
        )

    return {
        "action":        "item_compare",
        "success":       True,
        "response_text": response_text,
        "items":         [item_a, item_b],
        "verification":  verification,
        "error":         None,
    }


# =====================================================================
# HANDLER 4: explanation_generate
# Triggered by: EXPLANATION_WHY  |  Strategy: PARTIAL
# =====================================================================

def handle_explanation_generate(retrieval_input: dict, memory_context: dict | None = None) -> dict:
    """
    Generates a justified explanation for why an item was recommended,
    grounded in the user's matched_prefs and consistent with prior_claims.
    """
    payload        = retrieval_input.get("payload", {})
    user_message   = retrieval_input.get("user_message", "")
    article_id     = payload.get("article_id", "")
    prior_claims   = payload.get("prior_claims", [])
    matched_prefs  = payload.get("matched_prefs", [])

    article_id = _override_from_session(article_id, user_message, memory_context, "explanation_generate")
    print(f"  [explanation_generate] Article: {article_id}")
    print(f"  [explanation_generate] Prior claims: {len(prior_claims)}, Matched prefs: {len(matched_prefs)}")

    metadata = _resolve_article_metadata(article_id, retrieval_input)
    if not metadata:
        return _clarification_response(
            "explanation_generate", retrieval_input,
            "asked why an item was recommended, but the item they mean couldn't be identified",
        )

    item_info = _format_article_for_response(metadata)

    # Build prior claims context (do not contradict active claims)
    claims_str = ""
    active_claims = [c for c in prior_claims if c.get("status") == "active"]
    if active_claims:
        claims_parts = [f"- {c['claim_text']} (type: {c['claim_type']})" for c in active_claims]
        claims_str = (
            "\n\nIMPORTANT — You have already told the customer these facts (do NOT contradict them):\n"
            + "\n".join(claims_parts)
        )

    # Build matched preferences context
    prefs_str = ""
    if matched_prefs:
        prefs_parts = [
            f"- {p['attribute_name'].replace('_', ' ')}: {p['attribute_value']} (weight: {p['weight']:.0%})"
            for p in matched_prefs
        ]
        prefs_str = "\n\nThe customer's preferences that match this item:\n" + "\n".join(prefs_parts)

    kb_fact = _kb_fact(metadata, user_message)
    kb_line = f"- Fashion knowledge (verified): {kb_fact}\n" if kb_fact else ""
    pref_ctx = _preference_context(memory_context)

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Explain why we recommended the {item_info['prod_name']}:\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Description: {metadata.get('detail_desc', 'N/A')}\n"
        f"{kb_line}"
        f"{prefs_str}{pref_ctx}{claims_str}\n\n"
        f"Generate a warm, conversational 2-3 sentence explanation. "
        f"Base it on the matched preferences above. "
        f"Do NOT contradict any prior claims listed above."
    )

    print("  [explanation_generate] Running 3-layer hallucination guard...")
    response_text, verification = _vlm_verified_response(prompt, metadata, article_id)

    # ── NLI prior-claim consistency check ─────────────────────────────
    # The prompt asks the LLM not to contradict prior claims; this
    # verifies it actually obeyed, using the same DeBERTa NLI model as
    # CoVe. One regeneration attempt, then template fallback.
    if response_text and active_claims:
        def _contradicted(text: str) -> list[str]:
            bad = []
            for c in active_claims:
                ok, score = cove_verifier.check_claim_consistency(text, c["claim_text"])
                status = "PASS" if ok else "FAIL"
                print(f"  [Claim | NLI] [{status}] \"{c['claim_text'][:60]}\" "
                      f"contra_score={score:.3f}")
                if not ok:
                    bad.append(c["claim_text"])
            return bad

        contradicted = _contradicted(response_text)
        verification["prior_claims_checked"]    = len(active_claims)
        verification["prior_claims_consistent"] = not contradicted
        if contradicted:
            print("  [explanation_generate] Prior-claim contradiction — regenerating...")
            regenerated = _call_llm(
                prompt + "\n\n[Your previous answer contradicted these facts you "
                "already told the customer: " + "; ".join(contradicted) +
                ". Rewrite the explanation so it stays consistent with them.]"
            )
            if regenerated and not _contradicted(regenerated):
                response_text = regenerated
                verification["prior_claims_consistent"] = True
            else:
                response_text = None   # fall through to the safe template below

    if not response_text:
        if matched_prefs:
            reasons = [
                f"it matches your preference for {p['attribute_value']} {p['attribute_name'].replace('_', ' ')}"
                for p in matched_prefs[:3]
            ]
            response_text = f"We recommended the {item_info['prod_name']} because " + ", and ".join(reasons) + "."
        else:
            response_text = (
                f"The {item_info['prod_name']} is a great {item_info['colour_group_name']} "
                f"{item_info['product_type_name']} that fits your style."
            )

    return {
        "action":        "explanation_generate",
        "success":       True,
        "response_text": response_text,
        "items":         [item_info],
        "verification":  verification,
        "error":         None,
    }


# =====================================================================
# HANDLER 5: item_detail_lookup
# Triggered by: SELECTION_REFERENCE  |  Strategy: PARTIAL
# =====================================================================

def handle_item_detail_lookup(retrieval_input: dict, memory_context: dict | None = None) -> dict:
    """
    Fetches and presents all details for an item the user pointed at.
    m3 has already resolved a reference like 'the first one' to an article_id.
    """
    payload      = retrieval_input.get("payload", {})
    user_message = retrieval_input.get("user_message", "")
    article_id   = payload.get("article_id", "")

    article_id = _override_from_session(article_id, user_message, memory_context, "item_detail_lookup")
    print(f"  [item_detail_lookup] Article: {article_id}")

    metadata = _resolve_article_metadata(article_id, retrieval_input)
    if not metadata:
        return _clarification_response(
            "item_detail_lookup", retrieval_input,
            "asked for more details about an item, but the item they refer to "
            "doesn't match anything shown so far",
        )

    item_info   = _format_article_for_response(metadata)
    detail_desc = metadata.get("detail_desc", "")

    kb_fact  = _kb_fact(metadata, user_message)
    kb_line  = f"- Fashion knowledge (verified): {kb_fact}\n" if kb_fact else ""
    pref_ctx = _preference_context(memory_context)

    prompt = (
        f"You are a helpful fashion assistant. A customer asked: \"{user_message}\"\n\n"
        f"Present the full details of this item in a friendly, conversational way (3-4 sentences):\n"
        f"- Name: {item_info['prod_name']}\n"
        f"- Type: {item_info['product_type_name']}\n"
        f"- Colour: {item_info['colour_group_name']}\n"
        f"- Department: {item_info['department_name']}\n"
        f"- Category: {item_info['product_group_name']}\n"
        f"- Appearance: {item_info['graphical_appearance_name']}\n"
        f"- Description: {detail_desc}\n"
        f"{kb_line}{pref_ctx}\n\n"
        f"Be informative and enthusiastic. Highlight the key selling points."
    )

    print("  [item_detail_lookup] Running 3-layer hallucination guard...")
    response_text, verification = _vlm_verified_response(prompt, metadata, article_id)

    if not response_text:
        response_text = (
            f"Here are the details for the {item_info['prod_name']}: "
            f"It's a {item_info['colour_group_name']} {item_info['product_type_name']} "
            f"from the {item_info['department_name']} department. "
            f"{detail_desc}"
        )

    # Outfit tip: deterministic Colour Harmony pairing line when the user
    # asks what goes with the item ("what can I wear with this?").
    if response_text and _detect_outfit_intent(user_message):
        partners, note = kb_retriever.harmony_partners(item_info["colour_group_name"])
        if partners:
            response_text += (
                f" Style tip: {item_info['colour_group_name']} pairs beautifully "
                f"with {', '.join(partners[:3])}"
            )
            response_text += f" — {note[0].lower() + note[1:]}." if note else "."

    return {
        "action":        "item_detail_lookup",
        "success":       True,
        "response_text": response_text,
        "items":         [item_info],
        "verification":  verification,
        "error":         None,
    }


# =====================================================================
# HANDLER 6: No retrieval
# Triggered by: FEEDBACK, CHITCHAT  |  Strategy: NONE
# =====================================================================

def handle_no_retrieval(memory_context: dict) -> dict:
    """Handles turns where no retrieval is needed (FEEDBACK or CHITCHAT)."""

    # Clarification path: M3 could not resolve the user's request (e.g.
    # "compare the first and second one" when only one item was shown).
    if memory_context.get("needs_clarification"):
        reason = memory_context.get("clarification_reason", "")
        print(f"  [no_retrieval] CLARIFICATION: {reason}")
        prompt = (
            f"You are a friendly fashion assistant. The customer's last request "
            f"couldn't be fulfilled for this reason: {reason}\n"
            f"Write a brief, warm response (1-2 sentences) that explains the "
            f"situation in plain language and asks what they'd like to do next. "
            f"Never mention technical terms like payloads, IDs or errors."
        )
        fallback = (
            "I'm not quite sure which items you mean — I may not have shown "
            "enough options yet. Would you like me to find some more items first?"
        )
        return {
            "action":        None,
            "success":       True,
            "response_text": _call_llm(prompt) or fallback,
            "items":         [],
            "error":         None,
        }

    feedback = memory_context.get("feedback")

    if feedback:
        sentiment_score  = feedback.get("sentiment_score", 0.0)
        is_positive      = feedback.get("is_positive", False)
        feedback_type    = feedback.get("feedback_type", "neutral")
        item_name        = feedback.get("item_reacted_to", {}).get("prod_name", "that item")

        print(f"  [no_retrieval] FEEDBACK: sentiment={sentiment_score:.1f}, type={feedback_type}")

        # Multi-item awareness: M3 always attributes feedback to the first
        # item, but after a multi-item turn (e.g. an outfit pair) "I love it"
        # is ambiguous — acknowledge the whole set instead of silently
        # praising item 1. Uses M2's own session items cache.
        session_items = _session_items_get((memory_context or {}).get("session_id") or "") or []
        names = [it.get("prod_name") for it in session_items if it.get("prod_name")]
        multi = len(names) >= 2
        if multi:
            names_str = ", ".join(names[:-1]) + f" and the {names[-1]}"

        if is_positive and multi:
            print(f"  [no_retrieval] Feedback follows a {len(names)}-item turn — acknowledging the set.")
            prompt   = (
                f"You are a friendly fashion assistant. The customer just expressed positive feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) about the items you showed together: "
                f"the {names_str}. It's not clear which one (or all) they mean, so celebrate how well "
                f"the items work together (1-2 sentences) and ask whether they'd like to go with "
                f"everything or focus on one of them."
            )
            fallback = (
                f"So glad you love them! The {names_str} really do work well together. "
                f"Would you like to go with the whole set, or shall I tell you more about one of them?"
            )
        elif is_positive:
            prompt   = (
                f"You are a friendly fashion assistant. The customer just expressed positive feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) about the {item_name}. "
                f"Write a brief, warm response (1-2 sentences) congratulating their choice "
                f"and asking if they'd like to see similar items or proceed to purchase."
            )
            fallback = (
                f"Great choice! The {item_name} is an excellent pick. "
                f"Would you like to see similar items, or shall I help with anything else?"
            )
        elif multi:
            prompt   = (
                f"You are a friendly fashion assistant. The customer just expressed negative feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) after being shown these items together: "
                f"the {names_str}. It's not clear which one they dislike, so acknowledge their "
                f"reaction empathetically (1-2 sentences) and ask which item missed the mark or "
                f"what they'd prefer instead."
            )
            fallback = (
                f"I'm sorry those weren't quite right. Was it the {names_str.replace(', ', ' or the ').replace(' and the ', ' or the ')} "
                f"that missed the mark? Tell me what you'd prefer and I'll find something better!"
            )
        else:
            prompt   = (
                f"You are a friendly fashion assistant. The customer just expressed negative feedback "
                f"(sentiment: {sentiment_score:.1f}/1.0) about the {item_name}. "
                f"Write a brief, empathetic response (1-2 sentences) acknowledging their reaction "
                f"and offering to search for something different."
            )
            fallback = (
                f"I understand the {item_name} wasn't quite right. "
                f"Let me know what you'd prefer and I'll find something better for you!"
            )

        return {
            "action":        None,
            "success":       True,
            "response_text": _call_llm(prompt) or fallback,
            "items":         [],
            "error":         None,
        }

    # CHITCHAT path
    print("  [no_retrieval] CHITCHAT: Generating conversational response.")
    has_history = bool(memory_context.get("dialogue_state", {}).get("hard_constraints"))

    if has_history:
        prompt   = (
            "You are a friendly fashion assistant. The customer is making small talk "
            "during a shopping conversation. Respond briefly and warmly (1-2 sentences), "
            "then gently steer back to helping them find fashion items."
        )
        fallback = "Of course! I'm here to help. What kind of fashion items are you looking for today?"
    else:
        prompt   = (
            "You are a friendly fashion assistant. A new customer just greeted you. "
            "Welcome them warmly (1-2 sentences) and invite them to describe what "
            "kind of clothing or style they're looking for."
        )
        fallback = (
            "Welcome! I'm your fashion assistant. "
            "Tell me what you're looking for — a specific item, colour, or style — and I'll find the perfect match!"
        )

    return {
        "action":        None,
        "success":       True,
        "response_text": _call_llm(prompt) or fallback,
        "items":         [],
        "error":         None,
    }
