# m3_implementation/text_rag/core/personalized_ranker.py
#
# Selects which candidates to recommend, and records WHY each one was chosen.
#
# THE PROBLEM THIS SOLVES:
#   Hard filters (colour=Black, type=Trousers) narrow 41,794 articles to a few
#   hundred. Every one of them satisfies the request equally, so the filters
#   cannot decide which 2 to show. Previously that decision fell through to
#   Qdrant's ordering, which is identical for every user asking the same
#   question. This module makes the decision personal and, crucially, makes it
#   EXPLAINABLE: every component that fires emits a sentence containing a real
#   number, and those sentences travel with the item into the evidence bundle.
#
# TWO SIGNAL FAMILIES:
#   A. User fit      — the user's own purchase history and stated preferences.
#                      Different per user, so the same question yields
#                      different items for different people.
#   B. Buying stats  — how the wider customer base behaves toward this article
#                      (popularity, age-group affinity, repeat rate, trend).
#                      Same for everyone, but weighted differently per user.
#
# CONFIDENCE-ADAPTIVE BLENDING:
#   A shopper with 700 transactions gets user-fit weighted heavily. A brand-new
#   shopper has no history to fit, so the blend shifts toward buying stats
#   ("popular with people your age") rather than showing them nothing personal.
#   See _blend_weights().
#
# EVERY REASON IS A TEMPLATE, NEVER LLM TEXT:
#   Reason strings are built here from numbers, so they cannot hallucinate and
#   they double as verifiable facts for the hallucination checker.

import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from text_rag.db.article_stats import (
    AGE_BUCKETS, MIN_AGE_SUPPORT_ARTICLE, MIN_AGE_SUPPORT_GROUP,
)

# ── Component weights ─────────────────────────────────────────────────────────
# Group A (user fit) and Group B (buying stats) are normalised separately then
# blended by _blend_weights(), so tuning one family cannot silently swamp the
# other.

W_USER_FIT = {
    "colour_affinity":        0.30,
    "type_affinity":          0.22,
    "garment_affinity":       0.14,
    "pattern_affinity":       0.12,
    "section_affinity":       0.10,
    "price_fit":              0.20,
    "gender_fit":             0.14,
    "session_preference":     0.26,
}

W_BUYING_STATS = {
    "popularity":             0.26,
    "age_group_match":        0.24,
    "repeat_rate":            0.14,
    "recency_trend":          0.10,
}

# Semantic relevance is neither user fit nor social proof — it is "does this
# answer the question at all". It is added on top of the blend at a fixed
# weight so a personalised item can never outrank a clearly irrelevant one.
W_SEMANTIC = 0.35

# Dislikes are a hard demotion, applied after blending. This is a TOTAL cap per
# item, not a per-hit amount: a profile whose dislike list has drifted to cover
# many attributes must not be able to drive scores far negative. One disliked
# attribute and four disliked attributes both cost at most this much.
W_DISLIKE_PENALTY = 0.50

# Attributes that describe WHO a garment is for (age/gender segment) rather than
# what it looks like. A rejection is never evidence that the user dislikes an
# entire demographic segment, and treating it that way is actively harmful: with
# Ladieswear/Menswear/Divided all marked disliked, adult clothing gets demoted
# and Baby/Children rises to the top. Positive gender alignment is already
# handled by _c_gender_fit.
_DEMOGRAPHIC_ATTRS = {"index_group_name"}

# A value holding at least this share of the user's purchase history is treated
# as a staple and can never be penalised. Someone with 37.5% black purchases does
# not dislike black, whatever a stray feedback signal recorded.
PROTECTED_HISTORY_PCT = 8.0

# When the best remaining candidate has no reason to show, a candidate WITH a
# reason may be promoted ahead of it if it is within this margin. Keeps cards
# explainable without letting a weak item beat a clearly better one.
EXPLAINABILITY_MARGIN = 0.15

# A user needs at least this many purchases before their history is treated as
# fully reliable. Below it, weight shifts toward buying stats.
FULL_CONFIDENCE_PURCHASES = 60

# Minimum lift before an age-match reason is worth showing to the user.
AGE_LIFT_THRESHOLD = 1.15

# Popularity percentile below which no popularity reason is emitted.
POPULARITY_PCT_FLOOR = 60.0


@dataclass
class Component:
    """One scoring signal that fired for one candidate."""
    name:   str
    delta:  float           # contribution to the final score
    reason: Optional[str]   # user-facing sentence, or None if not worth showing


@dataclass
class ScoredItem:
    """A candidate article with its score and full reasoning trace."""
    article:    dict
    score:      float                  = 0.0
    components: list[Component]        = field(default_factory=list)

    @property
    def article_id(self) -> str:
        return str(self.article.get("article_id", ""))

    @property
    def has_reasons(self) -> bool:
        """True when at least one component produced a user-facing reason."""
        return any(c.reason for c in self.components if c.delta > 0)

    def reasons(self, limit: int = 3) -> list[str]:
        """
        Top reason strings, strongest contribution first.

        An item can win on silent signals alone (semantic relevance, gender fit,
        a price just outside the user's band). Showing that item with an empty
        "Why this for you" block looks broken, so a single honest fallback line
        is returned instead — it states the truth, that the item was picked for
        matching the request rather than for a personalised signal.
        """
        fired = [c for c in self.components if c.reason and c.delta > 0]
        fired.sort(key=lambda c: c.delta, reverse=True)
        if fired:
            return [c.reason for c in fired[:limit]]

        ptype = (self.article.get("product_type_name") or "item").lower()
        return [f"One of the closest matches to what you asked for in {ptype}s"]

    def breakdown(self) -> list[dict]:
        """Full audit trail — every component, including silent ones."""
        return [
            {"name": c.name, "delta": round(c.delta, 4), "reason": c.reason}
            for c in self.components
        ]


# ── Stats cache ───────────────────────────────────────────────────────────────

class ArticleStatsCache:
    """
    Holds article_stats and group_stats in memory.

    41,794 rows is roughly 12 MB as plain dicts, and scoring must not issue a
    query per candidate, so the whole table is loaded once on first use.
    Everything downstream is a dict lookup.
    """

    def __init__(self):
        self._articles: dict[str, dict] = {}
        self._groups:   dict[tuple, dict] = {}
        self._loaded = False

    async def load(self, force: bool = False):
        if self._loaded and not force:
            return
        from text_rag.db.postgres_client import get_pool
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM article_stats")
                for r in rows:
                    self._articles[str(r["article_id"])] = dict(r)
                grows = await conn.fetch("SELECT * FROM group_stats")
                for g in grows:
                    self._groups[(g["level"], g["group_key"])] = dict(g)
            self._loaded = True
            print(f"[StatsCache] loaded {len(self._articles)} article_stats, "
                  f"{len(self._groups)} group_stats")
        except Exception as e:
            # Missing tables must degrade to "no buying stats", never crash a
            # recommendation. Run: python -m text_rag.db.article_stats --build
            print(f"[StatsCache] WARNING: could not load stats ({e}). "
                  f"Buying-stat signals disabled. "
                  f"Run: python -m text_rag.db.article_stats --build")
            self._loaded = True

    @property
    def available(self) -> bool:
        return bool(self._articles)

    def get(self, article_id) -> Optional[dict]:
        return self._articles.get(str(article_id))

    def group(self, level: str, key: str) -> Optional[dict]:
        return self._groups.get((level, key))

    def age_distribution(self, stats: dict) -> tuple[Optional[dict], str]:
        """
        Returns (bucket_counts, support_level) using hierarchical backoff.

        The median article has 2 unique buyers, so its own age histogram is
        usually too thin to trust. We walk down to denser levels until support
        is sufficient: article -> type_colour -> garment_group -> global.
        """
        if not stats:
            return None, "none"

        def _dist(src):
            return {
                "16-25": src.get("age_16_25", 0),
                "26-35": src.get("age_26_35", 0),
                "36-50": src.get("age_36_50", 0),
                "51+":   src.get("age_51_plus", 0),
            }

        if stats.get("age_known", 0) >= MIN_AGE_SUPPORT_ARTICLE:
            return _dist(stats), "article"

        tc = self.group("type_colour",
                        f"{stats.get('product_type_name') or ''}|{stats.get('colour_group_name') or ''}")
        if tc and tc.get("age_known", 0) >= MIN_AGE_SUPPORT_GROUP:
            return _dist(tc), "type_colour"

        gg = self.group("garment_group", stats.get("garment_group_name") or "Unknown")
        if gg and gg.get("age_known", 0) >= MIN_AGE_SUPPORT_GROUP:
            return _dist(gg), "garment_group"

        gl = self.group("global", "ALL")
        if gl:
            return _dist(gl), "global"
        return None, "none"

    def global_age_share(self, bucket: str) -> float:
        """Base rate for an age bucket across the whole catalogue."""
        gl = self.group("global", "ALL")
        if not gl or not gl.get("age_known"):
            return 0.0
        key = {"16-25": "age_16_25", "26-35": "age_26_35",
               "36-50": "age_36_50", "51+": "age_51_plus"}[bucket]
        return gl.get(key, 0) / gl["age_known"]


_stats_cache = ArticleStatsCache()


async def get_stats_cache() -> ArticleStatsCache:
    await _stats_cache.load()
    return _stats_cache


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_price(article) -> Optional[float]:
    p = article.get("avg_price")
    if p is None:
        return None
    try:
        return float(p)
    except (TypeError, ValueError):
        return None


def _blend_weights(hints: dict) -> tuple[float, float]:
    """
    Returns (user_fit_multiplier, buying_stats_multiplier).

    A shopper with a long history gets their own behaviour weighted heavily.
    A cold-start shopper has nothing to fit, so social proof carries the
    decision instead of falling back to an arbitrary catalogue order.
    """
    n = (hints or {}).get("total_purchases", 0) or 0
    confidence = min(1.0, n / FULL_CONFIDENCE_PURCHASES)
    # Ranges from (0.30 user / 1.00 stats) cold to (1.00 user / 0.45 stats) warm.
    user_mult  = 0.30 + 0.70 * confidence
    stats_mult = 1.00 - 0.55 * confidence
    return round(user_mult, 3), round(stats_mult, 3)


def _ordinal(rank: int) -> str:
    return {1: "most-bought", 2: "2nd most-bought", 3: "3rd most-bought"}.get(
        rank, f"#{rank} most-bought"
    )


# ── Component scorers ─────────────────────────────────────────────────────────
# Each returns a Component. A component with reason=None still contributes to
# the score but is not shown to the user (too weak or too generic to be worth
# a line on the card).

def _c_attribute_affinity(
    name: str, article_value: str, pct_map: dict, ordered: list,
    weight: float, label: str, unit: str,
) -> Component:
    """
    Shared scorer for colour / type / garment / pattern / section affinity.
    Score scales with how large a share of the user's history that value holds.
    """
    if not article_value or not pct_map:
        return Component(name, 0.0, None)
    pct = pct_map.get(article_value)
    if pct is None:
        return Component(name, 0.0, None)

    # Rank within the user's own top list drives a decaying bonus, and the raw
    # share drives magnitude — a colour at 34% of purchases matters more than
    # one at 6% even if both are "top 5".
    rank  = (ordered.index(article_value) + 1) if article_value in ordered else 5
    share = min(pct / 100.0, 1.0)
    delta = weight * (0.45 + 0.55 * share) * (1.0 - 0.10 * (rank - 1))
    delta = max(delta, 0.0)

    if rank == 1:
        reason = f"{article_value} is your {_ordinal(1)} {label} ({pct:.0f}% of your purchases)"
    else:
        reason = f"{article_value} is your {_ordinal(rank)} {label} ({pct:.0f}% of your purchases)"
    if unit == "section":
        reason = f"From {article_value}, one of your top {label}s ({pct:.0f}% of purchases)"
    return Component(name, round(delta, 4), reason)


def _c_price_fit(article: dict, hints: dict) -> Component:
    """Rewards items inside the user's habitual spend band."""
    price = _norm_price(article)
    rng   = (hints or {}).get("preferred_price_range")
    if price is None or not rng or len(rng) != 2:
        return Component("price_fit", 0.0, None)
    lo, hi = float(rng[0]), float(rng[1])
    if hi <= 0:
        return Component("price_fit", 0.0, None)

    if lo <= price <= hi:
        return Component(
            "price_fit", W_USER_FIT["price_fit"],
            f"£{price:.2f} is inside your usual £{lo:.2f}-£{hi:.2f} spend range",
        )
    # Outside the band: decay with relative distance, no reason shown.
    span = max(hi - lo, 1.0)
    dist = (lo - price) if price < lo else (price - hi)
    delta = W_USER_FIT["price_fit"] * max(0.0, 1.0 - dist / (span * 2)) * 0.5
    return Component("price_fit", round(delta, 4), None)


def _c_gender_fit(article: dict, hints: dict) -> Component:
    """Aligns index_group with the gender inferred from purchase history."""
    gender = (hints or {}).get("inferred_gender")
    groups = {"female": ["Ladieswear", "Divided"], "male": ["Menswear"]}.get(gender or "", [])
    if not groups:
        return Component("gender_fit", 0.0, None)
    if article.get("index_group_name") in groups:
        # Deliberately silent: "it's womenswear" is not a persuasive reason.
        return Component("gender_fit", W_USER_FIT["gender_fit"], None)
    return Component("gender_fit", 0.0, None)


def _c_session_preference(article: dict, boosts: list) -> Component:
    """
    Preferences stated during the conversation (liked attributes).
    Note only article-backed attributes can match; 'style'/'occasion' have no
    article column, so they are scored via soft_constraints upstream instead.
    """
    if not boosts:
        return Component("session_preference", 0.0, None)
    hits = []
    total = 0.0
    for b in boosts:
        attr, val, wt = b.get("attribute"), b.get("value"), b.get("weight", 0)
        if attr and val and article.get(attr) == val:
            total += wt
            hits.append(val)
    if not hits:
        return Component("session_preference", 0.0, None)
    delta = W_USER_FIT["session_preference"] * min(total, 1.5)
    return Component(
        "session_preference", round(delta, 4),
        f"Matches what you told me you like: {', '.join(hits[:2])}",
    )


def _c_popularity(stats: dict) -> Component:
    """
    Shrunk popularity within the article's product type.

    Uses popularity_pct (percentile of the Dirichlet-smoothed share) rather
    than raw buy_count, because at a median of 3 purchases per article raw
    counts are mostly noise.
    """
    if not stats:
        return Component("popularity", 0.0, None)
    pct  = float(stats.get("popularity_pct") or 50.0)
    buys = int(stats.get("buy_count") or 0)
    uniq = int(stats.get("unique_buyers") or 0)

    delta = W_BUYING_STATS["popularity"] * max(0.0, (pct - 50.0) / 50.0)

    # Only claim popularity when it is both above the floor and backed by
    # enough distinct people to be a real statement.
    if pct >= POPULARITY_PCT_FLOOR and uniq >= 3:
        ptype = stats.get("product_type_name") or "this category"
        reason = (f"Popular choice: bought {buys} times by {uniq} different customers, "
                  f"top {max(1, round(100 - pct))}% of {ptype}")
        return Component("popularity", round(delta, 4), reason)
    return Component("popularity", round(delta, 4), None)


def _c_age_group_match(stats: dict, hints: dict, cache: ArticleStatsCache) -> Component:
    """
    Lift of the user's age bucket among this article's buyers vs the catalogue
    base rate, with backoff when the article's own sample is too thin.
    """
    bucket = (hints or {}).get("age_bucket")
    if not bucket or bucket not in AGE_BUCKETS or not stats:
        return Component("age_group_match", 0.0, None)

    dist, support = cache.age_distribution(stats)
    if not dist or support in ("none", "global"):
        # 'global' carries no discriminating information — every article would
        # score identically, so treat it as no signal.
        return Component("age_group_match", 0.0, None)

    total = sum(dist.values())
    if total <= 0:
        return Component("age_group_match", 0.0, None)

    share = dist.get(bucket, 0) / total
    base  = cache.global_age_share(bucket)
    if base <= 0:
        return Component("age_group_match", 0.0, None)

    lift  = share / base
    delta = W_BUYING_STATS["age_group_match"] * max(0.0, min(math.log(max(lift, 0.01)) / math.log(2.5), 1.0))

    if lift >= AGE_LIFT_THRESHOLD and share >= 0.25:
        scope = {
            "article":       "its buyers",
            "type_colour":   f"buyers of {stats.get('colour_group_name','')} {stats.get('product_type_name','')}".strip(),
            "garment_group": f"buyers in {stats.get('garment_group_name','')}".strip(),
        }.get(support, "its buyers")
        return Component(
            "age_group_match", round(delta, 4),
            f"{share*100:.0f}% of {scope} are aged {bucket}, like you "
            f"({lift:.1f}x the average)",
        )
    return Component("age_group_match", round(delta, 4), None)


def _c_repeat_rate(stats: dict) -> Component:
    """Share of buyers who came back for the same article — a quality proxy."""
    if not stats:
        return Component("repeat_rate", 0.0, None)
    rate = float(stats.get("repeat_rate") or 0.0)
    uniq = int(stats.get("unique_buyers") or 0)
    delta = W_BUYING_STATS["repeat_rate"] * min(rate * 2.0, 1.0)

    if rate >= 0.25 and uniq >= 4:
        return Component(
            "repeat_rate", round(delta, 4),
            f"{rate*100:.0f}% of its buyers bought it more than once",
        )
    return Component("repeat_rate", round(delta, 4), None)


def _c_recency_trend(stats: dict) -> Component:
    """Share of recent sales in the trend window vs the window before it."""
    if not stats:
        return Component("recency_trend", 0.0, None)
    share = float(stats.get("trend_share") or 0.0)
    rec   = int(stats.get("recent_count") or 0)
    delta = W_BUYING_STATS["recency_trend"] * max(0.0, (share - 0.5) * 2.0)

    if share >= 0.65 and rec >= 3:
        return Component(
            "recency_trend", round(delta, 4),
            f"Trending: {share*100:.0f}% of its sales happened in the last 3 months",
        )
    return Component("recency_trend", round(delta, 4), None)


def _concept(value) -> str:
    """
    Loose key for comparing values that name the same thing across columns.
    `product_type_name='Shirt'` and `garment_group_name='Shirts'` are the same
    concept, so a request for shirts must protect both from being penalised.
    """
    return re.sub(r"s$", "", str(value or "").strip().lower())


def _protected_values(hints: dict) -> set:
    """Values the user buys often enough that a dislike entry cannot be right."""
    protected = set()
    for key in ("colour_pcts", "type_pcts", "garment_pcts", "pattern_pcts"):
        for value, pct in ((hints or {}).get(key, {}) or {}).items():
            if pct is not None and pct >= PROTECTED_HISTORY_PCT:
                protected.add(value)
    return protected


def _c_dislike_penalty(
    article: dict, penalties: dict, filters: dict, hints: dict = None
) -> Component:
    """
    Demotes attribute values the user has rejected.

    Three guards, because dislike lists drift. Feedback on a single item records
    every one of its attributes as disliked, so after a few rejections the list
    can name most of the catalogue — including things the user demonstrably
    likes and things they just asked for.

      1. Never penalise what was requested this turn, matched loosely so
         'Shirt' also protects the 'Shirts' garment group.
      2. Never penalise a staple of the user's purchase history.
      3. Never penalise a demographic segment (see _DEMOGRAPHIC_ATTRS).

    The total is capped at W_DISLIKE_PENALTY so one drifted profile cannot push
    a score deeply negative.
    """
    if not penalties:
        return Component("dislike_penalty", 0.0, None)

    requested = {_concept(v) for v in (filters or {}).values() if isinstance(v, str)}
    protected = _protected_values(hints)

    hits = []
    for attr, disliked in penalties.items():
        if attr in _DEMOGRAPHIC_ATTRS:
            continue
        val = article.get(attr)
        if not val or val not in disliked:
            continue
        if _concept(val) in requested:
            continue
        if val in protected:
            continue
        hits.append(f"{attr}={val}")

    if not hits:
        return Component("dislike_penalty", 0.0, None)

    delta = -min(W_DISLIKE_PENALTY * len(hits), W_DISLIKE_PENALTY)
    return Component("dislike_penalty", round(delta, 4), None)


def _c_semantic(article: dict, max_score: float) -> Component:
    """
    Qdrant cosine similarity, normalised against the best score in this
    candidate set. PostgreSQL candidates have no _score; they get the neutral
    midpoint so the two sources are comparable on one scale.
    """
    raw = article.get("_score")
    if raw is None:
        return Component("semantic_relevance", round(W_SEMANTIC * 0.5, 4), None)
    if max_score <= 0:
        return Component("semantic_relevance", 0.0, None)
    return Component("semantic_relevance", round(W_SEMANTIC * (float(raw) / max_score), 4), None)


# ── Main ranker ───────────────────────────────────────────────────────────────

class PersonalizedRanker:
    """
    Scores and selects candidates. Stateless apart from the shared stats cache.

    Usage:
        ranker = PersonalizedRanker()
        scored = await ranker.rank(candidates, payload, filters)
        top    = ranker.select(scored, quantity=2)
    """

    async def rank(
        self,
        candidates: list[dict],
        payload:    dict,
        filters:    dict = None,
    ) -> list[ScoredItem]:
        """Scores every candidate and returns them sorted best-first."""
        if not candidates:
            return []

        hints    = payload.get("purchase_history_hints", {}) or {}
        boosts   = payload.get("preference_boosts", []) or []
        penalty  = payload.get("penalties", {}) or {}
        filters  = filters or {}

        cache = await get_stats_cache()
        user_mult, stats_mult = _blend_weights(hints)

        max_sem = 0.0
        for c in candidates:
            s = c.get("_score")
            if s is not None:
                try:
                    max_sem = max(max_sem, float(s))
                except (TypeError, ValueError):
                    pass

        colour_pcts  = hints.get("colour_pcts",  {}) or {}
        type_pcts    = hints.get("type_pcts",    {}) or {}
        garment_pcts = hints.get("garment_pcts", {}) or {}
        pattern_pcts = hints.get("pattern_pcts", {}) or {}
        section_pcts = hints.get("section_pcts", {}) or {}

        top_colours = hints.get("top_colours", []) or []
        top_types   = hints.get("top_product_types", []) or []

        scored: list[ScoredItem] = []
        for art in candidates:
            stats = cache.get(art.get("article_id"))

            # Qdrant payloads carry no product_code, which the variant dedup in
            # select() needs. Backfill it from the stats table when missing.
            if stats and art.get("product_code") is None:
                art["product_code"] = stats.get("product_code")

            user_components = [
                _c_attribute_affinity(
                    "colour_affinity", art.get("colour_group_name"), colour_pcts,
                    top_colours, W_USER_FIT["colour_affinity"], "colour", "colour",
                ),
                _c_attribute_affinity(
                    "type_affinity", art.get("product_type_name"), type_pcts,
                    top_types, W_USER_FIT["type_affinity"], "product type", "type",
                ),
                _c_attribute_affinity(
                    "garment_affinity", art.get("garment_group_name"), garment_pcts,
                    list(garment_pcts.keys()), W_USER_FIT["garment_affinity"],
                    "category", "garment",
                ),
                _c_attribute_affinity(
                    "pattern_affinity", art.get("graphical_appearance_name"), pattern_pcts,
                    list(pattern_pcts.keys()), W_USER_FIT["pattern_affinity"],
                    "pattern", "pattern",
                ),
                _c_attribute_affinity(
                    "section_affinity", art.get("section_name"), section_pcts,
                    list(section_pcts.keys()), W_USER_FIT["section_affinity"],
                    "section", "section",
                ),
                _c_price_fit(art, hints),
                _c_gender_fit(art, hints),
                _c_session_preference(art, boosts),
            ]

            stat_components = [
                _c_popularity(stats),
                _c_age_group_match(stats, hints, cache),
                _c_repeat_rate(stats),
                _c_recency_trend(stats),
            ]

            # Blend the two families by user-history confidence.
            for c in user_components:
                c.delta = round(c.delta * user_mult, 4)
            for c in stat_components:
                c.delta = round(c.delta * stats_mult, 4)

            flat = [
                _c_semantic(art, max_sem),
                *user_components,
                *stat_components,
                _c_dislike_penalty(art, penalty, filters, hints),
            ]

            total = round(sum(c.delta for c in flat), 4)
            scored.append(ScoredItem(article=art, score=total, components=flat))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def select(
        self,
        scored:   list[ScoredItem],
        quantity: int = 2,
        diversify: bool = True,
    ) -> list[ScoredItem]:
        """
        Picks the final items: highest score first, one variant per product,
        and (when more than 2 are requested) spread across colours.
        """
        if not scored:
            return []

        # Collapse colourway variants of the same product. Without this a query
        # for black trousers returns the identical product twice under two
        # article_ids, which reads as a broken recommendation.
        seen_products = set()
        deduped: list[ScoredItem] = []
        for s in scored:
            art = s.article
            key = art.get("product_code")
            if key is None:
                # Fall back to name+type when product_code is absent.
                key = f"{art.get('prod_name','')}|{art.get('product_type_name','')}"
            if key in seen_products:
                continue
            seen_products.add(key)
            deduped.append(s)

        if diversify and quantity > 2:
            # Colour spread for multi-item requests, preserving score order
            # within each colour.
            spread: list[ScoredItem] = []
            used_colours = set()
            for s in deduped:
                colour = (s.article.get("colour_group_name") or "").lower()
                if colour and colour in used_colours:
                    continue
                used_colours.add(colour)
                spread.append(s)
            leftover = [s for s in deduped if s not in spread]
            deduped = spread + leftover

        return self._pick_explainable(deduped, quantity)

    @staticmethod
    def _pick_explainable(ordered: list[ScoredItem], quantity: int) -> list[ScoredItem]:
        """
        Takes the top `quantity` items, preferring ones that can explain
        themselves.

        An item selected purely on silent signals shows an empty card, which
        reads as a broken recommendation. When the next best candidate has no
        reason, a candidate that does is promoted ahead of it — but only if it
        scores within EXPLAINABILITY_MARGIN, so a clearly better item is never
        displaced just for being quiet.
        """
        pool   = list(ordered)
        picked: list[ScoredItem] = []

        while pool and len(picked) < quantity:
            head = pool[0]
            if head.has_reasons:
                picked.append(pool.pop(0))
                continue

            swap_idx = next(
                (
                    i for i, cand in enumerate(pool[1:], start=1)
                    if cand.has_reasons and (head.score - cand.score) <= EXPLAINABILITY_MARGIN
                ),
                None,
            )
            if swap_idx is None:
                picked.append(pool.pop(0))
            else:
                promoted = pool.pop(swap_idx)
                print(f"[RANKER] promoted {promoted.article_id} "
                      f"(score={promoted.score:.3f}) over {head.article_id} "
                      f"(score={head.score:.3f}) — has a reason to show")
                picked.append(promoted)

        return picked

    @staticmethod
    def match_percent(item: ScoredItem, selected: list[ScoredItem]) -> Optional[int]:
        """
        A 0-100 match figure for display, scaled against the best score shown.

        Raw scores are unbounded and can be negative once penalties apply, so
        they cannot go on a card directly. Returns None when the whole set
        scored at or below zero — in that case nothing meaningful was matched
        and no badge should be shown at all.
        """
        best = max((s.score for s in selected), default=0.0)
        if best <= 0 or item.score <= 0:
            return None
        return int(round(min(item.score / best, 1.0) * 100))

    @staticmethod
    def log(scored: list[ScoredItem], selected: list[ScoredItem]) -> None:
        """Prints the ranking decision in the same style as the rest of the pipeline."""
        print(f"[RANKER] scored {len(scored)} candidates, selected {len(selected)}")
        for s in scored[:5]:
            fired = [f"{c.name}={c.delta:+.3f}" for c in s.components if abs(c.delta) > 0.001]
            mark = "->" if s in selected else "  "
            print(f"  [RANKER] {mark} {s.article_id} "
                  f"{str(s.article.get('prod_name',''))[:24]:24s} "
                  f"score={s.score:.3f}  {' '.join(fired[:5])}")
        for s in selected:
            for r in s.reasons():
                print(f"     [WHY] {s.article_id}: {r}")
