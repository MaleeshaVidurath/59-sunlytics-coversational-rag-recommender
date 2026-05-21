"""
KB Retriever — Fashion Psychology Knowledge Base Interface (NOVELTY 5)

Provides three public methods consumed by m2_handlers.py:
  - score()           → float added to Phase 2 final_score
  - get_context()     → string injected into Phase 4 LLM reranking prompt
  - get_explanation() → string appended to Phase 6 explanation prompt

All lookups resolve aliases before hitting the KB so natural language
style words ("chic", "laid back", "work") map correctly to KB keys.
"""

from m2_multimodal_rag.knowledge_base.fashion_kb import (
    COLOR_PSYCHOLOGY,
    OCCASION_RULES,
    KANSEI_MAPPING,
    KANSEI_ALIASES,
    OCCASION_ALIASES,
    APPEARANCE_RULES,
    GENDER_COLOUR_RULES,
    COLOR_HARMONY,
)


class FashionKBRetriever:

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_occasion(self, raw: str | None) -> str | None:
        if not raw:
            return None
        key = raw.strip().lower()
        return OCCASION_ALIASES.get(key, key if key in OCCASION_RULES else None)

    def _resolve_kansei(self, raw: str | None) -> str | None:
        if not raw:
            return None
        key = raw.strip().lower()
        return KANSEI_ALIASES.get(key, key if key in KANSEI_MAPPING else None)

    # ------------------------------------------------------------------
    # PUBLIC METHOD 1 — score()
    # Called in Phase 2 of handle_catalog_search.
    # Returns a float to add/subtract from final_score.
    # ------------------------------------------------------------------
    def score(
        self,
        item_colour: str,
        item_type: str,
        item_appearance: str = "",
        occasion: str | None = None,
        style_word: str | None = None,
        index_group_name: str = "",
    ) -> float:
        """
        Computes a psychology-grounded score for a single candidate item.

        Scoring logic:
          A) Occasion × colour/type match  — from OCCASION_RULES
          B) Kansei × colour/type/appearance match — from KANSEI_MAPPING
          C) Colour self-consistency with occasion — from COLOR_PSYCHOLOGY

        Returns a float (positive = boost, negative = penalty).
        """
        total = 0.0
        colour = str(item_colour).strip()
        itype  = str(item_type).strip()
        iapp   = str(item_appearance).strip()

        occ_key    = self._resolve_occasion(occasion)
        kansei_key = self._resolve_kansei(style_word)

        # ── A: Occasion rules ────────────────────────────────────────
        if occ_key and occ_key in OCCASION_RULES:
            rules  = OCCASION_RULES[occ_key]
            weight = rules.get("score_weight", 0.15)

            if colour in rules.get("preferred_colours", []):
                total += weight
            if colour in rules.get("avoid_colours", []):
                total -= weight * 1.2          # penalty slightly heavier than boost

            if itype in rules.get("preferred_types", []):
                total += weight * 0.7
            if itype in rules.get("avoid_types", []):
                total -= weight * 0.8

            if iapp and iapp in rules.get("preferred_appearance", []):
                total += weight * 0.4

        # ── B: Kansei mapping ────────────────────────────────────────
        if kansei_key and kansei_key in KANSEI_MAPPING:
            kansei = KANSEI_MAPPING[kansei_key]
            weight = kansei.get("score_weight", 0.10)

            if colour in kansei.get("preferred_colours", []):
                total += weight
            if colour in kansei.get("avoid_colours", []):
                total -= weight * 0.8

            if itype in kansei.get("preferred_types", []):
                total += weight * 0.8
            if itype in kansei.get("avoid_types", []):
                total -= weight * 0.8

            if iapp and iapp in kansei.get("preferred_appearance", []):
                total += weight * 0.5
            if iapp and iapp in kansei.get("avoid_appearance", []):
                total -= weight * 0.5

        # ── C: Colour self-consistency reinforcement ─────────────────
        if colour in COLOR_PSYCHOLOGY:
            psych = COLOR_PSYCHOLOGY[colour]
            if occ_key:
                if occ_key in psych.get("good_occasions", []):
                    total += 0.05
                if occ_key in psych.get("bad_occasions", []):
                    total -= 0.08
            if kansei_key and kansei_key in psych.get("kansei_match", []):
                total += 0.04

        # ── D: Appearance Rules (NOVELTY 5 — Improvement 1) ──────────
        # Boosts items whose graphical_appearance_name matches the active
        # occasion/Kansei rules; penalises mismatched textures.
        # Paper: Nagamachi (1995); Usip et al. (2020)
        if iapp and iapp in APPEARANCE_RULES:
            app_rule = APPEARANCE_RULES[iapp]
            app_w    = app_rule.get("score_weight", 0.05)
            if occ_key and occ_key in app_rule.get("occasion_fit", []):
                total += app_w
            if occ_key and occ_key in app_rule.get("occasion_avoid", []):
                total -= app_w * 1.2
            if kansei_key and kansei_key in app_rule.get("kansei_fit", []):
                total += app_w * 0.8
            if kansei_key and kansei_key in app_rule.get("kansei_avoid", []):
                total -= app_w * 0.8

        # ── E: Gender-Aware Rules (NOVELTY 5 — Improvement 2) ────────
        # Applies demographic-specific colour and type expectations.
        # E.g. Pink unusual for Menswear → small penalty; Black unusual
        # for Children → small penalty.
        # Paper: Kodzoman (2019)
        igender = str(index_group_name).strip()
        if igender and igender in GENDER_COLOUR_RULES:
            gender_rule = GENDER_COLOUR_RULES[igender]
            gw          = gender_rule.get("penalty_weight", 0.0)
            if gw > 0:
                if colour in gender_rule.get("unusual_colours", []):
                    total -= gw
                if itype in gender_rule.get("unusual_types", []):
                    total -= gw * 0.6

        return round(total, 4)

    # ------------------------------------------------------------------
    # PUBLIC METHOD 2 — get_context()
    # Called in Phase 4 to enrich the LLM reranking prompt.
    # Returns a short, readable string of the active KB rules.
    # ------------------------------------------------------------------
    def get_context(
        self,
        occasion: str | None = None,
        style_word: str | None = None,
    ) -> str:
        """
        Returns a one-paragraph KB context string for injection into LLM prompts.
        Empty string if no KB rules apply.
        """
        parts = []

        occ_key    = self._resolve_occasion(occasion)
        kansei_key = self._resolve_kansei(style_word)

        if occ_key and occ_key in OCCASION_RULES:
            rules = OCCASION_RULES[occ_key]
            pref_c = ", ".join(rules["preferred_colours"][:4])
            pref_t = ", ".join(rules["preferred_types"][:4])
            parts.append(
                f"Occasion '{occ_key}' psychology: prefer {pref_c} colours and "
                f"{pref_t} types. {rules['psych_basis']}."
            )

        if kansei_key and kansei_key in KANSEI_MAPPING:
            kansei = KANSEI_MAPPING[kansei_key]
            pref_c = ", ".join(kansei["preferred_colours"][:4])
            pref_t = ", ".join(kansei["preferred_types"][:3])
            parts.append(
                f"Style '{kansei_key}' Kansei mapping: prefer {pref_c} colours and "
                f"{pref_t} types. {kansei['description']}."
            )

        return " | ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # PUBLIC METHOD 3 — get_explanation()
    # Called in Phase 6 to produce a psychology-grounded explanation fact.
    # Returns a sentence the LLM can include in its explanation.
    # ------------------------------------------------------------------
    def get_explanation(
        self,
        item_colour: str,
        item_type: str,
        occasion: str | None = None,
        style_word: str | None = None,
    ) -> str:
        """
        Returns a psychology-grounded explanation sentence for a recommended item.
        Example: "Black was chosen because it conveys authority — ideal for interviews."
        """
        colour     = str(item_colour).strip()
        itype      = str(item_type).strip()
        occ_key    = self._resolve_occasion(occasion)
        kansei_key = self._resolve_kansei(style_word)

        facts = []

        # Colour psychology fact
        if colour in COLOR_PSYCHOLOGY:
            psych = COLOR_PSYCHOLOGY[colour]
            note  = psych.get("psych_note", "")
            if note:
                facts.append(note)

        # Occasion match explanation
        if occ_key and occ_key in OCCASION_RULES:
            rules = OCCASION_RULES[occ_key]
            if colour in rules.get("preferred_colours", []):
                facts.append(
                    f"{colour} is psychologically suited for {occ_key} settings — "
                    f"{rules['psych_basis'].lower()}"
                )
            if itype in rules.get("preferred_types", []):
                facts.append(
                    f"A {itype} is an appropriate garment choice for {occ_key} occasions"
                )

        # Kansei match explanation
        if kansei_key and kansei_key in KANSEI_MAPPING:
            kansei = KANSEI_MAPPING[kansei_key]
            if colour in kansei.get("preferred_colours", []):
                facts.append(
                    f"For a '{kansei_key}' aesthetic: {kansei['description'].lower()}"
                )

        if not facts:
            return ""

        # Return the most informative single fact
        return facts[0]

    # ------------------------------------------------------------------
    # PUBLIC METHOD 4 — detect_kansei_from_message()  (NOVELTY 5 — Improvement 3)
    # Scans raw user_message for implicit Kansei style words.
    # Called in Phase 2 when soft_constraints["style"] is absent, so the
    # KB scoring still captures emotional intent the user expressed naturally.
    # Paper: Nagamachi (1995) — Kansei Engineering from consumer language.
    # ------------------------------------------------------------------
    def detect_kansei_from_message(self, user_message: str) -> list:
        """
        Returns a list of resolved KB kansei keys found in the user message.
        Checks both direct KB keys and the full KANSEI_ALIASES dictionary.
        """
        if not user_message:
            return []
        msg = user_message.strip().lower()
        found = []
        for key in KANSEI_MAPPING:
            if key in msg and key not in found:
                found.append(key)
        for alias, canonical in KANSEI_ALIASES.items():
            if alias in msg and canonical not in found:
                found.append(canonical)
        return found

    # ------------------------------------------------------------------
    # PUBLIC METHOD 5 — harmony_score()  (NOVELTY 5 — Improvement 4)
    # Evaluates how well two item colours pair together visually.
    # Positive = complementary (good pairing), negative = clashing.
    # Used in Phase 5 post-MMR to log outfit coherence for the pair.
    # Paper: Guo Hui et al. (2023) — colour harmony in fashion design.
    # ------------------------------------------------------------------
    def harmony_score(self, colour1: str, colour2: str) -> float:
        """
        Returns a float in [-0.12, +0.15] indicating colour harmony.
        Checks both directions (colour1→colour2 and colour2→colour1).
        """
        if not colour1 or not colour2 or colour1 == colour2:
            return 0.0
        entry = COLOR_HARMONY.get(colour1, {})
        if colour2 in entry.get("complementary", []):
            return 0.15
        if colour2 in entry.get("analogous", []):
            return 0.08
        if colour2 in entry.get("clashing", []):
            return -0.12
        entry_rev = COLOR_HARMONY.get(colour2, {})
        if colour1 in entry_rev.get("complementary", []):
            return 0.15
        if colour1 in entry_rev.get("analogous", []):
            return 0.08
        if colour1 in entry_rev.get("clashing", []):
            return -0.12
        return 0.0

    # ------------------------------------------------------------------
    # PUBLIC METHOD 6 — get_clip_terms()  (NOVELTY 5 — Improvement 5)
    # Returns CLIP-optimised visual search vocabulary for Phase 1.
    # Combines occasion clip_visual_terms with Kansei preferred colours/types
    # so the FAISS retrieval benefits from domain-specific visual language.
    # Paper: Nagamachi (1995); Usip et al. (2020)
    # ------------------------------------------------------------------
    def get_clip_terms(
        self,
        occasion: str | None = None,
        style_word: str | None = None,
    ) -> str:
        """
        Returns a space-separated string of visual terms for CLIP retrieval.
        Empty string when no KB rules apply.
        """
        terms = []
        occ_key    = self._resolve_occasion(occasion)
        kansei_key = self._resolve_kansei(style_word)
        if occ_key and occ_key in OCCASION_RULES:
            t = OCCASION_RULES[occ_key].get("clip_visual_terms", "")
            if t:
                terms.append(t)
        if kansei_key and kansei_key in KANSEI_MAPPING:
            k       = KANSEI_MAPPING[kansei_key]
            colours = " ".join(k.get("preferred_colours", [])[:3]).lower()
            types_  = " ".join(k.get("preferred_types", [])[:3]).lower()
            if colours or types_:
                terms.append(f"{colours} {types_}".strip())
        return " ".join(terms)

    # ------------------------------------------------------------------
    # PUBLIC METHOD 7 — debug_score()
    # Prints a breakdown of how the KB score was computed (for testing).
    # ------------------------------------------------------------------
    def debug_score(
        self,
        item_colour: str,
        item_type: str,
        item_appearance: str = "",
        occasion: str | None = None,
        style_word: str | None = None,
        index_group_name: str = "",
    ) -> None:
        occ_key    = self._resolve_occasion(occasion)
        kansei_key = self._resolve_kansei(style_word)
        total      = self.score(
            item_colour, item_type, item_appearance, occasion, style_word, index_group_name
        )

        print(f"\n  [KB Score Debug] colour={item_colour}, type={item_type}")
        print(f"    occasion resolved : {occ_key}")
        print(f"    kansei resolved   : {kansei_key}")
        print(f"    KB score          : {total:+.4f}")
        ctx = self.get_context(occasion, style_word)
        if ctx:
            print(f"    KB context        : {ctx[:120]}...")
        expl = self.get_explanation(item_colour, item_type, occasion, style_word)
        if expl:
            print(f"    KB explanation    : {expl[:120]}")


# Global singleton
kb_retriever = FashionKBRetriever()
