# m3_implementation/memory/core/contradiction_detector.py
#
# CROSS-TURN CONSISTENCY LAYER
#
# WHAT THIS ANSWERS, AND WHAT IT DELIBERATELY DOES NOT:
#   The hallucination guard asks "is this turn faithful to its own evidence?".
#   This component asks a question the guard structurally cannot:
#
#       "Is this turn consistent with everything the system has already told
#        this user — including when this turn's evidence is silent about the
#        attribute, and including when the catalogue has changed underneath us?"
#
#   The previous implementation could not answer that. It wrote the current
#   turn's evidence into a graph and read it straight back out, so its
#   comparison reduced to response-vs-current-evidence — the guard's job, done
#   a second time with a second model call. Anything genuinely cross-turn was
#   invisible to it, and the one case where history is the ONLY available
#   reference (a follow-up turn served from cache, with no product evidence)
#   was skipped outright.
#
# THREE FAILURE CLASSES, TWO OF THEM OURS:
#   1. Turn-local unfaithfulness   response disagrees with THIS turn's evidence.
#                                  Owned by the hallucination guard. Recorded
#                                  here as "deferred" and never counted, so the
#                                  two components stop double-reporting.
#   2. Cross-turn drift            response disagrees with what the system
#                                  asserted on an EARLIER turn, with no
#                                  catalogue change to justify it. Corrected
#                                  against the ledger.
#   3. Ground-truth revision       the database value changed between turns.
#                                  The old statement was true when made, so it
#                                  is superseded rather than corrected, the new
#                                  value is adopted, and the earlier turns that
#                                  quoted the old value are flagged for an
#                                  in-chat correction notice.
#
# HOW A REVISION IS EVEN VISIBLE:
#   Cached and PARTIAL turns never touch PostgreSQL, so their evidence bundle
#   can itself be stale. Every factual turn therefore re-verifies the candidate
#   products against the live articles table (one indexed query, <= a dozen
#   ids) before any comparison happens. Without that, a mid-session price
#   change would be re-served from Redis forever with nothing noticing.
#
# EXTRACTION:
#   Handled by text_rag.core.assertion_extractor, the same module the
#   hallucination guard uses to split sentences and map items to sentences.
#   One extraction pass, two consumers — and no second LLM call, which also
#   removes the Groq rate-limit failure mode that used to make this component
#   silently give up.
#
# OUTPUT STRUCTURE (superset of the previous one — existing callers unaffected):
#   {
#     "response_text":       str   — corrected response (or original)
#     "contradiction_found": bool
#     "contradiction_count": int
#     "contradictions":      list
#     "claims_stored":       int
#     "product_ids":         list
#     "product_names":       list
#     "revisions":           list  — catalogue changes detected this turn
#     "revision_count":      int
#     "deferred_count":      int   — turn-local cases left to the guard
#   }

import os
import sys
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memory.db.mongo import get_db, get_collection_name
from memory.core.assertion_ledger import (
    AssertionLedger, TRACKED_ATTRIBUTES, values_differ, normalise_value,
)
from text_rag.core.assertion_extractor import (
    extract_assertions, replace_in_scope, norm_ws,
)
from text_rag.core.nli_model import nli_scores

# Evaluation capture hook — no-op unless CONTRA_EVAL_CAPTURE=1
# (see test_result/contradiction_result)
try:
    from test_result.contradiction_result.capture import capture_case as _capture_contra_case
except Exception:
    def _capture_contra_case(**_kwargs):
        pass


# Actions that make factual product claims and are therefore worth reconciling.
FACTUAL_ACTIONS = {
    "catalog_search", "item_attribute_lookup", "item_compare",
    "explanation_generate", "item_detail_lookup",
}

# NLI confirmation gate for categorical drift. Its job is to let benign
# paraphrases through ("Dress" -> "maxi dress") while stopping real swaps.
NLI_CONFIRMATION_THRESHOLD = 0.5

# Attributes where a difference is decided by string logic alone.
# DeBERTa is documented (see hallucination_checker) to be unreliable on numbers
# and proper nouns: it scores "£11.08" against "£13.58" as neutral, and two
# unrelated product names as neutral too. Sending those through NLI would only
# suppress true positives.
_STRING_DECIDED = {"price", "name"}

# One live re-verification query per factual turn. Set CONTRA_DB_REVERIFY=0 to
# disable (evidence is then trusted as-is and catalogue revisions on cached
# turns become invisible again).
_DB_REVERIFY = os.getenv("CONTRA_DB_REVERIFY", "1") == "1"

_MAX_REVERIFY_IDS = 16


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_price(price) -> str:
    """Normalises any price representation to the '£X.XX' form used everywhere."""
    if price is None or price == "":
        return ""
    if isinstance(price, str):
        text = price.strip()
        if "£" in text:
            return text
        try:
            return f"£{float(text):.2f}"
        except (ValueError, TypeError):
            return text
    try:
        return f"£{float(price):.2f}"
    except (ValueError, TypeError):
        return str(price).strip()


# ── Candidate products ────────────────────────────────────────────────────────

def _ref_from_evidence_item(item: dict) -> Optional[dict]:
    """Normalises one evidence-bundle item to the ledger's attribute names."""
    aid = str(item.get("article_id", "")).strip()
    if not aid:
        return None
    return {
        "article_id":   aid,
        "name":         str(item.get("name") or item.get("prod_name") or "").strip(),
        "colour":       str(item.get("colour") or item.get("colour_group_name") or "").strip(),
        "price":        _fmt_price(item.get("price") if item.get("price") not in (None, "")
                                   else item.get("avg_price")),
        "product_type": str(item.get("type") or item.get("product_type_name") or "").strip(),
    }


def _refs_from_evidence(evidence: dict) -> list[dict]:
    """
    Products this turn's evidence bundle actually carries. These are the ones
    the hallucination guard has already checked, so their attributes mark the
    boundary between "guard territory" and "ledger territory".
    """
    refs   = []
    action = evidence.get("action", "")

    if action == "catalog_search":
        source = evidence.get("items", [])
    elif action == "item_compare":
        source = [evidence.get("item_a") or {}, evidence.get("item_b") or {}]
    elif action in ("item_attribute_lookup", "item_detail_lookup", "explanation_generate"):
        source = [evidence.get("article") or {}] + list(evidence.get("articles", []) or [])
    else:
        source = []

    for item in source:
        ref = _ref_from_evidence_item(item or {})
        if ref and ref["name"]:
            refs.append(ref)
    return refs


def _refs_from_context(retrieval_input: dict) -> list[dict]:
    """
    Products carried in the session's context window. On a follow-up turn the
    evidence bundle may name nothing at all, yet the user is plainly still
    talking about these — so they remain valid subjects for a consistency check.
    """
    if not retrieval_input:
        return []
    refs = []
    for _slot, item in (retrieval_input.get("items_in_context") or {}).items():
        if not item:
            continue
        aid = str(item.get("article_id", "")).strip()
        if not aid:
            continue
        refs.append({
            "article_id":   aid,
            "name":         str(item.get("prod_name") or item.get("name") or "").strip(),
            "colour":       str(item.get("colour_group_name") or item.get("colour") or "").strip(),
            "price":        _fmt_price(item.get("price")),
            "product_type": str(item.get("product_type_name") or item.get("type") or "").strip(),
        })
    return refs


def _focus_ids(retrieval_input: dict) -> set:
    """
    Article ids the turn itself resolved as its subject — the target of a detail
    lookup, an attribute question, or the two sides of a comparison.

    These are attributable even when the response refers to them only by
    pronoun ("It's Navy"), because the router already decided which item the
    user meant.
    """
    payload = (retrieval_input or {}).get("payload") or {}
    ids = set()
    for key in ("article_id", "article_id_a", "article_id_b"):
        value = payload.get(key)
        if value:
            ids.add(str(value))
    for entry in payload.get("article_ids_list") or []:
        value = (entry or {}).get("article_id")
        if value:
            ids.add(str(value))
    return ids


def _mentions(response_text: str, name: str) -> bool:
    """True when the response names this product outright."""
    if not name:
        return False
    return norm_ws(name).lower() in norm_ws(response_text).lower()


def _attributable_refs(
    candidates:    list[dict],
    evidence_ids:  set,
    focus_ids:     set,
    response_text: str,
) -> list[dict]:
    """
    Narrows the candidate set to the products the response is ACTUALLY about.

    WHY THIS GATE EXISTS — it fixes a real corruption bug:
      The candidate set is deliberately wide (evidence + session context +
      ledger) so a follow-up turn can still be checked when its evidence names
      nothing. But the item→sentence matcher assigns EVERY item it is given to
      some sentence, as long as similarity clears a loose threshold. Hand it
      three candidates for a reply that discusses one, and the two spare
      products get force-matched onto sentences describing the third — after
      which their "values" are read out of the wrong sentence and reported as
      contradictions.

      Observed live: a correct reply about FREDRIK SHORTS (Dark Blue, £3.02)
      was rewritten into "Olivia shorts ... Dark Grey", because two context-only
      products had been matched onto its sentences.

    THE RULE:
      A product may be attributed a value only if the turn is genuinely about
      it — that is, this turn's evidence supplied it, or the router resolved it
      as the focus, or the response names it outright. A product the reply never
      mentions cannot be contradicted by that reply.

      Evidence and focus items are exempt from the name test on purpose: a
      swapped or invented name is itself the error we want to catch, and gating
      on the name would hide exactly that case.
    """
    keep, dropped = [], []
    for ref in candidates:
        aid = ref["article_id"]
        if (aid in evidence_ids
                or aid in focus_ids
                or _mentions(response_text, ref.get("name", ""))):
            keep.append(ref)
        else:
            dropped.append(ref.get("name") or aid)

    if dropped:
        print(f"[CONTRA] not discussed in this reply, so not attributable: "
              f"{dropped}")
    return keep


def _merge_refs(*groups: list[dict]) -> list[dict]:
    """First occurrence of an article_id wins — groups are passed most-trusted first."""
    merged, seen = [], set()
    for group in groups:
        for ref in group:
            aid = ref.get("article_id")
            if not aid or aid in seen:
                continue
            seen.add(aid)
            merged.append(ref)
    return merged


async def _live_db_values(article_ids: list[str]) -> dict:
    """
    Current catalogue values straight from PostgreSQL, keyed by article_id.

    This is the only reading that can be trusted to be fresh: the evidence
    bundle may have been assembled from the Redis session cache, which by
    design holds what was true when the item was first shown.
    Returns {} if the lookup is disabled or fails — the caller then falls back
    to the evidence bundle and simply loses revision detection for that turn.
    """
    if not _DB_REVERIFY or not article_ids:
        return {}

    # article_id is a BIGINT column; a non-numeric id would raise inside the
    # query helper and cost the whole turn its re-verification, so those are
    # dropped here rather than allowed to poison the batch.
    numeric = [aid for aid in article_ids if str(aid).strip().isdigit()]
    if not numeric:
        return {}

    try:
        from text_rag.db.postgres_client import get_articles_by_ids
        rows = await get_articles_by_ids(numeric[:_MAX_REVERIFY_IDS])
    except Exception as e:
        print(f"[CONTRA] live DB re-verification unavailable: {e}")
        return {}

    out = {}
    for row in rows:
        aid = str(row.get("article_id", ""))
        if not aid:
            continue
        out[aid] = {
            "name":         str(row.get("prod_name") or "").strip(),
            "colour":       str(row.get("colour_group_name") or "").strip(),
            "price":        _fmt_price(row.get("avg_price")),
            "product_type": str(row.get("product_type_name") or "").strip(),
        }
    return out


# ── NLI confirmation ──────────────────────────────────────────────────────────

_PHRASE = {
    "colour":       lambda name, v: f"The {name} is {v} in colour.",
    "product_type": lambda name, v: f"The {name} is a {v}.",
    "price":        lambda name, v: f"The {name} costs {v}.",
    "name":         lambda name, v: f"The product is called {v}.",
}


def _is_refinement(a, b) -> bool:
    """
    True when one value is a more specific rendering of the other
    ("Dress" vs "maxi dress"). Those are how the assistant legitimately
    paraphrases a catalogue term and must never be reported as a conflict.
    """
    na, nb = normalise_value(a), normalise_value(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def _confirm_drift(product_name: str, attribute: str, prior_value, new_value) -> tuple:
    """
    Decides whether a value difference is a genuine contradiction.

    Price and name are exact values — string inequality already settles it, and
    routing them through NLI would only add false negatives.
    Categorical attributes go through DeBERTa, which is what keeps benign
    rewording from being reported.

    Returns (is_contradiction, score).
    """
    if attribute in _STRING_DECIDED:
        return True, 1.0

    if _is_refinement(prior_value, new_value):
        return False, 0.0

    phrase     = _PHRASE.get(attribute,
                             lambda name, v: f"The {name} has {attribute} {v}.")
    name       = product_name or "product"
    premise    = phrase(name, prior_value)
    hypothesis = phrase(name, new_value)

    scores = nli_scores(premise, hypothesis)
    score  = scores["contradiction"]
    return (score > NLI_CONFIRMATION_THRESHOLD
            and score > scores["entailment"]), score


# ── The reconciliation decision ───────────────────────────────────────────────

def _pick_anchor(slot_truth, truth_is_live: bool, prior: Optional[dict]) -> tuple:
    """
    Chooses what an unguarded statement should be judged against.

        prior_assertion   what we told THIS USER on an earlier turn. Tried
                          first, because when a reply disagrees with it the
                          failure IS a self-contradiction, and naming it as one
                          is the truthful description — the catalogue merely
                          corroborates.
        live_evidence     the catalogue's own current answer, read this turn.
                          Used when there is no usable prior: a first mention,
                          or a prior that a revision has already retired.
        session_context   the value session memory holds for the item, carried
                          from the turn it was first shown. Last resort, for
                          articles with no live database row (or when
                          re-verification is switched off).

    WHY PRIOR CAN BE TRUSTED AHEAD OF THE DATABASE:
      A stale prior can never reach this point. Evidence is applied to the
      ledger BEFORE any comparison, so a catalogue value that has moved raises a
      revision, and the revision marks every assertion of the old value
      "superseded". prior_assertion returns only "active" ones. Therefore:

        DB changed since   → prior is None      → falls through to live_evidence
        DB unchanged       → prior == DB value  → same verdict either way

      The correction is identical under both orderings; only the recorded
      reason differs, and this ordering records the accurate one.

    Returns (anchor_value, anchor_source). ("", "") when nothing is known.
    """
    if prior:
        return prior["value"], "prior_assertion"
    if truth_is_live and slot_truth:
        return slot_truth, "live_evidence"
    if slot_truth:
        return slot_truth, "session_context"
    return "", ""


def _classify_assertion(
    stated,
    slot_truth,
    is_guarded:    bool,
    revision:      Optional[dict],
    prior:         Optional[dict],
    attribute:     str,
    product_name:  str,
    truth_is_live: bool = False,
) -> dict:
    """
    Decides what one stated value means, given everything known about its slot.
    This is the reconciliation table from the module docstring, in code.

    Returns:
        kind           "stale" | "defer" | "ok" | "drift"
        record_value   the value to write into the ledger for this turn
        record_status  "active" or "deferred"
        correct_to     replacement value for the response, or "" to leave it
        prior          the prior assertion, when there is one
        anchor         the value the decision was made against
        anchor_source  where that value came from (see _pick_anchor)
        score          confirmation score
    """
    def verdict(kind, record_value=stated, record_status="active",
                correct_to="", score=0.0, anchor="", anchor_source=""):
        return {"kind": kind, "record_value": record_value,
                "record_status": record_status, "correct_to": correct_to,
                "prior": prior, "anchor": anchor,
                "anchor_source": anchor_source, "score": score}

    # 1. The catalogue moved and the response quoted the value that was true
    #    before it moved. Not a fabrication — a stale reading. The response is
    #    brought up to date and the user is told why, via the revision notice.
    #
    #    Checked BEFORE the guard hand-off below: in a revision the evidence IS
    #    present, so deferring would let the guard call an aged quote a
    #    hallucination, which it is not.
    if (revision
            and not values_differ(stated, revision["old_value"])
            and values_differ(stated, revision["new_value"])):
        return verdict("stale",
                       record_value=revision["new_value"],
                       correct_to=revision["new_value"],
                       anchor=revision["old_value"],
                       anchor_source="revision")

    # 2. This turn's own evidence covers the attribute, so the hallucination
    #    guard has already judged it. Recording the disagreement keeps the audit
    #    trail complete, but counting it here would double-report a single
    #    failure, and anchoring a later turn to it would propagate a known-bad
    #    value — hence status "deferred".
    if is_guarded and slot_truth:
        if values_differ(stated, slot_truth):
            return verdict("defer", record_status="deferred",
                           anchor=slot_truth, anchor_source="turn_evidence")
        return verdict("ok", anchor=slot_truth, anchor_source="turn_evidence")

    # 3. The guard did not check this attribute this turn, so nothing else will
    #    judge this statement unless we do. Take the strongest reference we hold
    #    and compare against that.
    anchor, anchor_source = _pick_anchor(slot_truth, truth_is_live, prior)

    if not anchor or not values_differ(stated, anchor):
        return verdict("ok", anchor=anchor, anchor_source=anchor_source)

    confirmed, score = _confirm_drift(product_name, attribute, anchor, stated)
    if not confirmed:
        print(f"[CONTRA] {attribute}: {stated!r} vs {anchor_source} {anchor!r} "
              f"not confirmed (score={score:.3f}) — benign rewording")
        return verdict("ok", score=score, anchor=anchor,
                       anchor_source=anchor_source)

    return verdict("drift", correct_to=anchor, score=score,
                   anchor=anchor, anchor_source=anchor_source)


def _drift_entry(
    verdict:      dict,
    article_id:   str,
    attribute:    str,
    product_name: str,
    stated,
    turn_ordinal: int,
) -> dict:
    """
    Builds the log record for one corrected statement, and prints it.

    `anchor_source` is carried through faithfully because it is the honest
    distinction for the evaluation:
        prior_assertion  — contradicts what we told this user earlier
                           (turn_distance says how far back)
        live_evidence    — contradicts the catalogue, on a slot this turn's
                           evidence never showed the guard
        session_context  — contradicts what session memory holds for the item
    """
    prior         = verdict.get("prior")
    anchor        = verdict["anchor"]
    anchor_source = verdict["anchor_source"]

    from_prior     = anchor_source == "prior_assertion" and prior is not None
    distance       = turn_ordinal - prior["turn_ordinal"] if from_prior else 0
    anchor_turn_id = prior["turn_id"] if from_prior else ""

    where = (f"turn -{distance} said {anchor!r}" if from_prior
             else f"{anchor_source} says {anchor!r}")
    print(f"[CONTRA-DETECTED] ⚠ DRIFT {article_id}.{attribute} | "
          f"{where}, now {stated!r} | score={verdict['score']:.3f}")

    return {
        "kind":            "cross_turn_drift",
        "article_id":      article_id,
        "article_name":    product_name,
        "attribute":       attribute,
        "evidence_value":  anchor,
        "extracted_value": stated,
        "anchor_source":   anchor_source,
        "anchor_turn_id":  anchor_turn_id,
        "turn_distance":   distance,
        "nli_score":       verdict["score"],
        "corrected":       True,
    }


# ── MongoDB writes ────────────────────────────────────────────────────────────

async def _log_contradiction(
    session_id: str, user_id: str, turn_id: str, entry: dict,
    collection_prefix: str = "m3",
) -> None:
    db   = get_db()
    coll = get_collection_name("contradiction_log", collection_prefix)
    try:
        await db[coll].insert_one({
            "session_id":        session_id,
            "user_id":           user_id,
            "turn_id":           turn_id,
            "detected_at":       _now(),
            "kind":              entry.get("kind", "cross_turn_drift"),
            "article_id":        entry.get("article_id", ""),
            "article_name":      entry.get("article_name", ""),
            "attribute":         entry.get("attribute", ""),
            "evidence_value":    entry.get("evidence_value", ""),
            "extracted_value":   entry.get("extracted_value", ""),
            "anchor_source":     entry.get("anchor_source", ""),
            "anchor_turn_id":    entry.get("anchor_turn_id", ""),
            "turn_distance":     entry.get("turn_distance", 0),
            "nli_score":         entry.get("nli_score", 0.0),
            "resolution":        "response_corrected",
            "member_model":      collection_prefix,
        })
    except Exception as e:
        print(f"[CONTRA] contradiction log write failed: {e}")


async def _log_revision(
    session_id: str, user_id: str, turn_id: str, revision: dict,
    collection_prefix: str = "m3",
) -> None:
    """
    Records a catalogue change so the chat history can be annotated.
    `affected_turn_ids` names the assistant turns that quoted the old value;
    the sessions endpoint joins on those to show the notice in the right place.
    """
    db   = get_db()
    coll = get_collection_name("revision_notices", collection_prefix)
    try:
        await db[coll].insert_one({
            "session_id":        session_id,
            "user_id":           user_id,
            "detected_turn_id":  turn_id,
            "detected_at":       _now(),
            "article_id":        revision.get("article_id", ""),
            "product_name":      revision.get("product_name", ""),
            "attribute":         revision.get("attribute", ""),
            "old_value":         revision.get("old_value", ""),
            "new_value":         revision.get("new_value", ""),
            "old_version":       revision.get("old_version", 0),
            "new_version":       revision.get("new_version", 0),
            "affected_turn_ids": [t.get("turn_id") for t in revision.get("affected_turns", [])
                                  if t.get("turn_id")],
            "member_model":      collection_prefix,
        })
    except Exception as e:
        print(f"[CONTRA] revision notice write failed: {e}")


# ── Main component ────────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Reconciles each assistant response against the session's Assertion Ledger.
    See the module docstring for the three failure classes and their ownership.
    """

    async def check_and_resolve(
        self,
        response_text:     str,
        evidence:          dict,
        session_id:        str,
        user_id:           str,
        turn_id:           str,
        collection_prefix: str  = "m3",
        retrieval_input:   dict = None,
    ) -> dict:
        action = (evidence or {}).get("action", "no_retrieval")
        print("\n[CONTRA] ━━━ cross-turn reconciliation ━━━")
        print(f"[CONTRA] action={action} session={session_id[:12] if session_id else '?'} "
              f"turn={turn_id}")

        if action not in FACTUAL_ACTIONS:
            print(f"[CONTRA] SKIP: action={action} makes no factual product claim")
            return self._empty(response_text)

        try:
            return await self._reconcile(
                response_text, evidence, session_id, user_id, turn_id,
                action, collection_prefix, retrieval_input,
            )
        except Exception as e:
            import traceback
            print(f"[CONTRA] reconciliation failed: {e}")
            traceback.print_exc()
            return self._empty(response_text)

    async def _reconcile(
        self,
        response_text:     str,
        evidence:          dict,
        session_id:        str,
        user_id:           str,
        turn_id:           str,
        action:            str,
        collection_prefix: str,
        retrieval_input:   dict,
    ) -> dict:

        # ── 1. Open the ledger and register this turn ─────────────────────────
        ledger  = await AssertionLedger.load(session_id, collection_prefix)
        ordinal = ledger.begin_turn(
            turn_id, action,
            (retrieval_input or {}).get("retrieval_strategy", ""),
        )
        graph_before = ledger.legacy_node_view()   # for the evaluation harness
        print(f"[LEDGER] turn ordinal={ordinal} {ledger.stats()}")

        # ── 2. Decide which products this turn could be talking about ────────
        evidence_refs = _refs_from_evidence(evidence)
        context_refs  = _refs_from_context(retrieval_input)
        ledger_refs   = ledger.known_products()

        candidates = _merge_refs(evidence_refs, context_refs, ledger_refs)
        if not candidates:
            print("[CONTRA] no candidate products anywhere — nothing to reconcile")
            await ledger.save(session_id, collection_prefix)
            return self._empty(response_text)

        evidence_ids = {r["article_id"] for r in evidence_refs}
        print(f"[CONTRA] candidates={len(candidates)} "
              f"(evidence={len(evidence_refs)} context={len(context_refs)} "
              f"ledger={len(ledger_refs)})")

        # ── 3. Establish ground truth, preferring a live database read ───────
        # `guarded`    — (article, attribute) pairs this turn's own evidence
        #                covered, and which therefore belong to the guard.
        # `truth_live` — pairs whose value came from the live catalogue read
        #                rather than from a possibly-stale bundle. Only those
        #                are strong enough to overrule conversation history.
        live       = await _live_db_values([r["article_id"] for r in candidates])
        truth      = {}
        truth_live = set()
        guarded    = set()

        for ref in candidates:
            aid   = ref["article_id"]
            fresh = live.get(aid)
            for attribute in TRACKED_ATTRIBUTES:
                fresh_value = (fresh or {}).get(attribute)
                value       = fresh_value or ref.get(attribute, "")
                if value:
                    truth[(aid, attribute)] = value
                if fresh_value:
                    truth_live.add((aid, attribute))
                if aid in evidence_ids and ref.get(attribute):
                    guarded.add((aid, attribute))

        # ── 4. Feed ground truth to the ledger; catch catalogue revisions ────
        revisions = []
        for ref in candidates:
            aid = ref["article_id"]
            for attribute in TRACKED_ATTRIBUTES:
                value = truth.get((aid, attribute))
                if not value:
                    continue
                revision = ledger.apply_evidence(
                    aid, attribute, value,
                    name_hint=truth.get((aid, "name"), ref.get("name", "")),
                )
                if revision:
                    revisions.append(revision)

        # Truth may have moved on this turn — re-read it from the ledger so the
        # comparisons below always use the post-revision value.
        revised_slots = {(r["article_id"], r["attribute"]): r for r in revisions}

        # ── 5. Read what the response actually states ────────────────────────
        # The wide candidate set above exists so revisions are caught on every
        # product the session knows. Attribution is a different question: only
        # products this reply is genuinely about may be assigned values from it.
        attributable = _attributable_refs(
            candidates, evidence_ids, _focus_ids(retrieval_input), response_text,
        )

        # Described with the ledger's current truth so the item→sentence
        # matching is done against reality, not a stale cached description.
        described = [
            {
                "article_id":   r["article_id"],
                "name":         truth.get((r["article_id"], "name"), r.get("name", "")),
                "colour":       truth.get((r["article_id"], "colour"), r.get("colour", "")),
                "price":        truth.get((r["article_id"], "price"), r.get("price", "")),
                "product_type": truth.get((r["article_id"], "product_type"),
                                          r.get("product_type", "")),
            }
            for r in attributable
        ]
        extraction = extract_assertions(response_text, described, verbose=False)
        assertions = extraction["assertions"]
        sentences  = extraction["sentences"]
        print(f"[CONTRA] attributable={len(described)}/{len(candidates)} → "
              f"{len(assertions)} assertion(s) from {len(sentences)} sentence(s)")

        names_by_id = {d["article_id"]: d["name"] for d in described}

        # ── 6. Reconcile each assertion ──────────────────────────────────────
        corrected_text = response_text
        contradictions = []
        stale_fixes    = []
        deferred       = 0

        for a in assertions:
            aid, attribute = a["article_id"], a["attribute"]
            stated, sent_idx = a["value"], a["sentence_idx"]
            pname = names_by_id.get(aid, "")

            verdict = _classify_assertion(
                stated        = stated,
                slot_truth    = truth.get((aid, attribute), ""),
                is_guarded    = (aid, attribute) in guarded,
                revision      = revised_slots.get((aid, attribute)),
                prior         = ledger.prior_assertion(aid, attribute),
                attribute     = attribute,
                product_name  = pname,
                truth_is_live = (aid, attribute) in truth_live,
            )

            ledger.record_assertion(
                aid, attribute, verdict["record_value"],
                status=verdict["record_status"], sentence_idx=sent_idx,
                name_hint=pname,
            )

            if verdict["correct_to"]:
                corrected_text = replace_in_scope(
                    corrected_text, sentences, sent_idx,
                    stated, verdict["correct_to"],
                )

            kind = verdict["kind"]

            if kind == "stale":
                stale_fixes.append({
                    "article_id": aid, "attribute": attribute,
                    "from": stated, "to": verdict["correct_to"],
                })
                print(f"[CONTRA] STALE-VALUE FIX {aid}.{attribute}: "
                      f"{stated!r} → {verdict['correct_to']!r}")

            elif kind == "defer":
                deferred += 1
                print(f"[CONTRA] DEFER {aid}.{attribute}: stated={stated!r} "
                      f"evidence={verdict['anchor']!r} — hallucination guard owns this")

            elif kind == "drift":
                entry = _drift_entry(verdict, aid, attribute, pname,
                                     stated, ordinal)
                ledger.mark_contradiction(aid, attribute, stated,
                                          verdict["anchor"], verdict["score"])
                contradictions.append(entry)
                await _log_contradiction(session_id, user_id, turn_id, entry,
                                         collection_prefix)

        # ── 7. Persist revision notices ──────────────────────────────────────
        for revision in revisions:
            if revision.get("affected_turns"):
                await _log_revision(session_id, user_id, turn_id, revision,
                                    collection_prefix)

        # ── 8. Save the ledger ───────────────────────────────────────────────
        await ledger.save(session_id, collection_prefix)

        # How many corrections came from each kind of reference. The
        # prior_assertion count is the one that is genuinely cross-turn; the
        # others are claims the guard never saw. Reporting them apart keeps the
        # headline number honest.
        anchor_breakdown = {}
        for c in contradictions:
            src = c.get("anchor_source", "unknown")
            anchor_breakdown[src] = anchor_breakdown.get(src, 0) + 1

        print(f"[CONTRA] result: contradictions={len(contradictions)} "
              f"{anchor_breakdown} revisions={len(revisions)} "
              f"stale_fixes={len(stale_fixes)} deferred={deferred} "
              f"| {ledger.stats()}")
        if corrected_text != response_text:
            print(f"[CONTRA] corrected response: {corrected_text[:200]!r}")

        _capture_contra_case(
            session_id=session_id, turn_id=turn_id, action=action,
            product_refs=candidates, graph_before=graph_before,
            extracted_claims={a["article_id"]: {} for a in assertions},
            response_in=response_text, response_out=corrected_text,
            contradictions=contradictions, collection_prefix=collection_prefix,
        )

        return {
            "response_text":       corrected_text,
            "contradiction_found": len(contradictions) > 0,
            "contradiction_count": len(contradictions),
            "contradictions":      contradictions,
            "claims_stored":       len(candidates),
            # The products this reply actually spoke about — not every product
            # the session happens to remember. Downstream callers treat these as
            # "what this message is about", so the wide candidate set would be
            # actively misleading here.
            "product_ids":         [r["article_id"] for r in attributable],
            "product_names":       [r.get("name", "") for r in attributable],
            "revisions":           revisions,
            "revision_count":      len(revisions),
            "stale_fixes":         stale_fixes,
            "deferred_count":      deferred,
            "anchor_breakdown":    anchor_breakdown,
            "cross_turn_count":    anchor_breakdown.get("prior_assertion", 0),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty(self, response_text: str) -> dict:
        return {
            "response_text":       response_text,
            "contradiction_found": False,
            "contradiction_count": 0,
            "contradictions":      [],
            "claims_stored":       0,
            "product_ids":         [],
            "product_names":       [],
            "revisions":           [],
            "revision_count":      0,
            "stale_fixes":         [],
            "deferred_count":      0,
            "anchor_breakdown":    {},
            "cross_turn_count":    0,
        }
