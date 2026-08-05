# m3_implementation/text_rag/core/rag_pipeline.py
#
# Main entry point for the Text RAG system.
#
# COMPLETE FLOW:
#   1. Receive pipeline output (retrieval_input + memory_context)
#   2. EvidenceAssembler queries PostgreSQL + Qdrant → evidence bundle
#   3. ResponseGenerator builds action-specific prompt → Ollama LLM → response
#   4. HallucinationChecker runs NLI sentence-level validation
#   5. If hallucination detected → regenerate with stricter prompt
#      (up to MAX_REGENERATION_ATTEMPTS = 3 times)
#   6. If still hallucinating after 3 attempts → return with flag
#   7. ContradictionDetector reconciles the response against the session's
#      Assertion Ledger and the live catalogue — cross-turn drift is corrected,
#      catalogue revisions are recorded against the turns they affect
#   8. store_response() called on the memory pipeline
#   9. Return final structured result to caller
#
# STEPS 4 AND 7 ASK DIFFERENT QUESTIONS:
#   step 4  is this turn faithful to ITS OWN evidence?
#   step 7  is this turn consistent with what we already told this user, and
#           is the catalogue still saying what it said back then?
#   Attributes covered by step 4 are deliberately not re-judged in step 7.
#
# OUTPUT STRUCTURE:
#   {
#       response_text:        str   — the response to show to user
#       hallucination_flag:   bool  — True if unresolved hallucination
#       flagged_sentences:    list  — sentences that failed NLI check
#       hallucination_score:  float — 0.0-1.0 severity
#       attempt_count:        int   — how many attempts were made
#       contradiction_found:  bool  — cross-turn drift detected and corrected
#       revisions:            list  — catalogue values that changed mid-session
#       action:               str   — which action was triggered
#       items_recommended:    list  — for catalog_search, items returned
#       evidence_used:        dict  — evidence bundle for audit/debugging
#   }

import os
import sys
import asyncio
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from text_rag.config import MAX_REGENERATION_ATTEMPTS
from text_rag.core.evidence_assembler    import EvidenceAssembler
from text_rag.core.response_generator    import ResponseGenerator
from text_rag.core.hallucination_checker import HallucinationChecker
from memory.core.contradiction_detector  import ContradictionDetector

# Evaluation capture hook — no-op unless EVAL_CAPTURE=1
# (see test_result/hallucination_result)
try:
    from test_result.hallucination_result.capture import capture_case as _capture_case
except Exception:
    def _capture_case(**_kwargs):
        pass

# Actions that skip hallucination checking (no factual product claims)
_SKIP_HALLUCINATION_CHECK = {"no_retrieval"}

# Actions where we ALWAYS check (factual product claims made)
_ALWAYS_CHECK = {
    "catalog_search",
    "item_attribute_lookup",
    "item_compare",
    "explanation_generate",
    "item_detail_lookup",
}


class TextRAGPipeline:
    """
    Main Text RAG pipeline. Instantiate once at startup.

    Usage:
        rag = TextRAGPipeline()
        result = await rag.process(pipeline_output, memory_pipeline)
        # result["response_text"] → send to user
        # result["hallucination_flag"] → log or display warning
    """

    def __init__(self):
        self.assembler    = EvidenceAssembler()
        self.generator    = ResponseGenerator()
        self.checker      = HallucinationChecker()
        self.contradector = ContradictionDetector()
        print("[TextRAGPipeline] Initialised.")

    async def process(
        self,
        pipeline_output:   dict,
        memory_pipeline=None,
        store_response:    bool = True,
        collection_prefix: str  = "m3",
    ) -> dict:
        """
        Processes one turn through the complete Text RAG pipeline.

        Args:
            pipeline_output:  The dict returned by memory_pipeline.process_turn()
                              Must contain: retrieval_input, memory_context,
                              user_id, session_id, label, retrieval_strategy
            memory_pipeline:  The MemoryPipeline instance (for store_response)
            store_response:   Whether to call store_response() after generation

        Returns the structured result dict described in this module's docstring.
        """
        retrieval_input = pipeline_output.get("retrieval_input")
        memory_context  = pipeline_output.get("memory_context", {})
        user_id         = pipeline_output.get("user_id", "")
        session_id      = pipeline_output.get("session_id", "")
        label           = pipeline_output.get("label", "CHITCHAT")
        strategy        = pipeline_output.get("retrieval_strategy", "NO")

        print("\n" + "="*60)
        print(f"[RAG] ━━━ process() called ━━━")
        print(f"[RAG] label={label} strategy={strategy}")
        print(f"[RAG] session={session_id[:12] if session_id else '?'}")
        _ri_dbg = retrieval_input or {}
        print(f"[RAG] retrieval_input action={_ri_dbg.get('action','None')}")
        _ri_payload = _ri_dbg.get("payload") or {}
        print(f"[RAG] filters={_ri_payload.get('filters',{})}")
        # ── Step 1: Handle not-relevant / session-context-blocked inputs ───────
        if memory_context.get("not_relevant") or memory_context.get("session_context_blocked"):
            refusal = memory_context.get(
                "refusal_message",
                "I can only help with fashion and clothing recommendations."
            )
            return self._build_result(
                response_text=refusal,
                action="no_retrieval",
                attempt_count=0,
                hallucination_flag=False,
                flagged_sentences=[],
                hallucination_score=0.0,
                items_recommended=[],
                evidence={},
            )

        # ── Step 1.5: Cached recommendation (similar-question hit) ────────────
        # When the CSE detects a near-identical INITIAL_REQUEST in the same session,
        # pipeline.py injects the previous recommendation into memory_context so we
        # can re-present it without hitting the assembler / Qdrant / PostgreSQL again.
        if memory_context.get("is_cached_recommendation"):
            cached_items_raw = memory_context.get("cached_recommendation", [])
            print(f"[RAG] cached recommendation hit — {len(cached_items_raw)} items, skipping assembler")

            # Map ItemInContext field names → assembler/generator field names
            mapped_items = []
            for it in cached_items_raw:
                mapped_items.append({
                    "article_id":           str(it.get("article_id", "")),
                    "name":                 it.get("prod_name") or it.get("name") or "",
                    "type":                 it.get("product_type_name") or it.get("type") or "",
                    "colour":               it.get("colour_group_name") or it.get("colour") or "",
                    "index_group":          it.get("index_group_name") or it.get("index_group") or None,
                    "section":              it.get("section_name") or it.get("section") or None,
                    "garment_group":        it.get("garment_group_name") or it.get("garment_group") or None,
                    "material_description": it.get("detail_desc") or it.get("material_description") or None,
                    "pattern":              it.get("graphical_appearance_name") or it.get("pattern") or None,
                    "price":                it.get("price"),
                    # Carry the stored justification through so a re-presented
                    # recommendation keeps its "why" instead of losing it.
                    "why":                  it.get("why") or [],
                    "match_percent":        it.get("match_percent"),
                })

            cached_evidence = {"action": "catalog_search", "items": mapped_items}

            try:
                cached_response = await self.generator.generate(
                    evidence=cached_evidence,
                    strictness=0,
                )
            except Exception as e:
                print(f"[RAG] cached generation error: {e}")
                cached_response = self._fallback_response("catalog_search", cached_evidence)

            if not cached_response:
                cached_response = self._fallback_response("catalog_search", cached_evidence)

            # A re-presented recommendation is built entirely from session cache
            # and never touches PostgreSQL, so it is the single most likely place
            # to quote a price or colour the catalogue has since changed. Running
            # the reconciler here is what lets that be caught and annotated
            # instead of being served stale forever.
            cached_contra = await self._reconcile(
                response_text=cached_response,
                evidence=cached_evidence,
                pipeline_output=pipeline_output,
                retrieval_input=retrieval_input,
                collection_prefix=collection_prefix,
            )
            cached_response = cached_contra["response_text"]

            cached_response += "\n\nWould you like to see new recommendations for this?"

            if store_response and memory_pipeline and session_id and user_id:
                try:
                    rec_items = [
                        {
                            "article_id":                item["article_id"],
                            "prod_name":                 item.get("name") or "",
                            "product_type_name":         item.get("type") or "",
                            "colour_group_name":         item.get("colour") or "",
                            "price":                     item.get("price"),
                            "index_group_name":          item.get("index_group") or None,
                            "section_name":              item.get("section") or None,
                            "garment_group_name":        item.get("garment_group") or None,
                            "detail_desc":               item.get("material_description") or None,
                            "graphical_appearance_name": item.get("pattern") or None,
                            "why":                       item.get("why") or None,
                            "match_percent":             item.get("match_percent"),
                        }
                        for item in mapped_items if item.get("article_id")
                    ]
                    await memory_pipeline.store_response(
                        session_id=session_id,
                        user_id=user_id,
                        bot_response=cached_response,
                        recommended_items=rec_items if rec_items else None,
                        trigger_label=label,
                        retrieval_strategy="PARTIAL",
                    )
                except Exception as e:
                    print(f"[RAG] cached store_response error: {e}")

            return self._build_result(
                response_text=cached_response,
                action="catalog_search",
                attempt_count=1,
                hallucination_flag=False,
                flagged_sentences=[],
                hallucination_score=0.0,
                items_recommended=mapped_items,
                evidence=cached_evidence,
                contradiction_found=cached_contra["contradiction_found"],
                contradiction_count=cached_contra["contradiction_count"],
                contradictions_detail=cached_contra["contradictions"],
                product_ids=cached_contra["product_ids"],
                product_names=cached_contra["product_names"],
                revisions=cached_contra.get("revisions", []),
            )

        # ── Step 2: Assemble evidence ───────────────────────────────────────
        _ri_a = (retrieval_input or {}).get("action", "NO_INPUT")
        print(f"\n[DBG-4] EVIDENCE ASSEMBLY: action={_ri_a}")
        print(f"[RAG] ─── Step 2: assembling evidence...")
        try:
            evidence = await self.assembler.assemble(
                retrieval_input=retrieval_input,
                memory_context=memory_context,
            )
        except Exception as e:
            print(f"[TextRAGPipeline] Evidence assembly error: {e}")
            evidence = {"action": "no_retrieval", "error": str(e)}

        action = evidence.get("action", "no_retrieval")
        _ev_items = evidence.get("items", [])
        print(f"[RAG] evidence: action={action} items={len(_ev_items)}")
        for _ei in _ev_items:
            print(f"  [RAG-ITEM] {str(_ei.get('article_id','?'))[:12]} | {str(_ei.get('name',_ei.get('prod_name','?')))[:30]} | {_ei.get('colour',_ei.get('colour_group_name','?'))} | {_ei.get('price',_ei.get('avg_price','?'))}")
        _ev_items = evidence.get("items", [])
        print(f"[DBG-4] EVIDENCE DONE: action={action} items={len(_ev_items)}")
        [print(f"  [DBG-4] ITEM: {it.get('article_id','?')} | {str(it.get('name','?'))[:30]} | {it.get('colour','?')} | {it.get('price','')}" ) for it in _ev_items]

        # ── Step 3 + 4 + 5: Generate → Check → Regenerate loop ─────────────
        skip_checking       = action in _SKIP_HALLUCINATION_CHECK
        print(f"[RAG] skip_hallucination={skip_checking} (action={action})")
        response_text       = ""
        check_result        = {}
        attempt_count       = 0
        final_response      = ""
        final_flag          = False
        final_flagged       = []
        final_score         = 0.0
        contradicted_fields = []

        for attempt in range(MAX_REGENERATION_ATTEMPTS):
            attempt_count = attempt + 1
            strictness    = attempt  # 0=normal, 1=strict, 2=strictest

            # Guard: 0 items in evidence means the LLM will hallucinate product
            # names and prices — skip generation entirely and return a honest message.
            if action == "catalog_search" and not evidence.get("items"):
                print("[RAG] 0 items in evidence — skipping LLM to prevent hallucination")
                final_response = (
                    "Sorry, we don't have any products matching that description at the moment. "
                    "Try searching for a different product type, or remove specific filters "
                    "such as price range or category to see more options."
                )
                final_flag = False
                break

            # Generate response
            try:
                print(f"[DBG-5] OLLAMA GENERATE: attempt={attempt_count} strictness={strictness} action={action}")
                print(f"[RAG] ─── generate: attempt={attempt_count} strictness={strictness} action={action}")
                response_text = await self.generator.generate(
                    evidence=evidence,
                    strictness=strictness,
                    contradicted_fields=contradicted_fields if strictness > 0 else None,
                )
            except Exception as e:
                print(f"[TextRAGPipeline] Generation error (attempt {attempt_count}): {e}")
                response_text = self._fallback_response(action, evidence)

            if not response_text:
                response_text = self._fallback_response(action, evidence)

            # Skip NLI check for non-factual actions
            if skip_checking:
                final_response = response_text
                final_flag     = False
                break

            # Check for hallucinations
            try:
                check_result = self.checker.check(response_text, evidence)
            except Exception as e:
                print(f"[TextRAGPipeline] Hallucination check error: {e}")
                check_result = {"passed": True, "has_hallucination": False,
                                "flagged_sentences": [], "hallucination_score": 0.0,
                                "contradicted_fields": []}

            # Evaluation capture — persists (evidence, response, check) pairs
            # for the hallucination evaluation test set. No-op unless EVAL_CAPTURE=1.
            _capture_case(
                evidence=evidence,
                response_text=response_text,
                action=action,
                attempt=attempt_count,
                session_id=session_id,
                user_message=_ri_dbg.get("user_message", ""),
                check_result=check_result,
            )

            if check_result.get("passed", True):
                # Passed — no hallucinations
                final_response = response_text
                final_flag     = False
                final_flagged  = []
                final_score    = check_result.get("hallucination_score", 0.0)
                print(f"[RAG-HALL] ✓ PASSED attempt={attempt_count} chars={len(response_text)}")
                print(f"[DBG-6] HALL PASS: response_len={len(response_text)} chars")
                break
            else:
                # Failed — capture which fields were wrong for the next attempt's prompt
                contradicted_fields = check_result.get("contradicted_fields", [])
                final_flagged       = check_result.get("flagged_sentences", [])
                final_score         = check_result.get("hallucination_score", 0.0)
                print(
                    f"[TextRAGPipeline] Hallucination detected on attempt {attempt_count}. "
                    f"Score: {final_score:.3f}. "
                    f"Flagged: {check_result.get('n_flagged', 0)} sentences. "
                    f"Contradicted fields: {contradicted_fields}. "
                    f"Retrying with strictness={strictness + 1}..."
                )
                final_response = response_text
                final_flag     = True

                if attempt == MAX_REGENERATION_ATTEMPTS - 1:
                    # Final attempt — keep response but mark as flagged
                    print(
                        f"[TextRAGPipeline] WARNING: Hallucination unresolved after "
                        f"{MAX_REGENERATION_ATTEMPTS} attempts. Returning with flag."
                    )

        # ── Step 6: Cross-turn consistency reconciliation ───────────────────
        # The hallucination guard has just verified this turn against its OWN
        # evidence. This step verifies it against the session's Assertion
        # Ledger — everything the system has already told this user — and
        # against the live catalogue, which may have changed since those
        # earlier turns. retrieval_input is passed so the reconciler can still
        # identify the products under discussion when the evidence bundle names
        # none of them (follow-up turns answered from session memory).
        contra_result = await self._reconcile(
            response_text=final_response,
            evidence=evidence,
            pipeline_output=pipeline_output,
            retrieval_input=retrieval_input,
            collection_prefix=collection_prefix,
        )
        final_response        = contra_result["response_text"]
        contradiction_found   = contra_result["contradiction_found"]
        contradiction_count   = contra_result["contradiction_count"]
        contradictions_detail = contra_result["contradictions"]
        product_ids           = contra_result["product_ids"]
        product_names         = contra_result["product_names"]
        revisions             = contra_result.get("revisions", [])

        # ── Step 7: Store response in memory ─────────────────────────────────
        items_recommended = []
        if action == "catalog_search":
            items_recommended = evidence.get("items", [])

        if store_response and memory_pipeline and session_id and user_id:
            try:
                rec_items = []
                for item in items_recommended:
                    if item.get("article_id"):
                        rec_items.append({
                            "article_id":                str(item["article_id"]),
                            "prod_name":                 item.get("name") or "",
                            "product_type_name":         item.get("type") or "",
                            "colour_group_name":         item.get("colour") or "",
                            "price":                     item.get("price_raw"),
                            "index_group_name":          item.get("index_group") or None,
                            "section_name":              item.get("section") or None,
                            "garment_group_name":        item.get("garment_group") or None,
                            "detail_desc":               item.get("material_description") or None,
                            "graphical_appearance_name": item.get("pattern") or None,
                            # Persist the ranking justification so reopening this
                            # chat later still shows why each item was picked.
                            "why":                       item.get("why") or None,
                            "match_percent":             item.get("match_percent"),
                        })

                await memory_pipeline.store_response(
                    session_id=session_id,
                    user_id=user_id,
                    bot_response=final_response,
                    recommended_items=rec_items if rec_items else None,
                    trigger_label=label,
                    retrieval_strategy=strategy,
                )
            except Exception as e:
                print(f"[TextRAGPipeline] store_response error: {e}")

        print(f"\n[RAG-OUT] ━━━ final result ━━━")
        print(f"[RAG-OUT] response: {repr(final_response[:150])}")
        print(f"[RAG-OUT] action={action} items={len(items_recommended)} hall={final_flag} contra={contradiction_found}")
        print(f"[RAG-OUT] product_ids={product_ids}")
        return self._build_result(
            response_text=final_response,
            action=action,
            attempt_count=attempt_count,
            hallucination_flag=final_flag,
            flagged_sentences=final_flagged,
            hallucination_score=final_score,
            items_recommended=items_recommended,
            evidence=evidence,
            contradiction_found=contradiction_found,
            contradiction_count=contradiction_count,
            contradictions_detail=contradictions_detail,
            product_ids=product_ids,
            product_names=product_names,
            revisions=revisions,
        )

    async def _reconcile(
        self,
        response_text:     str,
        evidence:          dict,
        pipeline_output:   dict,
        retrieval_input:   dict,
        collection_prefix: str,
    ) -> dict:
        """
        Runs the cross-turn consistency check, never letting a failure there
        block the response. Returns the detector's own empty result on error so
        callers always see the same keys.
        """
        try:
            result = await self.contradector.check_and_resolve(
                response_text=response_text,
                evidence=evidence,
                session_id=pipeline_output.get("session_id", ""),
                user_id=pipeline_output.get("user_id", ""),
                turn_id=pipeline_output.get("turn_id", ""),
                collection_prefix=collection_prefix,
                retrieval_input=retrieval_input,
            )
            print(f"[DBG-7] CONTRADICTION: found={result['contradiction_found']} "
                  f"count={result['contradiction_count']} "
                  f"cross_turn={result.get('cross_turn_count', 0)} "
                  f"anchors={result.get('anchor_breakdown', {})} "
                  f"revisions={result.get('revision_count', 0)} "
                  f"deferred={result.get('deferred_count', 0)} "
                  f"claims_stored={result.get('claims_stored', 0)}")
            return result
        except Exception as e:
            print(f"[TextRAGPipeline] Cross-turn reconciliation error: {e}")
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
                "deferred_count":      0,
            }

    def _build_result(
        self,
        response_text:        str,
        action:               str,
        attempt_count:        int,
        hallucination_flag:   bool,
        flagged_sentences:    list,
        hallucination_score:  float,
        items_recommended:    list,
        evidence:             dict,
        contradiction_found:  bool = False,
        contradiction_count:  int  = 0,
        contradictions_detail:list = None,
        product_ids:          list = None,
        product_names:        list = None,
        revisions:            list = None,
    ) -> dict:
        return {
            # ── Response ───────────────────────────────────────────────
            "response_text":          response_text,

            # ── Hallucination check results ────────────────────────────
            "hallucination_flag":     hallucination_flag,
            "hallucination_score":    hallucination_score,
            "flagged_sentences":      [
                {"sentence": f["sentence"], "score": f.get("nli_scores", {})}
                for f in flagged_sentences
            ],
            "attempt_count":          attempt_count,

            # ── Cross-turn consistency results ─────────────────────────
            "contradiction_found":    contradiction_found,
            "contradiction_count":    contradiction_count,
            "contradictions":         contradictions_detail or [],

            # Catalogue values that changed mid-session. Not errors — the
            # earlier statements were true when made — so they are reported
            # separately and annotated against the turns that quoted them.
            "revisions":              revisions or [],

            # ── Product references ─────────────────────────────────────
            "product_ids":            product_ids or [],
            "product_names":          product_names or [],

            # ── Action metadata ────────────────────────────────────────
            "action":                 action,
            "items_recommended":      items_recommended,
        }

    def _fallback_response(self, action: str, evidence: dict) -> str:
        """Safe fallback when generation fails."""
        if action == "catalog_search":
            items = evidence.get("items", [])
            if items:
                names = [i.get("name", "item") for i in items[:2]]
                return f"I found these options for you: {' and '.join(names)}."
        elif action == "item_attribute_lookup":
            article = evidence.get("article") or {}
            return (f"Here are the details for {article.get('name','the item')}: "
                    f"{(article.get('material_description') or 'No description available')[:150]}")
        elif action == "no_retrieval":
            return evidence.get("refusal_message", "How can I help you today?")
        return "I'm here to help with fashion recommendations. What are you looking for?"


# ── Setup helper ───────────────────────────────────────────────────────────────

async def setup_databases(force_reload: bool = False):
    """
    Initialises PostgreSQL and Qdrant databases.
    Run this once at application startup.

    Steps:
      1. Create PostgreSQL schema
      2. Load 41,794 articles into PostgreSQL
      3. Create Qdrant collection
      4. Index all articles as vectors in Qdrant

    Args:
        force_reload: If True, clears and reloads everything
    """
    from text_rag.db.postgres_client import create_schema, load_articles, get_pool, close_pool
    from text_rag.db.qdrant_client   import create_collection, index_articles

    print("\n[Setup] Step 1/4: PostgreSQL schema...")
    await create_schema()

    print("[Setup] Step 2/4: Loading articles into PostgreSQL...")
    pg_count = await load_articles(force_reload=force_reload)
    print(f"[Setup] PostgreSQL: {pg_count} articles ready.")

    print("[Setup] Step 3/4: Qdrant collection...")
    create_collection()

    print("[Setup] Step 4/4: Indexing articles in Qdrant (this may take 10-15 mins)...")
    qdrant_count = index_articles(force_reload=force_reload)
    print(f"[Setup] Qdrant: {qdrant_count} articles indexed.")

    await close_pool()
    print("[Setup] All databases ready.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Text RAG Pipeline setup and test")
    parser.add_argument("--setup", action="store_true", help="Setup databases")
    parser.add_argument("--force", action="store_true", help="Force reload")
    args = parser.parse_args()

    if args.setup:
        asyncio.run(setup_databases(force_reload=args.force))
