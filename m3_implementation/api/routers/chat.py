# m3_implementation/api/routers/chat.py
import asyncio
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_memory_pipeline, get_rag_pipeline


# ── Fire-and-forget: send pipeline_output to friend modules ───────────────────
# Ports are configured in .env:
#   M2_MULTIMODAL_URL=http://127.0.0.1:8001
#   M1_GRAPH_URL=http://127.0.0.1:8002
import os as _os

_M2_URL = _os.getenv("M2_MULTIMODAL_URL", "")   # Friend A multimodal RAG
_M1_URL = _os.getenv("M1_GRAPH_URL",      "")   # Friend B graph RAG

async def _fire_and_forget(url: str, pipeline_output: dict, module_name: str):
    """
    Sends pipeline_output to a friend module's /api/process endpoint.
    Does NOT wait for response — your chatbot returns immediately to the user.
    The friend module processes independently in the background.
    """
    if not url:
        return  # Not configured — skip silently

    body = {
        "retrieval_input": pipeline_output.get("retrieval_input"),
        "memory_context":  pipeline_output.get("memory_context") or {},
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{url}/api/process", json=body)
        print(f"[CHAT] Sent pipeline_output to {module_name} ({url})")
    except Exception as e:
        # Non-fatal — friend module may not be running yet
        print(f"[CHAT] Could not reach {module_name} at {url}: {type(e).__name__}")


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id:            str
    customer_id:        str
    message:            str
    session_id:         Optional[str] = None   # None = start new session
    force_new_session:  bool = False            # True = ignore existing session, start fresh


@router.post("")
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Runs the full pipeline:
      1. Memory pipeline — classify + enrich
      2. Text RAG — evidence + generate + hallucination check
      3. Contradiction detector — cross-turn consistency
      4. Returns structured response to frontend

    Returns:
      response_text:         The bot's response to show
      session_id:            Session ID (new or existing)
      label:                 DistilBERT classification label
      confidence:            Classification confidence
      retrieval_strategy:    FULL / PARTIAL / NO
      action:                catalog_search / item_attribute_lookup / etc
      items_recommended:     List of product dicts (for product cards)
      product_ids:           Article IDs mentioned
      product_names:         Product names mentioned
      hallucination_flag:    True if unresolved hallucination
      contradiction_found:   True if contradiction was detected and fixed
      contradiction_count:   Number of contradictions found
    """
    memory = get_memory_pipeline()
    rag    = get_rag_pipeline()

    if not memory or not rag:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Server starting up."
        )

    print("\n" + "="*60)
    print(f"[CHAT] ━━━ NEW REQUEST ━━━")
    print(f"[CHAT] user_id={req.user_id[:20]}")
    print(f"[CHAT] message='{req.message[:80]}'")
    print(f"[CHAT] session_id={req.session_id} force_new={req.force_new_session}")
    try:
        # If force_new_session, clear Redis active session pointer first
        if req.force_new_session:
            try:
                from memory.db.redis_client import get_redis
                redis = await get_redis()
                await redis.delete(f"user:{req.user_id}:active_session")
                print(f"[Chat] Force new session for user {req.user_id}")
            except Exception as e:
                print(f"[Chat] Redis clear error (non-fatal): {e}")

        print(f"[CHAT] ─── Step 1: calling memory.process_turn...")
        # Step 1: Memory pipeline
        pipeline_output = await memory.process_turn(
            user_id=req.user_id,
            message=req.message,
            session_id=None if req.force_new_session else req.session_id,
            customer_id=req.customer_id,
        )

        print(f"[CHAT] ─── Memory pipeline done")
        import json as _json
        print("\n" + "="*60)
        print("[SPEC-CHECK] ━━━ FULL pipeline_output (vs retrieval_input_reference_v2) ━━━")
        _ri = pipeline_output.get("retrieval_input") or {}
        _mc = pipeline_output.get("memory_context") or {}
        _payload = _ri.get("payload") or {}
        print(f"[SPEC] user_id              = {pipeline_output.get('user_id','MISSING')[:30]}")
        print(f"[SPEC] session_id           = {pipeline_output.get('session_id','MISSING')}")
        print(f"[SPEC] label                = {pipeline_output.get('label','MISSING')}")
        print(f"[SPEC] confidence           = {pipeline_output.get('confidence','MISSING')}")
        print(f"[SPEC] retrieval_strategy   = {pipeline_output.get('retrieval_strategy','MISSING')}")
        print(f"[SPEC] classifier_input     = {str(pipeline_output.get('classifier_input','MISSING'))[:80]}")
        print(f"[SPEC] side_effects         = {pipeline_output.get('side_effects','MISSING')}")
        print(f"[SPEC-RI] retrieval_input fields:")
        if _ri:
            print(f"  action                  = {_ri.get('action','MISSING')}")
            print(f"  retrieval_strategy      = {_ri.get('retrieval_strategy','MISSING')}")
            print(f"  user_message            = {str(_ri.get('user_message','MISSING'))[:80]}")
            _ctx = _ri.get("items_in_context") or {}
            print(f"  items_in_context.item_a = {_ctx.get('item_a','None')}")
            print(f"  items_in_context.item_b = {_ctx.get('item_b','None')}")
            print(f"  exclude_ids             = {_ri.get('exclude_ids','MISSING')}")
            print(f"[SPEC-PAYLOAD] payload fields (action={_ri.get('action','?')})")
            if _ri.get("action") in ("catalog_search",):
                print(f"  filters                 = {_payload.get('filters','MISSING')}")
                print(f"  soft_constraints [NEW]  = {_payload.get('soft_constraints','MISSING')}")
                print(f"  preference_boosts       = {_payload.get('preference_boosts','MISSING')}")
                _ph = _payload.get("purchase_history_hints") or {}
                print(f"  purchase_history_hints [NEW]:")
                print(f"    top_colours           = {_ph.get('top_colours','MISSING')}")
                print(f"    top_product_types     = {_ph.get('top_product_types','MISSING')}")
                print(f"    inferred_gender       = {_ph.get('inferred_gender','MISSING')}")
                print(f"    budget_tier           = {_ph.get('budget_tier','MISSING')}")
                print(f"    preferred_price_range = {_ph.get('preferred_price_range','MISSING')}")
                print(f"    dominant_colour       = {_ph.get('dominant_colour','MISSING')}")
                print(f"    dominant_type         = {_ph.get('dominant_type','MISSING')}")
                print(f"  penalties               = {_payload.get('penalties','MISSING')}")
            elif _ri.get("action") == "item_attribute_lookup":
                print(f"  article_id              = {_payload.get('article_id','MISSING')}")
                print(f"  attribute_topic         = {_payload.get('attribute_topic','MISSING')}")
            elif _ri.get("action") == "item_compare":
                print(f"  article_id_a            = {_payload.get('article_id_a','MISSING')}")
                print(f"  article_id_b            = {_payload.get('article_id_b','MISSING')}")
                print(f"  comparison_dimension    = {_payload.get('comparison_dimension','MISSING')}")
                print(f"  preference_weights      = {_payload.get('preference_weights','MISSING')}")
            elif _ri.get("action") == "explanation_generate":
                print(f"  article_id              = {_payload.get('article_id','MISSING')}")
                print(f"  prior_claims            = {_payload.get('prior_claims','MISSING')}")
                print(f"  matched_prefs           = {_payload.get('matched_prefs','MISSING')}")
            elif _ri.get("action") == "item_detail_lookup":
                print(f"  article_id              = {_payload.get('article_id','MISSING')}")
        else:
            print("  retrieval_input = None (FEEDBACK/CHITCHAT — correct per spec)")
            _fb = _mc.get("feedback") or {}
            if _fb:
                print(f"  memory_context.feedback.sentiment_score = {_fb.get('sentiment_score','MISSING')}")
                print(f"  memory_context.feedback.is_positive     = {_fb.get('is_positive','MISSING')}")
                print(f"  memory_context.feedback.feedback_type   = {_fb.get('feedback_type','MISSING')}")
                print(f"  memory_context.feedback.item_reacted_to = {_fb.get('item_reacted_to','MISSING')}")
        print("[SPEC-CHECK] ━━━ end pipeline_output ━━━")
        print(f"[CHAT] pipeline_output keys: {list(pipeline_output.keys())}")
        print(f"[CHAT] label={pipeline_output.get('label')} conf={pipeline_output.get('confidence',0):.1%} strategy={pipeline_output.get('retrieval_strategy')}")
        _ri_tmp = pipeline_output.get("retrieval_input") or {}
        print(f"[CHAT] session_id={pipeline_output.get('session_id','?')} action={_ri_tmp.get('action','NO_RETRIEVAL')}")
        _ri = pipeline_output.get("retrieval_input") or {}
        _payload_dbg = _ri.get('payload') or {}
        print(f"[CHAT] filters={_payload_dbg.get('filters',{})}")
        print(f"[CHAT] soft_constraints={_payload_dbg.get('soft_constraints',{})}")
        print(f"[CHAT] purchase_hints_present={bool(_payload_dbg.get('purchase_history_hints'))}")
        _cse = pipeline_output.get("cse", {})
        if _cse:
            print(f"[CHAT] CSE: score={_cse.get('sufficiency_score','?'):.4f} "
                  f"tier={_cse.get('tier','?')} override={_cse.get('override','?')}")
            print(f"[CHAT] CSE: D1_ref={_cse.get('d1_referent','?'):.2f} "
                  f"D2_pred={_cse.get('d2_predicate','?'):.2f} "
                  f"D3_cat={_cse.get('d3_catalog_needed','?'):.2f} "
                  f"D4_param={_cse.get('d4_parametric','?'):.2f} "
                  f"D5_known={_cse.get('d5_item_set_known','?'):.2f}")
        # ── Fire and forget: send to friend modules ────────────────────────
        import asyncio as _asyncio
        _asyncio.ensure_future(_fire_and_forget(_M2_URL, pipeline_output, "M2 Multimodal RAG"))
        _asyncio.ensure_future(_fire_and_forget(_M1_URL, pipeline_output, "M1 Graph RAG"))

        print(f"[CHAT] ─── Step 2: calling rag.process...")
        # Step 2: Text RAG pipeline (includes hallucination + contradiction)
        rag_result = await rag.process(
            pipeline_output=pipeline_output,
            memory_pipeline=memory,
            store_response=True,
        )

        print(f"[CHAT] ─── RAG done")
        print(f"[CHAT] rag_result keys: {list(rag_result.keys())}")
        print(f"[CHAT] response_text: '{rag_result.get('response_text','')[:100]}'")
        print(f"[CHAT] action={rag_result.get('action')} items={len(rag_result.get('items_recommended',[]))} hall={rag_result.get('hallucination_flag')} contra={rag_result.get('contradiction_found')}")
        for _itm in rag_result.get("items_recommended",[]):
            print(f"[CHAT] ITEM: {_itm.get('article_id','?')} | {str(_itm.get('name','?'))[:30]} | {_itm.get('colour','?')} | {_itm.get('price','?')}")
        # Extract items for product cards
        items = []
        for item in rag_result.get("items_recommended", []):
            if item.get("article_id") or item.get("name"):
                items.append({
                    "article_id":  str(item.get("article_id", "")),
                    "name":        item.get("name", ""),
                    "colour":      item.get("colour", ""),
                    "type":        item.get("type", ""),
                    "price":       item.get("price", ""),
                    "description": (item.get("material_description") or "")[:120],
                    "pattern":     item.get("pattern", ""),
                })

        print(f"[CHAT] ─── Returning final response to frontend")
        return {
            "response_text":       rag_result.get("response_text", ""),
            "session_id":          pipeline_output.get("session_id", ""),
            "label":               pipeline_output.get("label", ""),
            "confidence":          round(pipeline_output.get("confidence", 0), 4),
            "retrieval_strategy":  pipeline_output.get("retrieval_strategy", "NO"),
            "action":              rag_result.get("action", ""),
            "items_recommended":   items,
            "product_ids":         rag_result.get("product_ids", []),
            "product_names":       rag_result.get("product_names", []),
            "hallucination_flag":  rag_result.get("hallucination_flag", False),
            "hallucination_score": rag_result.get("hallucination_score", 0.0),
            "attempt_count":       rag_result.get("attempt_count", 1),
            "contradiction_found": rag_result.get("contradiction_found", False),
            "contradiction_count": rag_result.get("contradiction_count", 0),
            "contradictions":      rag_result.get("contradictions", []),
            "cse":                 pipeline_output.get("cse", {}),
        }

    except Exception as e:
        print(f"[Chat API] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
