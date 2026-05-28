# m3_implementation/api/routers/chat.py
import asyncio
import json as _json
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_memory_pipeline, get_rag_pipeline
from memory.core.rl_signal_collector import get_rl_collector, LABEL_NAME_TO_ID


def _make_json_safe(obj: dict) -> dict:
    """
    Strip all non-JSON-serializable values (datetime, Pydantic models, etc.)
    by serialising through json.dumps(default=str) → json.loads.
    This converts datetime → ISO string, and any unknown type → str.
    """
    return _json.loads(_json.dumps(obj, default=str))


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

    body = _make_json_safe({
        "retrieval_input": pipeline_output.get("retrieval_input"),
        "memory_context":  pipeline_output.get("memory_context") or {},
    })
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{url}/api/process", json=body)
        print(f"[CHAT] Sent pipeline_output to {module_name} ({url})")
    except Exception as e:
        # Non-fatal — friend module may not be running yet
        print(f"[CHAT] Could not reach {module_name} at {url}: {type(e).__name__}")


async def _call_m2_sync(pipeline_output: dict, timeout: float = 30.0):
    """Calls M2 synchronously.
    Returns the response dict on both success=True and success=False (M2 is reachable).
    Returns None only when M2 cannot be reached at all (timeout, connection error).
    """
    if not _M2_URL:
        return None
    body = _make_json_safe({
        "retrieval_input": pipeline_output.get("retrieval_input"),
        "memory_context":  pipeline_output.get("memory_context") or {},
    })
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{_M2_URL}/api/process", json=body)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                print(f"[CHAT] M2 response received: action={data.get('action')} items={len(data.get('items', []))}")
            else:
                print(f"[CHAT] M2 application error: {data.get('error')}")
            return data  # always return — None only means unreachable
    except Exception as e:
        print(f"[CHAT] M2 sync call failed ({type(e).__name__}): {e}")
        return None


async def _call_m1_sync(pipeline_output: dict, timeout: float = 30.0):
    """Calls M1 synchronously.
    Returns the response dict on both success=True and success=False (M1 is reachable).
    Returns None only when M1 cannot be reached at all (timeout, connection error).
    """
    if not _M1_URL:
        return None
    body = _make_json_safe({
        "retrieval_input": pipeline_output.get("retrieval_input"),
        "memory_context":  pipeline_output.get("memory_context") or {},
    })
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{_M1_URL}/api/process", json=body)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                print(f"[CHAT] M1 response received: action={data.get('action')} items={len(data.get('items', []))}")
            else:
                print(f"[CHAT] M1 application error: {data.get('error')}")
            return data  # always return — None only means unreachable
    except Exception as e:
        print(f"[CHAT] M1 sync call failed ({type(e).__name__}): {e}")
        return None


def _remap_member_item(item: dict) -> dict:
    """
    Normalise a single item from M1/M2 response to the field names
    chat.py's item extraction loop expects.
    M2 uses prod_name/colour_group_name/product_type_name/detail_desc;
    M1 may differ — try both names so either works.
    """
    price_raw = item.get("price") or item.get("avg_price")
    if price_raw and not isinstance(price_raw, str):
        price_str = f"£{price_raw}"
    else:
        price_str = price_raw or ""
    return {
        "article_id":  str(item.get("article_id", "")),
        "name":        item.get("name")    or item.get("prod_name", ""),
        "colour":      item.get("colour")  or item.get("colour_group_name", ""),
        "type":        item.get("type")    or item.get("product_type_name", ""),
        "price":       price_str,
        "description": (item.get("description") or item.get("detail_desc") or "")[:120],
        "pattern":     item.get("pattern") or item.get("graphical_appearance_name", ""),
        "explanation": item.get("explanation", ""),
        "score":       item.get("score"),
        "image_url":   item.get("image_url", ""),
    }


def _normalize_member_response(data: dict) -> dict:
    """Convert M1/M2 API response to the same structure as rag.process() output.
    When success=False, passes M2/M1's own error message as response_text.
    """
    if not data.get("success", True):
        # M2/M1 is reachable but couldn't fulfil the request (e.g. article not found).
        # Surface their error message directly — do not show "unavailable".
        error_text = data.get("error") or data.get("message") or "The selected module could not process this request."
        return {
            "response_text":       error_text,
            "hallucination_flag":  False,
            "hallucination_score": 0.0,
            "flagged_sentences":   [],
            "attempt_count":       1,
            "contradiction_found": False,
            "contradiction_count": 0,
            "contradictions":      [],
            "product_ids":         [],
            "product_names":       [],
            "action":              data.get("action", ""),
            "items_recommended":   [],
        }
    remapped_items = [_remap_member_item(i) for i in data.get("items", [])]
    return {
        "response_text":       data.get("response_text", data.get("message", "")),
        "hallucination_flag":  data.get("hallucination_flag", False),
        "hallucination_score": data.get("hallucination_score", 0.0),
        "flagged_sentences":   data.get("flagged_sentences", []),
        "attempt_count":       data.get("attempt_count", 1),
        "contradiction_found": data.get("contradiction_found", False),
        "contradiction_count": data.get("contradiction_count", 0),
        "contradictions":      data.get("contradictions", []),
        "product_ids":         data.get("product_ids") or [i["article_id"] for i in remapped_items if i["article_id"]],
        "product_names":       data.get("product_names") or [i["name"] for i in remapped_items if i["name"]],
        "action":              data.get("action", ""),
        "items_recommended":   remapped_items,
    }


def _member_unavailable_response(member_label: str, pipeline_output: dict) -> dict:
    """Return a user-facing response dict when a member module cannot be reached."""
    return {
        "response_text":       f"{member_label} is currently unavailable. Please try again later or start a new chat and select a different model.",
        "session_id":          pipeline_output.get("session_id", ""),
        "label":               pipeline_output.get("label", ""),
        "confidence":          round(pipeline_output.get("confidence", 0), 4),
        "retrieval_strategy":  pipeline_output.get("retrieval_strategy", "NO"),
        "action":              "",
        "items_recommended":   [],
        "product_ids":         [],
        "product_names":       [],
        "hallucination_flag":  False,
        "hallucination_score": 0.0,
        "attempt_count":       1,
        "contradiction_found": False,
        "contradiction_count": 0,
        "contradictions":      [],
        "cse":                 pipeline_output.get("cse", {}),
        "recommendation_id":   None,
        "turn_id":             pipeline_output.get("turn_id", ""),
    }


async def _enrich_member_items(items: list) -> list:
    """
    M2/M1 responses often only contain article_id without name/colour/price.
    This fetches the full product details from the local PostgreSQL catalog
    and maps them to the field names chat.py's item extraction expects.
    Non-fatal — returns original items if lookup fails.
    """
    if not items:
        return items
    if all(item.get("name") for item in items):
        return items  # already complete, no lookup needed

    article_ids = [str(item.get("article_id", "")) for item in items if item.get("article_id")]
    if not article_ids:
        return items

    try:
        from text_rag.db.postgres_client import get_articles_by_ids
        pg_rows = await get_articles_by_ids(article_ids)
        pg_map = {str(row["article_id"]): row for row in pg_rows}
        print(f"[CHAT] item enrichment: fetched {len(pg_map)} product(s) from catalog for ids={article_ids}")

        enriched = []
        for item in items:
            aid = str(item.get("article_id", ""))
            pg = pg_map.get(aid, {})
            avg_price = pg.get("avg_price")
            enriched.append({
                "article_id":  aid,
                "name":        pg.get("prod_name")                  or item.get("name", ""),
                "colour":      pg.get("colour_group_name")          or item.get("colour", ""),
                "type":        pg.get("product_type_name")          or item.get("type", ""),
                "price":       f"£{avg_price}" if avg_price else    item.get("price", ""),
                "description": (pg.get("detail_desc") or "")[:120],
                "pattern":     pg.get("graphical_appearance_name")  or item.get("pattern", ""),
            })
        return enriched
    except Exception as _enr_err:
        print(f"[CHAT] item enrichment warning (non-fatal): {_enr_err}")
        return items


def _to_storage_item(item: dict) -> dict:
    """
    Convert a remapped item (name/colour/type/price-as-string) to the field
    names that ItemInContext requires (prod_name/colour_group_name/
    product_type_name/price-as-float-or-None).
    Used when calling store_response() for M2/M1 items.
    """
    import re as _re
    price_raw = item.get("price") or ""
    price_float = None
    if price_raw:
        m = _re.search(r"[\d.]+", str(price_raw))
        if m:
            try:
                price_float = float(m.group())
            except ValueError:
                pass
    return {
        "article_id":               str(item.get("article_id", "")),
        "prod_name":                item.get("name") or item.get("prod_name", ""),
        "product_type_name":        item.get("type") or item.get("product_type_name", ""),
        "colour_group_name":        item.get("colour") or item.get("colour_group_name", ""),
        "graphical_appearance_name":item.get("pattern") or item.get("graphical_appearance_name"),
        "detail_desc":              item.get("description") or item.get("detail_desc"),
        "price":                    price_float,
    }


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id:            str
    customer_id:        str
    message:            str
    session_id:         Optional[str] = None   # None = start new session
    force_new_session:  bool = False            # True = ignore existing session, start fresh
    selected_model:     str  = "m3"            # "m1", "m2", or "m3" — locked per session


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

        # ── Pre-read model lock so process_turn uses the right CSE ──────────
        # For existing sessions we look up the lock BEFORE process_turn so the
        # correct member CSE (M1/M2/M3) is selected.  For new sessions the lock
        # doesn't exist yet — fall back to the request value.
        _early_model = req.selected_model or "m3"
        _lookup_sid  = (None if req.force_new_session else req.session_id) or ""
        if _lookup_sid:
            try:
                from memory.db.mongo import get_db as _get_db_early
                _db_early = _get_db_early()
                _early_lock = await _db_early.session_model_locks.find_one(
                    {"session_id": _lookup_sid}, {"selected_model": 1}
                )
                if _early_lock and _early_lock.get("selected_model"):
                    _early_model = _early_lock["selected_model"]
                    print(f"[CHAT] early model lock read → {_early_model}")
            except Exception as _em_err:
                print(f"[CHAT] early lock read warning (non-fatal): {_em_err}")

        print(f"[CHAT] ─── Step 1: calling memory.process_turn (member_model={_early_model})...")
        # Step 1: Memory pipeline
        pipeline_output = await memory.process_turn(
            user_id=req.user_id,
            message=req.message,
            session_id=None if req.force_new_session else req.session_id,
            customer_id=req.customer_id,
            member_model=_early_model,
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
            print(f"[CHAT] CSE: tier={_cse.get('tier','?')} override={_cse.get('override','?')} "
                  f"full_sub={_cse.get('full_subtype')} partial_sub={_cse.get('partial_subtype')}")
        # ── Determine effective model (session-level lock) ─────────────────
        # Uses a dedicated session_model_locks collection — never writes to
        # sessions, users, recommendations or any other existing collection.
        #
        # First message of a new session → stores {session_id, selected_model}
        # in session_model_locks (upsert, so safe to call every turn).
        # Subsequent messages → reads the stored model and uses it, ignoring
        # whatever selected_model the request carries.  This enforces the
        # "one model per session, unchangeable" contract.
        import asyncio as _asyncio
        _session_id_for_model = pipeline_output.get("session_id", "")
        effective_model = req.selected_model or "m3"
        try:
            from memory.db.mongo import get_db as _get_db_model
            from datetime import datetime as _dt, timezone as _tz
            _db_model = _get_db_model()
            _lock_doc = await _db_model.session_model_locks.find_one(
                {"session_id": _session_id_for_model},
                {"selected_model": 1},
            )
            if _lock_doc and _lock_doc.get("selected_model"):
                # Existing session — honour the locked model, ignore request value
                effective_model = _lock_doc["selected_model"]
                print(f"[CHAT] model lock: existing lock found → model={effective_model}")
            else:
                # New session — persist the model selection in its own collection
                await _db_model.session_model_locks.update_one(
                    {"session_id": _session_id_for_model},
                    {"$setOnInsert": {
                        "session_id":     _session_id_for_model,
                        "user_id":        pipeline_output.get("user_id", ""),
                        "selected_model": effective_model,
                        "locked_at":      _dt.now(_tz.utc).isoformat(),
                    }},
                    upsert=True,
                )
                print(f"[CHAT] model lock: new lock stored → model={effective_model}")
        except Exception as _me:
            print(f"[CHAT] model lock warning (non-fatal): {_me}")

        # ── Route pipeline_output to the selected member module only ──────
        # M3: processed locally by rag.process() below — no external call.
        # M2/M1: call synchronously and use their response exclusively.
        #         If the module is unreachable, return an error to the user
        #         immediately — NO fallback to M3.
        print(f"[CHAT] routing to selected_model={effective_model}")
        _member_rag_result = None
        if effective_model == "m2":
            print(f"[CHAT] M2 selected — calling M2 sync (timeout=30s)")
            _m2_resp = await _call_m2_sync(pipeline_output)
            if _m2_resp is None:
                print(f"[CHAT] M2 unavailable — returning error to user")
                return _member_unavailable_response("M2 · Multimodal RAG", pipeline_output)
            _member_rag_result = _normalize_member_response(_m2_resp)
        elif effective_model == "m1":
            print(f"[CHAT] M1 selected — calling M1 sync (timeout=30s)")
            _m1_resp = await _call_m1_sync(pipeline_output)
            if _m1_resp is None:
                print(f"[CHAT] M1 unavailable — returning error to user")
                return _member_unavailable_response("M1 · Graph RAG", pipeline_output)
            _member_rag_result = _normalize_member_response(_m1_resp)
        else:
            print(f"[CHAT] M3 selected — processing locally")

        # ── Store classifier_input in the turn document ───────────────────
        # Stores the [SEP]-joined DistilBERT input so rl_routes.py can
        # retrieve it when user clicks 👍 or 👎.
        # We store on the USER turn using pipeline_output turn_id.
        # We also patch the recommendation document with the user_turn_id
        # so rl_routes.py can find it via recommendation_id → user_turn_id.
        _user_turn_id     = pipeline_output.get("turn_id", "")
        _classifier_input = pipeline_output.get("classifier_input", "")
        if _user_turn_id and _classifier_input:
            try:
                from memory.db.mongo import get_db as _get_db_ci
                _db_ci = _get_db_ci()
                # Store classifier_input on the user turn
                await _db_ci.turns.update_one(
                    {"turn_id": _user_turn_id},
                    {"$set": {"classifier_input": _classifier_input}}
                )
                # Also update embedded turns in session document
                await _db_ci.sessions.update_one(
                    {"turns.turn_id": _user_turn_id},
                    {"$set": {"turns.$.classifier_input": _classifier_input}}
                )
            except Exception as _ci_err:
                print(f"[CHAT] classifier_input store warning (non-fatal): {_ci_err}")

        print(f"[CHAT] ─── Step 2: calling rag.process...")
        # Step 2: Text RAG — M3 only. M1/M2 already produced their result above.
        if _member_rag_result is not None:
            rag_result = _member_rag_result
            # M2/M1 often return only article_ids — enrich with full catalog details
            rag_result["items_recommended"] = await _enrich_member_items(
                rag_result.get("items_recommended", [])
            )
            print(f"[CHAT] ─── Using {effective_model.upper()} response (local RAG skipped)")
            # Store the M2/M1 response so CSE can populate currently_discussing
            # on follow-up turns (ATTRIBUTE_QUESTION, EXPLANATION_WHY, etc.)
            _member_items = rag_result.get("items_recommended", [])
            if _member_items:
                try:
                    # ItemInContext requires prod_name/colour_group_name/product_type_name.
                    # Our remapped items use name/colour/type — convert before storing.
                    _storage_items = [_to_storage_item(i) for i in _member_items]
                    await memory.store_response(
                        session_id=pipeline_output.get("session_id", ""),
                        user_id=pipeline_output.get("user_id", ""),
                        bot_response=rag_result.get("response_text", ""),
                        recommended_items=_storage_items,
                        trigger_label=pipeline_output.get("label", "UNKNOWN"),
                        retrieval_strategy=pipeline_output.get("retrieval_strategy", "UNKNOWN"),
                        collection_prefix=effective_model,
                    )
                    print(f"[CHAT] store_response OK for {effective_model.upper()} — {len(_member_items)} item(s) saved")
                except Exception as _sr_err:
                    print(f"[CHAT] store_response warning (non-fatal): {_sr_err}")
        else:
            rag_result = await rag.process(
                pipeline_output=pipeline_output,
                memory_pipeline=memory,
                store_response=True,
                collection_prefix=effective_model,
            )

        print(f"[CHAT] ─── RAG done")
        print(f"[CHAT] rag_result keys: {list(rag_result.keys())}")
        print(f"[CHAT] response_text: '{rag_result.get('response_text','')[:100]}'")
        print(f"[CHAT] action={rag_result.get('action')} items={len(rag_result.get('items_recommended',[]))} hall={rag_result.get('hallucination_flag')} contra={rag_result.get('contradiction_found')}")
        for _itm in rag_result.get("items_recommended",[]):
            print(f"[CHAT] ITEM: {_itm.get('article_id','?')} | {str(_itm.get('name','?'))[:30]} | {_itm.get('colour','?')} | {_itm.get('price','?')}")

        # ── Retrieve recommendation_id and patch user_turn_id ─────────────
        # store_response() saves recommendation linked to the BOT turn.
        # We also patch user_turn_id onto the recommendation so that
        # rl_routes.py can find the user turn (with classifier_input)
        # when the user clicks 👍 or 👎.
        _rec_id = None
        if rag_result.get("action") == "catalog_search" and rag_result.get("items_recommended"):
            try:
                from memory.db.mongo import get_db as _get_db_inner, get_collection_name as _gcn
                _db_inner  = _get_db_inner()
                _recs_coll = _gcn("recommendations", effective_model)
                _latest_rec = await _db_inner[_recs_coll].find_one(
                    {
                        "session_id": pipeline_output.get("session_id", ""),
                        "user_id":    pipeline_output.get("user_id", ""),
                    },
                    sort=[("created_at", -1)],
                )
                if _latest_rec:
                    _rec_id = _latest_rec.get("recommendation_id")
                    _user_turn_id_for_rec = pipeline_output.get("turn_id", "")
                    if _user_turn_id_for_rec and _rec_id:
                        await _db_inner[_recs_coll].update_one(
                            {"recommendation_id": _rec_id},
                            {"$set": {"user_turn_id": _user_turn_id_for_rec}}
                        )
            except Exception as _rec_err:
                print(f"[CHAT] recommendation_id lookup warning (non-fatal): {_rec_err}")
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

        # ── Collect implicit RL signal (next-turn behaviour) ──────────────
        # This fires silently — never affects the response to the user.
        # Looks at what label the user just sent vs what the PREVIOUS turn predicted.
        # Example: if last turn was REFINEMENT and this turn is SELECTION_REFERENCE
        # → the refinement worked → reward +0.7 stored in rl_experiences collection.
        try:
            from memory.db.mongo import get_db as _get_db
            _db  = _get_db()
            _rl  = get_rl_collector()

            _session_id = pipeline_output.get("session_id", "")
            _user_id    = pipeline_output.get("user_id", "")
            _cur_label  = pipeline_output.get("label", "")

            # Turns are stored embedded in the session document (not in a
            # separate turns collection) — load from db.sessions directly.
            _sess_doc  = await _db.sessions.find_one({"session_id": _session_id})
            _user_turns = []
            if _sess_doc:
                _user_turns = [
                    t for t in _sess_doc.get("turns", [])
                    if t.get("role") == "user"
                ]
                # list is chronological (oldest first); we need last 2

            if len(_user_turns) >= 2:
                _prev_turn = _user_turns[-2]   # second-to-last user turn
                _prev_cls  = _prev_turn.get("classification") or {}
                _prev_lbl  = _prev_cls.get("label", "")
                if _prev_lbl:
                    import asyncio as _aio
                    _aio.ensure_future(_rl.collect_implicit_signal(
                        session_id=      _session_id,
                        user_id=         _user_id,
                        prev_turn_id=    _prev_turn.get("turn_id", ""),
                        prev_label=      _prev_lbl,
                        prev_input_text= _prev_turn.get("classifier_input", "") or _prev_turn.get("content", ""),
                        prev_label_id=   LABEL_NAME_TO_ID.get(_prev_lbl, 0),
                        prev_strategy=   _prev_cls.get("retrieval_strategy", "FULL"),
                        prev_confidence= _prev_cls.get("confidence", 0.0),
                        next_label=      _cur_label,
                        db=              _db,
                    ))
                    print(f"[CHAT] RL implicit: {_prev_lbl}->{_cur_label} queued for session {_session_id[:12]}")
        except Exception as _rl_err:
            # RL signal collection NEVER breaks the main response
            print(f"[CHAT] RL implicit signal warning (non-fatal): {_rl_err}")

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
            # ── NEW: recommendation_id for RL explicit feedback ────────────
            # Frontend uses this to submit 👍/👎 via POST /api/rl/feedback
            "recommendation_id":   _rec_id,
            "turn_id":             pipeline_output.get("turn_id", ""),
        }

    except Exception as e:
        print(f"[Chat API] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
