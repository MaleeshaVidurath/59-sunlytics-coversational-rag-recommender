# m3_implementation/api/routers/chat.py
import asyncio
import json as _json
import re as _re
import time
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_memory_pipeline, get_rag_pipeline
from memory.core.rl_signal_collector import get_rl_collector, LABEL_NAME_TO_ID
from api.latency_logger import determine_tier, compute_response_status, log_turn


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

    _mem_ctx = dict(pipeline_output.get("memory_context") or {})
    _mem_ctx["session_id"] = pipeline_output.get("session_id")
    body = _make_json_safe({
        "retrieval_input": pipeline_output.get("retrieval_input"),
        "memory_context":  _mem_ctx,
    })
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{url}/api/process", json=body)
    except Exception as e:
        # Non-fatal — friend module may not be running yet
        pass


async def _call_m2_sync(pipeline_output: dict, timeout: float = 300.0):
    """Calls M2 synchronously.
    Returns the response dict on both success=True and success=False (M2 is reachable).
    Returns None only when M2 cannot be reached at all (timeout, connection error).
    """
    if not _M2_URL:
        return None
    # session_id lets M2 resolve follow-ups from its per-session candidate
    # cache instead of a full re-retrieval (M2 schema takes memory_context
    # as a free-form dict, so no schema change is needed).
    _mem_ctx = dict(pipeline_output.get("memory_context") or {})
    _mem_ctx["session_id"] = pipeline_output.get("session_id")
    body = _make_json_safe({
        "retrieval_input": pipeline_output.get("retrieval_input"),
        "memory_context":  _mem_ctx,
    })
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{_M2_URL}/api/process", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data  # always return — None only means unreachable
    except Exception as e:
        return None


async def _call_m1_sync(pipeline_output: dict, customer_id: str, timeout: float = 300.0):
    """Calls M1 synchronously.
    Returns the response dict on both success=True and success=False (M1 is reachable).
    Returns None only when M1 cannot be reached at all (timeout, connection error).
    """
    if not _M1_URL:
        return None
        
    ret_input = pipeline_output.get("retrieval_input") or {}
    ret_input["customer_id"] = customer_id
    
    body = _make_json_safe({
        "retrieval_input": ret_input,
        "memory_context":  pipeline_output.get("memory_context") or {},
    })

    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{_M1_URL}/api/process", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data  # always return — None only means unreachable
    except Exception as e:
        return None


# ── M2-only: per-item justification ("Why this for you") ─────────────────────
# Everything below runs solely on the M2 branch, gated by source == "m2".
# M1 and M3 keep the exact path they had before.

# Longest "why" line kept on a card. Past this a sentence wraps into a
# paragraph and stops reading as a bullet.
_WHY_LINE_MAX = 180

# Most justification lines shown per card — the limit M3's ranker also uses.
_WHY_LINES_MAX = 3


def _plural(word: str) -> str:
    """
    Plural of a product-type noun, for the fallback justification line.

    The catalog holds types like "Dress", "Shorts" and "Body" — a bare "+ s"
    produces "dresss" and "bodys", which reads as a bug on the card.
    """
    w = word.strip()
    if not w:
        return w
    if w.endswith(("ss", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("s"):
        return w          # "Trousers", "Shorts" — already plural
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def _m2_why(item: dict) -> list[str]:
    """
    User-facing justification lines for one M2 item.

    M2 emits a single `explanation` paragraph per item, already checked by its
    3-layer hallucination guard, so it is safe to render verbatim. It is split
    into sentences because the card draws one bullet per line.

    Never returns an empty list: an item whose explanation came back blank
    still gets an honest line built from its own attributes, so no card ever
    renders with an empty "Why this for you" block.
    """
    raw   = (item.get("explanation") or "").strip()
    lines = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", raw) if len(s.strip()) > 1]
    lines = [
        l if len(l) <= _WHY_LINE_MAX else l[:_WHY_LINE_MAX - 1].rstrip() + "…"
        for l in lines
    ]
    if lines:
        return lines[:_WHY_LINES_MAX]

    # Fallback — states the truth (it matched the request) without claiming a
    # personalised signal M2 never computed.
    colour = (item.get("colour") or item.get("colour_group_name") or "").strip()
    ptype  = _plural((item.get("type") or item.get("product_type_name") or "item").strip().lower())
    descriptor = f"{colour.lower()} {ptype}" if colour else ptype
    return [f"One of the closest matches to what you asked for in {descriptor}"]


def _attach_m2_match_percent(items: list[dict]) -> None:
    """
    Adds a 0-100 display figure to each M2 item, in place.

    M2's `final_score` sums FAISS relevance, preference boosts, CF history and
    KB signals minus penalties — unbounded, and negative once penalties
    dominate, so it can never go on a card raw. It is scaled against the best
    score in the same result set, the way PersonalizedRanker does it for M3.
    Left as None when no score is positive: nothing meaningful ranked, so no
    badge should be shown at all.
    """
    numeric = [i.get("score") for i in items if isinstance(i.get("score"), (int, float))]
    best    = max(numeric, default=0.0)

    for item in items:
        score = item.get("score")
        if best > 0 and isinstance(score, (int, float)) and score > 0:
            item["match_percent"] = int(round(min(score / best, 1.0) * 100))
        else:
            item["match_percent"] = None


def _remap_member_item(item: dict, source: str = "") -> dict:
    """
    Normalise a single item from M1/M2 response to the field names
    chat.py's item extraction loop expects.
    M2 uses prod_name/colour_group_name/product_type_name/detail_desc;
    M1 may differ — try both names so either works.

    `source` is "m2", "m1" or "" — only "m2" adds the justification fields.
    """
    price_raw = item.get("price") or item.get("avg_price")
    if price_raw and not isinstance(price_raw, str):
        price_str = f"£{price_raw}"
    else:
        price_str = price_raw or ""
    remapped = {
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
    if source == "m2":
        # Every M2 recommendation carries a justification — see _m2_why.
        remapped["why"] = _m2_why(item)
    return remapped


def _normalize_member_response(data: dict, source: str = "") -> dict:
    """Convert M1/M2 API response to the same structure as rag.process() output.
    When success=False, passes M2/M1's own error message as response_text.

    `source` is "m2", "m1" or "" — only "m2" gets justification fields added.
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
            "revisions":           [],
            "product_ids":         [],
            "product_names":       [],
            "action":              data.get("action", ""),
            "items_recommended":   [],
        }
    remapped_items = [_remap_member_item(i, source) for i in data.get("items", [])]
    if source == "m2":
        # Scaling needs the whole set, so it runs after every item is remapped.
        _attach_m2_match_percent(remapped_items)
    return {
        "response_text":       data.get("response_text", data.get("message", "")),
        "hallucination_flag":  data.get("hallucination_flag", False),
        "hallucination_score": data.get("hallucination_score", 0.0),
        "flagged_sentences":   data.get("flagged_sentences", []),
        "attempt_count":       data.get("attempt_count", 1),
        "contradiction_found": data.get("contradiction_found", False),
        "contradiction_count": data.get("contradiction_count", 0),
        "contradictions":      data.get("contradictions", []),
        "revisions":           data.get("revisions", []),
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
        "revisions":           [],
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

        enriched = []
        for item in items:
            aid = str(item.get("article_id", ""))
            pg = pg_map.get(aid, {})
            avg_price = pg.get("avg_price")
            enriched_item = {
                "article_id":  aid,
                "name":        pg.get("prod_name")                  or item.get("name", ""),
                "colour":      pg.get("colour_group_name")          or item.get("colour", ""),
                "type":        pg.get("product_type_name")          or item.get("type", ""),
                "price":       f"£{avg_price}" if avg_price else    item.get("price", ""),
                "description": (pg.get("detail_desc") or "")[:120],
                "pattern":     pg.get("graphical_appearance_name")  or item.get("pattern", ""),
            }
            # Catalog enrichment must not discard a justification the module
            # already produced. Only M2 items carry these keys, so this is a
            # no-op for M1.
            for _carry in ("why", "match_percent"):
                if _carry in item:
                    enriched_item[_carry] = item[_carry]
            enriched.append(enriched_item)
        return enriched
    except Exception as _enr_err:
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
        # Persist the justification so reopening a past chat still shows it.
        # Only M2 items carry these; for M1 they stay None, exactly as before.
        "why":                      item.get("why") or None,
        "match_percent":            item.get("match_percent"),
    }


async def _fetch_last_recommended_items(session_id: str, effective_model: str) -> list[str]:
    """
    Fetches the last two recommended article IDs for a given session from MongoDB.
    Respects the active model (m1, m2, m3) so it pulls from the correct collection.
    """
    if not session_id:
        return []
        
    try:
        from memory.db.mongo import get_db, get_collection_name
        db = get_db()
        recs_coll = get_collection_name("recommendations", effective_model)
        
        latest_rec = await db[recs_coll].find_one(
            {"session_id": session_id},
            sort=[("created_at", -1)]
        )
        
        if latest_rec and "items_recommended" in latest_rec:
            items = latest_rec["items_recommended"]
            return [str(item["article_id"]) for item in items[:2] if item.get("article_id")]
    except Exception as e:
        pass
        
    return []


from ..security.dependencies import get_current_account, verify_csrf
from ..security.models import AccountDocument

# Authenticate before the CSRF check, so an expired session answers 401 (which
# the client can act on by refreshing) rather than 403.
router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(get_current_account), Depends(verify_csrf)],
)


class ChatRequest(BaseModel):
    """
    A chat turn.

    `user_id` and `customer_id` used to be fields here. They are gone on
    purpose: both identify *whose* data the pipeline reads and writes, and
    accepting them from the request body let any caller act as any user. They
    are now derived from the caller's access token.
    """
    message:            str
    session_id:         Optional[str] = None   # None = start new session
    force_new_session:  bool = False            # True = ignore existing session, start fresh
    selected_model:     str  = "m3"            # "m1", "m2", or "m3" — locked per session


@router.post("")
async def chat(req: ChatRequest,
               account: AccountDocument = Depends(get_current_account)):
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
    # Identity comes from the verified access token, never from the payload.
    # A self-registered account has no H&M persona, so customer_id is empty and
    # the pipeline runs cold-start: semantic ranking, no purchase-history hints.
    user_id     = account.user_id or account.account_id
    customer_id = account.linked_customer_id or ""

    memory = get_memory_pipeline()
    rag    = get_rag_pipeline()

    if not memory or not rag:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Server starting up."
        )

    try:
        # If force_new_session, clear Redis active session pointer first
        if req.force_new_session:
            try:
                from memory.db.redis_client import get_redis
                redis = await get_redis()
                await redis.delete(f"user:{user_id}:active_session")
            except Exception as e:
                pass

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
            except Exception as _em_err:
                pass

        t_start = time.perf_counter()
        # Step 1: Memory pipeline
        pipeline_output = await memory.process_turn(
            user_id=user_id,
            message=req.message,
            session_id=None if req.force_new_session else req.session_id,
            customer_id=customer_id,
            member_model=_early_model,
        )

        t_memory_done = time.perf_counter()
        _ri = pipeline_output.get("retrieval_input") or {}
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
        except Exception as _me:
            pass

        # ── NEW FALLBACK LOGIC FOR COMPARISON ─────────────────────────────
        # If the user asked to compare but the memory context cache missed the IDs,
        # fetch the IDs from the last recommendation made in this session.
        if _ri and _ri.get("action") == "item_compare":
            _pl = _ri.get("payload", {})
            if not _pl.get("article_id_a") or not _pl.get("article_id_b"):
                _last_items = await _fetch_last_recommended_items(_session_id_for_model, effective_model)
                if len(_last_items) >= 2:
                    _pl["article_id_a"] = _last_items[0]
                    _pl["article_id_b"] = _last_items[1]
                    _ri["payload"] = _pl
                    pipeline_output["retrieval_input"] = _ri
        # ──────────────────────────────────────────────────────────────────

        # ── Route pipeline_output to the selected member module only ──────
        # M3: processed locally by rag.process() below — no external call.
        # M2/M1: call synchronously and use their response exclusively.
        #        If the module is unreachable, return an error to the user
        #        immediately — NO fallback to M3.
        _member_rag_result = None
        if effective_model == "m2":
            _m2_resp = await _call_m2_sync(pipeline_output)
            if _m2_resp is None:
                return _member_unavailable_response("M2 · Multimodal RAG", pipeline_output)
            _member_rag_result = _normalize_member_response(_m2_resp, "m2")
        elif effective_model == "m1":
            _m1_resp = await _call_m1_sync(pipeline_output, customer_id=customer_id)
            if _m1_resp is None:
                return _member_unavailable_response("M1 · Graph RAG", pipeline_output)
            _member_rag_result = _normalize_member_response(_m1_resp, "m1")

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
                pass

        # Step 2: Text RAG — M3 only. M1/M2 already produced their result above.
        if _member_rag_result is not None:
            rag_result = _member_rag_result
            # M2/M1 often return only article_ids — enrich with full catalog details
            rag_result["items_recommended"] = await _enrich_member_items(
                rag_result.get("items_recommended", [])
            )
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
                except Exception as _sr_err:
                    pass
        else:
            rag_result = await rag.process(
                pipeline_output=pipeline_output,
                memory_pipeline=memory,
                store_response=True,
                collection_prefix=effective_model,
            )

        t_rag_done = time.perf_counter()
        _tier, _sub_tier = determine_tier(pipeline_output)
        log_turn(
            session_id      = pipeline_output.get("session_id", ""),
            turn_id         = pipeline_output.get("turn_id", ""),
            label           = pipeline_output.get("label", ""),
            tier            = _tier,
            sub_tier        = _sub_tier,
            user_message    = req.message,
            memory_ms       = (t_memory_done - t_start)      * 1000,
            rag_ms          = (t_rag_done    - t_memory_done) * 1000,
            total_ms        = (t_rag_done    - t_start)       * 1000,
            response_status = compute_response_status(rag_result),
        )

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
                pass
        # Extract items for product cards
        items = []
        for item in rag_result.get("items_recommended", []):
            if item.get("article_id") or item.get("name"):
                # M3 items carry the product text as `material_description`;
                # M2 remaps it to `description`. Reading only the first name
                # dropped M2's text silently, so the M2 branch falls back to
                # its own key. M1 and M3 keep the original lookup untouched.
                _desc = item.get("material_description") or ""
                if not _desc and effective_model == "m2":
                    _desc = item.get("description") or ""
                items.append({
                    "article_id":  str(item.get("article_id", "")),
                    "name":        item.get("name", ""),
                    "colour":      item.get("colour", ""),
                    "type":        item.get("type", ""),
                    "price":       item.get("price", ""),
                    "description": _desc[:120],
                    "pattern":     item.get("pattern", ""),
                    # Why this item was selected for THIS user — template strings
                    # built from real statistics by PersonalizedRanker, never
                    # LLM text, so they are safe to render verbatim.
                    "why":           item.get("why", []),
                    # 0-100 display figure, or None when nothing meaningful
                    # matched. Never send the raw score to the UI: it is
                    # unbounded and goes negative once penalties apply.
                    "match_percent": item.get("match_percent"),
                })


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
        except Exception as _rl_err:
            # RL signal collection NEVER breaks the main response
            pass

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
            # Catalogue values that changed since an earlier turn quoted them.
            # Sent on the live turn so the correction appears immediately; the
            # session-history endpoint replays the same notices on reload.
            "revisions":           rag_result.get("revisions", []),
            "cse":                 pipeline_output.get("cse", {}),
            # ── NEW: recommendation_id for RL explicit feedback ────────────
            # Frontend uses this to submit 👍/👎 via POST /api/rl/feedback
            "recommendation_id":   _rec_id,
            "turn_id":             pipeline_output.get("turn_id", ""),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))