# m3_implementation/api/routers/chat.py
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.dependencies import get_memory_pipeline, get_rag_pipeline

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id:     str
    customer_id: str
    message:     str
    session_id:  Optional[str] = None   # None = start new session


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

    try:
        # Step 1: Memory pipeline
        pipeline_output = await memory.process_turn(
            user_id=req.user_id,
            message=req.message,
            session_id=req.session_id,
            customer_id=req.customer_id,
        )

        # Step 2: Text RAG pipeline (includes hallucination + contradiction)
        rag_result = await rag.process(
            pipeline_output=pipeline_output,
            memory_pipeline=memory,
            store_response=True,
        )

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
        }

    except Exception as e:
        print(f"[Chat API] Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
