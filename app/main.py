"""
Main FastAPI application for Module 2: Multimodal RAG.

This module exposes the endpoints for the conversational recommender system.
It acts as the API Gateway, handling both legacy text-based chat requests
and structured pipeline requests from the m3 Memory Pipeline.
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Local application imports
from m2_multimodal_rag.retrieval import m2_retriever
from m2_multimodal_rag.regeneration_loop import generator_loop
from m2_multimodal_rag.m2_action_router import m2_router
from shared.data_loader import data_loader

# =====================================================================
# Application Initialization & Configuration
# =====================================================================

app = FastAPI(
    title="M2 Conversational Recommender API",
    description="Multimodal RAG API for fashion recommendations.",
    version="1.0.0"
)

# Add CORS middleware to allow the React frontend (or other clients) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Note: In production, restrict this to your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Pydantic Request Models
# =====================================================================

class ChatRequest(BaseModel):
    """Model for legacy plain-text chat queries."""
    query: str


class PipelineRequest(BaseModel):
    """
    Model for the structured input from the m3 Memory Pipeline.
    This is the primary interface for the adaptive trigger system.
    """
    retrieval_input: Optional[dict] = None   # None indicates FEEDBACK or CHITCHAT actions
    memory_context: Optional[dict] = {}      # Contains user prefs, dialogue state, and feedback


# =====================================================================
# Core API Endpoints
# =====================================================================

@app.post("/api/process")
async def process_endpoint(request: PipelineRequest) -> dict:
    """
    Primary endpoint for the m3 pipeline integration.
    
    Accepts structured retrieval_input and delegates it to the M2 Action Router.
    The router determines the appropriate handler (e.g., catalog_search, item_compare)
    and executes the multimodal RAG logic.
    """
    action_type = request.retrieval_input.get('action') if request.retrieval_input else 'None (FEEDBACK/CHITCHAT)'
    print(f"\n[API] Received pipeline request. Action: {action_type}")
    
    try:
        # Route the request through the central M2 dispatcher
        result = m2_router.process_retrieval_input(
            retrieval_input=request.retrieval_input,
            memory_context=request.memory_context
        )
        
        # Post-process: Attach local image URLs for any recommended items
        if result.get("items"):
            for item in result["items"]:
                article_id = item.get("article_id", "")
                if article_id:
                    item["image_url"] = f"/api/images/{article_id}"
        
        return result

    except Exception as e:
        print(f"[API ERROR] Pipeline processing failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during pipeline processing.")


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> dict:
    """
    Legacy endpoint for backward compatibility.
    
    Accepts a raw text string, uses the VLM to clean it, runs a CLIP/FAISS
    search, and generates a visually verified explanation.
    """
    user_query = request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    print(f"\n[API] Received legacy chat query: {user_query}")
    
    try:
        # Step 1: Retrieve the best match using the older M2 Retriever
        recommendations = m2_retriever.get_recommendations(text_query=user_query, top_k=1)
        
        # Handle cases where the VLM flagged the query as irrelevant
        if recommendations == "IRRELEVANT_QUERY":
            return {
                "reply": "I am a fashion design recommender! Please ask me about clothing, styles, or colors.",
                "article_id": None,
                "image_url": None
            }
        
        # Handle cases where no items were found
        if not recommendations:
            return {
                "reply": "I'm sorry, I couldn't find any items matching your description.",
                "article_id": None,
                "image_url": None
            }
            
        top_article_id, _ = recommendations[0]
        
        # Step 2: Generate and Verify the explanation using the M2 Guard
        final_response = generator_loop.generate_faithful_explanation(
            article_id=top_article_id,
            force_hallucination_test=False
        )
        
        return {
            "reply": final_response,
            "article_id": top_article_id,
            "image_url": f"/api/images/{top_article_id}"
        }

    except Exception as e:
        print(f"[API ERROR] Legacy chat processing failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during chat processing.")


# =====================================================================
# Static Asset Endpoints
# =====================================================================

@app.get("/api/images/{article_id}")
async def get_image(article_id: str) -> FileResponse:
    """
    Serves the product image for a given article ID.
    
    Dynamically downloads the image from Kaggle if it isn't already cached locally.
    """
    try:
        image_path = data_loader.get_image(article_id)
        
        if image_path and image_path.exists():
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail=f"Image for article {article_id} not found.")
            
    except HTTPException:
        # Re-raise HTTPExceptions so they aren't masked as 500s by the generic block below
        raise
    except Exception as e:
        print(f"[API ERROR] Failed to serve image for {article_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching image.")
