from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from m2_multimodal_rag.retrieval import m2_retriever
from m2_multimodal_rag.regeneration_loop import generator_loop
from shared.data_loader import data_loader

app = FastAPI(title="M2 Conversational Recommender API")

# Add CORS middleware to allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required")

    print(f"\n[API] Received chat query: {user_query}")
    
    try:
        # 1. Retrieve the best match using M2 Retriever
        recommendations = m2_retriever.get_recommendations(text_query=user_query, top_k=1)
        
        if recommendations == "IRRELEVANT_QUERY":
            return {
                "reply": "I am a fashion design recommender! Please ask me about clothing, styles, or colors.",
                "article_id": None,
                "image_url": None
            }
        
        if not recommendations:
            return {
                "reply": "I'm sorry, I couldn't find any items matching your description.",
                "article_id": None,
                "image_url": None
            }
            
        top_article_id, top_score = recommendations[0]
        
        # 2. Generate and Verify the explanation using M2 Guard
        final_response = generator_loop.generate_faithful_explanation(
            article_id=top_article_id,
            force_hallucination_test=False
        )
        
        # Construct the local image URL so the frontend can fetch it
        image_url = f"/api/images/{top_article_id}"
        
        return {
            "reply": final_response,
            "article_id": top_article_id,
            "image_url": image_url
        }

    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/images/{article_id}")
async def get_image(article_id: str):
    """Serves the downloaded product image."""
    try:
        # Use data_loader to fetch the local path (downloads it from Kaggle if needed)
        image_path = data_loader.get_image(article_id)
        if image_path and image_path.exists():
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
