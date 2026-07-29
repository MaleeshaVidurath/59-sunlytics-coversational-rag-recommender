from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
from response_generator import generate_llm_response
import uvicorn

# Import your existing graph logic
from build_graph import construct_knowledge_graph
from graph_search import handle_retrieval_request

# ── NEW: Import the Async Semantic Router ──
from m1_router import heal_and_route_input

app = FastAPI(title="M1 Graph RAG API")

# 1. Load the Knowledge Graph into server memory on startup
print("Initializing Knowledge Graph for API...")
kg = construct_knowledge_graph()

class ProcessRequest(BaseModel):
    retrieval_input: Optional[Dict[str, Any]] = None
    memory_context: Optional[Dict[str, Any]] = {}

@app.post("/api/process")
async def process_request(req: ProcessRequest):
    if not req.retrieval_input:
        action = "NO_RETRIEVAL"
        user_message = req.memory_context.get("message", "Hello!") if req.memory_context else "Hello!"
        response_text, hall_flag, hall_score = generate_llm_response(action, user_message, [])
        return {
            "success": True, 
            "action": action, 
            "items": [], 
            "response_text": response_text,
            "hallucination_flag": hall_flag,
            "hallucination_score": hall_score
        }
    
    try:
        # ── AGENTIC RECOVERY HOOK (Now Async!) ──
        # M1 fixes the intent and fetches missing DB context automatically
        fixed_retrieval_input = await heal_and_route_input(req.retrieval_input, req.memory_context)
        
        action = fixed_retrieval_input.get("action")
        user_message = fixed_retrieval_input.get("user_message", "")
        
        # 1. Run the Graph Search
        result = handle_retrieval_request(kg, fixed_retrieval_input)
        raw_data = result.get("data", [])
        counterfactuals = result.get("counterfactuals", [])
        
        # 2. Generate Response
        response_text, hall_flag, hall_score = generate_llm_response(
            action, user_message, raw_data, counterfactuals=counterfactuals
        )
        
        return {
            "success": True,
            "action": action,
            "items": raw_data,
            "response_text": response_text,
            "hallucination_flag": hall_flag,
            "hallucination_score": hall_score
        }
    except Exception as e:
        print(f"Error processing graph request: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)