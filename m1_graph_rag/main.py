from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
from response_generator import generate_llm_response
import uvicorn

# Import your existing graph logic
from build_graph import construct_knowledge_graph
from graph_search import handle_retrieval_request

app = FastAPI(title="M1 Graph RAG API")

# 1. Load the Knowledge Graph into server memory on startup
print("Initializing Knowledge Graph for API...")
kg = construct_knowledge_graph()

# 2. Define the incoming JSON structure we expect from Module 3
class ProcessRequest(BaseModel):
    retrieval_input: Optional[Dict[str, Any]] = None
    memory_context: Optional[Dict[str, Any]] = {}

# 3. Expose the endpoint Module 3 is looking for
@app.post("/api/process")
async def process_request(req: ProcessRequest):
    # Handle CHITCHAT or FEEDBACK (where retrieval_input is null)
    if not req.retrieval_input:
        # For pure chitchat, we bypass the graph and just let the LLM talk
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
        action = req.retrieval_input.get("action")
        user_message = req.retrieval_input.get("user_message", "")
        
        # 1. Run the Graph Search (Math, Logic & Counterfactual Pruning Logs)
        result = handle_retrieval_request(kg, req.retrieval_input)
        raw_data = result.get("data", [])
        counterfactuals = result.get("counterfactuals", [])
        
        # 2. Generate the LLM Response with DeBERTa NLI Guard & Counterfactual Translation
        response_text, hall_flag, hall_score = generate_llm_response(
            action, user_message, raw_data, counterfactuals=counterfactuals
        )
        
        # 3. Package the response in the exact format Module 3 expects
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
    # Module 3 expects us on port 8002 by default
    uvicorn.run(app, host="127.0.0.1", port=8002)