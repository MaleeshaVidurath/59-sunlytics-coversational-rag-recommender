import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from m2_multimodal_rag.m2_action_router import m2_router
from shared.data_loader import data_loader
from .schemas import PipelineRequest, SimpleSearchRequest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# API Router & Endpoints
# =====================================================================

api_router = APIRouter(prefix="/api")

def _attach_image_urls(result_data: Dict[str, Any]) -> None:
    """Helper function to attach local image URLs to recommended items."""
    if not result_data.get("items"):
        return
        
    for item in result_data["items"]:
        article_id = item.get("article_id", "")
        if article_id:
            item["image_url"] = f"/api/images/{article_id}"


@api_router.post("/process")
async def process_endpoint(request: PipelineRequest) -> dict:
    # Convert the validated Pydantic model back to a standard Python dictionary for the M2 internal router
    retrieval_dict = request.retrieval_input.dict() if request.retrieval_input else None
    
    action_type = retrieval_dict.get('action') if retrieval_dict else 'None (FEEDBACK/CHITCHAT)'
    logger.info(f"Received pipeline request. Action: {action_type}")
    print(f"\n[M3->M2 RAW REQUEST]\n{json.dumps(request.dict(), indent=2, default=str)}\n")
    
    try:
        # Route the request through the central M2 dispatcher
        result = m2_router.process_retrieval_input(
            retrieval_input=retrieval_dict,
            memory_context=request.memory_context
        )
        
        # Post-process: Attach local image URLs
        _attach_image_urls(result)
        
        return result

    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during pipeline processing.")

@api_router.get("/images/{article_id}")
async def get_image(article_id: str) -> FileResponse:
    try:
        image_path = data_loader.get_image(article_id)
        
        if image_path and image_path.exists():
            return FileResponse(image_path)
        else:
            logger.warning(f"Image not found for article: {article_id}")
            raise HTTPException(status_code=404, detail=f"Image for article {article_id} not found.")
            
    except HTTPException:
        # Re-raise HTTPExceptions so they aren't masked as 500s
        raise
    except Exception as e:
        logger.error(f"Failed to serve image for {article_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while fetching image.")


# =====================================================================
# Application Initialization
# =====================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="M2 Conversational Recommender API",
        description="Multimodal RAG API for fashion recommendations.",
        version="1.0.0"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app

app = create_app()


