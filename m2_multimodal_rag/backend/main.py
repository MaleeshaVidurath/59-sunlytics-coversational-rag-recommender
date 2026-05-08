"""
Main FastAPI application for Module 2: Multimodal RAG.

This module exposes the endpoints for the conversational recommender system.
It acts as the API Gateway for handling structured pipeline requests from the 
m3 Memory Pipeline and serving static image assets.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Local application imports
from m2_multimodal_rag.m2_action_router import m2_router
from m2_multimodal_rag.m2_handlers import handle_image_catalog_search
from shared.data_loader import data_loader
from .schemas import PipelineRequest, SimpleSearchRequest


# =====================================================================
# Configuration & Logging setup
# =====================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# API Router & Endpoints
# =====================================================================

api_router = APIRouter(prefix="/api")
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def _attach_image_urls(result_data: Dict[str, Any]) -> None:
    """Helper function to attach local image URLs to recommended items."""
    if not result_data.get("items"):
        return
        
    for item in result_data["items"]:
        article_id = item.get("article_id", "")
        if article_id:
            item["image_url"] = f"/api/images/{article_id}"


def _parse_json_form_field(value: str | None, default: Any, field_name: str) -> Any:
    """Parses optional JSON form fields used by multipart image search."""
    if value is None or value == "":
        return default

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in '{field_name}' form field.")


@api_router.post("/process")
async def process_endpoint(request: PipelineRequest) -> dict:
    """
    Primary endpoint for the m3 pipeline integration.
    
    Accepts structured retrieval_input and delegates it to the M2 Action Router.
    The router determines the appropriate handler and executes the multimodal RAG logic.
    """
    # Convert the validated Pydantic model back to a standard Python dictionary for the M2 internal router
    retrieval_dict = request.retrieval_input.dict() if request.retrieval_input else None
    
    action_type = retrieval_dict.get('action') if retrieval_dict else 'None (FEEDBACK/CHITCHAT)'
    logger.info(f"Received pipeline request. Action: {action_type}")
    
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

@api_router.post("/simple")
async def simple_search_endpoint(request: SimpleSearchRequest) -> dict:
    """
    Simplified endpoint for frontend developers and team members to easily test the M2 
    backend without constructing the complex M3 Memory Pipeline payload.
    It automatically wraps the 'query' into a 'catalog_search' action.
    """
    # Construct a default M3 payload
    retrieval_dict = {
        "action": "catalog_search",
        "retrieval_strategy": "FULL",
        "user_message": request.query,
        "items_in_context": {},
        "exclude_ids": [],
        "payload": {
            "filters": {},
            "preference_boosts": [],
            "penalties": {}
        }
    }
    
    logger.info(f"Received simple search request. Query: '{request.query}'")
    
    try:
        # Route the request through the central M2 dispatcher
        result = m2_router.process_retrieval_input(
            retrieval_input=retrieval_dict,
            memory_context={}
        )
        
        # Post-process: Attach local image URLs
        _attach_image_urls(result)
        
        return result

    except Exception as e:
        logger.error(f"Simple search processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during simple search processing.")


@api_router.post("/image-search")
async def image_search_endpoint(
    image: UploadFile = File(...),
    query: str = Form(""),
    filters: str = Form("{}"),
    preference_boosts: str = Form("[]"),
    penalties: str = Form("{}"),
    exclude_ids: str = Form("[]"),
) -> dict:
    """
    M2-only image search endpoint for frontend uploads.

    Accepts multipart/form-data and bypasses M1/M3 so image-based requests can
    be handled fully inside M2.
    """
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    parsed_filters = _parse_json_form_field(filters, {}, "filters")
    parsed_boosts = _parse_json_form_field(preference_boosts, [], "preference_boosts")
    parsed_penalties = _parse_json_form_field(penalties, {}, "penalties")
    parsed_exclude_ids = _parse_json_form_field(exclude_ids, [], "exclude_ids")

    image_path = None
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            suffix = ".jpg"

        image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        image_path.write_bytes(image_bytes)
        logger.info(f"Received image search request. Query: '{query}', file: {image.filename}")

        result = handle_image_catalog_search(
            image_path=str(image_path),
            user_message=query,
            filters=parsed_filters,
            boosts=parsed_boosts,
            penalties=parsed_penalties,
            exclude_ids=parsed_exclude_ids,
        )

        _attach_image_urls(result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during image search processing.")
    finally:
        await image.close()
        if image_path and image_path.exists():
            try:
                image_path.unlink()
            except OSError as e:
                logger.warning(f"Temporary upload cleanup skipped for {image_path}: {e}")

@api_router.get("/m2output")
async def mock_endpoint() -> dict:
    """
    Mock endpoint that returns a pre-defined JSON response instantly.
    Useful for frontend developers to test UI rendering without needing 
    the heavy ML models or vector DB running.
    """
    return {
        "action": "catalog_search",
        "success": True,
        "response_text": "Here are some mock dresses for UI testing.",
        "items": [
            {
                "article_id": "0123456789", # You might want to use a real ID from the dataset here if they need images
                "prod_name": "Mock Summer Dress",
                "colour_group_name": "Red",
                "product_type_name": "Dress",
                "product_group_name": "Garment Full body",
                "department_name": "Womens Everyday",
                "index_group_name": "Ladieswear",
                "graphical_appearance_name": "Solid",
                "detail_desc": "A lightweight mock dress.",
                "explanation": "This is a mock response.",
                "score": 0.99,
                "image_url": "/api/images/0123456789" # Mock image URL format
            }
        ],
        "error": None
    }


@api_router.get("/images/{article_id}")
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
    """Factory function to initialize and configure the FastAPI application."""
    app = FastAPI(
        title="M2 Conversational Recommender API",
        description="Multimodal RAG API for fashion recommendations.",
        version="1.0.0"
    )

    # Add CORS middleware to allow external clients to connect
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Note: In production, restrict this to your actual frontend URL
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register the modular API router
    app.include_router(api_router)
    
    return app

# Instantiate the application
app = create_app()
