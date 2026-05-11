from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# =====================================================================
# Pydantic Request Models (Strict Template Injection from PDF)
# =====================================================================

# --- 1. Payload Sub-Models ---

class BoostTemplate(BaseModel):
    attribute: str
    value: str
    weight: float

class CatalogSearchPayload(BaseModel):
    filters: Dict[str, Any] = Field(default_factory=dict)
    preference_boosts: List[BoostTemplate] = Field(default_factory=list)
    penalties: Dict[str, List[str]] = Field(default_factory=dict)
    soft_constraints: Dict[str, Any] = Field(default_factory=dict)
    purchase_history_hints: Dict[str, Any] = Field(default_factory=dict)

class AttributeLookupPayload(BaseModel):
    article_id: str
    attribute_topic: str

class ItemComparePayload(BaseModel):
    article_id_a: str
    article_id_b: str
    comparison_dimension: str
    preference_weights: Dict[str, float] = Field(default_factory=dict)

class ClaimTemplate(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str
    status: str

class MatchedPrefTemplate(BaseModel):
    attribute_name: str
    attribute_value: str
    weight: float

class ExplanationGeneratePayload(BaseModel):
    article_id: str
    prior_claims: List[ClaimTemplate] = Field(default_factory=list)
    matched_prefs: List[MatchedPrefTemplate] = Field(default_factory=list)

class ItemDetailLookupPayload(BaseModel):
    article_id: str

# --- 2. Main Retrieval Input Model ---

class RetrievalInputModel(BaseModel):
    action: str
    retrieval_strategy: str
    user_message: str
    items_in_context: Dict[str, Optional[Dict[str, Any]]] = Field(default_factory=dict)
    exclude_ids: List[str] = Field(default_factory=list)
    
    # We use Union to allow any of the strict payload types defined above
    # The Dict fallback ensures we don't crash if m3 sends something slightly off
    payload: Union[
        CatalogSearchPayload, 
        AttributeLookupPayload, 
        ItemComparePayload, 
        ExplanationGeneratePayload, 
        ItemDetailLookupPayload,
        Dict[str, Any]  
    ]

# --- 3. Pipeline Request ---

class PipelineRequest(BaseModel):
    """
    Model for the structured input from the m3 Memory Pipeline.
    Strictly validates against the templates defined in retrieval_input_reference.pdf.
    """
    retrieval_input: Optional[RetrievalInputModel] = None   # None indicates FEEDBACK or CHITCHAT actions
    memory_context: Optional[Dict[str, Any]] = Field(default_factory=dict)

# --- 4. Simple Testing Request ---

class SimpleSearchRequest(BaseModel):
    """
    A simplified request model for frontend developers and other team members 
    to easily test the M2 backend without building the complex M3 payload.
    """
    query: str

