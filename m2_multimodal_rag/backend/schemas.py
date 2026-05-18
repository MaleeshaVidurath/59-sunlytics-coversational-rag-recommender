import logging
from typing import Optional, Dict, Any, List, Union, Literal, Type
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

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

ComparisonDimension = Literal[
    "price", "quality", "style_and_occasion", "material", "colour", "fit", "overall"
]

class ItemComparePayload(BaseModel):
    article_id_a: str
    article_id_b: str
    comparison_dimension: ComparisonDimension
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

# Maps each action string to its expected payload class.
# Used by the validator below to coerce and validate the payload
# using the action field rather than relying on Union ordering.
ACTION_PAYLOAD_MAP: Dict[str, Type[BaseModel]] = {
    "catalog_search": CatalogSearchPayload,
    "item_attribute_lookup": AttributeLookupPayload,
    "item_compare": ItemComparePayload,
    "explanation_generate": ExplanationGeneratePayload,
    "item_detail_lookup": ItemDetailLookupPayload,
}

# --- 2. Main Retrieval Input Model ---

class RetrievalInputModel(BaseModel):
    action: str
    retrieval_strategy: str
    user_message: str
    items_in_context: Dict[str, Optional[Dict[str, Any]]] = Field(default_factory=dict)
    exclude_ids: List[str] = Field(default_factory=list)
    payload: Union[
        CatalogSearchPayload,
        AttributeLookupPayload,
        ItemComparePayload,
        ExplanationGeneratePayload,
        ItemDetailLookupPayload,
        Dict[str, Any]
    ]

    @model_validator(mode='before')
    @classmethod
    def coerce_payload_by_action(cls, values: Any) -> Any:
        """
        Resolves the payload to the correct typed model using the action field,
        avoiding Union ordering ambiguity and logging when the fallback Dict is hit.
        """
        action = values.get('action') if isinstance(values, dict) else getattr(values, 'action', None)
        payload = values.get('payload') if isinstance(values, dict) else getattr(values, 'payload', None)

        if not isinstance(payload, dict):
            return values  # already a typed model instance, skip

        target_class = ACTION_PAYLOAD_MAP.get(action)
        if target_class:
            try:
                coerced = target_class(**payload)
                if isinstance(values, dict):
                    values['payload'] = coerced
            except Exception as e:
                logger.warning(
                    "[M2 Schema] Payload validation failed for action='%s': %s. "
                    "Falling back to raw Dict — handler may receive missing fields.",
                    action, e
                )
        else:
            logger.warning(
                "[M2 Schema] Unknown action='%s' — payload stored as raw Dict.", action
            )

        return values

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

