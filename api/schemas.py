from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    image_path: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    source: str
    confidence: float
    intent: Optional[str] = None
    node_path: list[str]
    order_id: Optional[int] = None
    blocked: bool = False
    groundedness: Optional[dict] = None
    retrieved_documents: Optional[list[dict]] = None
    tool_result: Optional[dict] = None


class ReturnRiskRequest(BaseModel):
    order_id: Optional[int] = None
    order_features: Optional[dict] = None


class OrderFeatures(BaseModel):
    product_category: str
    price_inr: float = Field(ge=0)
    discount_pct: float = Field(ge=0, le=100)
    payment_method: str
    customer_tenure_days: int = Field(ge=0)
    num_previous_orders: int = Field(ge=0)
    num_previous_returns: int = Field(ge=0)
    delivery_distance_km: float = Field(ge=0)
    delivery_days: int = Field(ge=0)
    is_weekend_order: int = Field(ge=0, le=1)
    rating_given: Optional[float] = Field(default=None, ge=0, le=5)
