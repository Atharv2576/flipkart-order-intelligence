"""The HTTP layer the console talks to. Transport only -- every route
delegates to the real agent or a real saved model; there is no answer table
anywhere in this package.

Run as: python3 -m api.main   ->  http://127.0.0.1:8000  (docs at /docs)
"""
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse, OrderFeatures, ReturnRiskRequest
from part1.common import FEATURES
from part3.agent import Conversation
from part3.config import GROUNDEDNESS_THRESHOLD, INTENT_ROUTING_FLOOR, USE_LIVE_LLM
from part3.kb import load_policy_documents
from part3.tools import check_return_risk, classify_product_image, lookup_order
from part3.warmup import warmup

logger = logging.getLogger("api")

ROOT = Path(__file__).resolve().parents[1]

@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()
    yield


app = FastAPI(title="Flipkart Order Intelligence & Support API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_conversations: dict[str, Conversation] = {}
_orders_df: pd.DataFrame | None = None


def _get_orders_df() -> pd.DataFrame:
    global _orders_df
    if _orders_df is None:
        _orders_df = pd.read_csv(ROOT / "orders_dataset.csv")
    return _orders_df


def _get_conversation(conversation_id: str | None) -> tuple[str, Conversation]:
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]
    new_id = conversation_id or str(uuid.uuid4())
    conversation = Conversation()
    _conversations[new_id] = conversation
    return new_id, conversation


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    conversation_id, conversation = _get_conversation(request.conversation_id)
    try:
        result = conversation.ask(request.message, image_path=request.image_path)
    except Exception:
        logger.exception("agent turn failed")
        raise HTTPException(status_code=500, detail="The agent could not process that message.")

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["response"]["answer"],
        source=result["response"]["source"],
        confidence=result["response"]["confidence"],
        intent=result.get("intent"),
        node_path=result["trace"],
        order_id=result.get("order_id"),
        blocked=bool(result.get("blocked")),
        groundedness=result.get("groundedness"),
        retrieved_documents=result.get("retrieved_documents"),
        tool_result=result.get("tool_result"),
    )


@app.post("/api/conversations/reset")
def reset_conversation():
    conversation_id = str(uuid.uuid4())
    _conversations[conversation_id] = Conversation()
    return {"conversation_id": conversation_id}


@app.get("/api/conversations/{conversation_id}/state")
def conversation_state(conversation_id: str):
    conversation = _conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Unknown conversation id.")
    return {
        "conversation_id": conversation_id,
        "turn_index": conversation.turn_index,
        "order_id": conversation.order_id,
        "order_features": conversation.order_features,
        "history": conversation.history,
    }


@app.post("/api/return-risk")
def return_risk(request: ReturnRiskRequest):
    if request.order_features is not None:
        order_features = request.order_features
    elif request.order_id is not None:
        order_features = lookup_order(request.order_id)
        if order_features is None:
            raise HTTPException(status_code=404, detail=f"No order with id {request.order_id}.")
    else:
        raise HTTPException(status_code=400, detail="Provide either order_id or order_features.")

    try:
        result = check_return_risk(order_features)
    except Exception:
        logger.exception("return-risk scoring failed")
        raise HTTPException(status_code=500, detail="Could not score this order.")
    return {"order_id": request.order_id, "order_features": order_features, **result}


@app.post("/api/classify")
def classify(filename: str):
    """Classifies one of the real, already-committed sample images by
    filename -- this project does not accept arbitrary uploaded images
    outside the Part 2 test-split exports.
    """
    safe_name = Path(filename).name
    image_path = ROOT / "data" / "sample_images" / safe_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Unknown sample image.")
    try:
        return classify_product_image(str(image_path))
    except Exception:
        logger.exception("classification failed")
        raise HTTPException(status_code=500, detail="Could not classify this image.")


@app.get("/api/sample-images")
def sample_images():
    import json

    manifest_path = ROOT / "data" / "sample_images" / "manifest.json"
    return json.loads(manifest_path.read_text())


@app.get("/api/policies")
def policies():
    documents = load_policy_documents()
    return {"count": len(documents), "documents": documents}


@app.get("/api/orders")
def list_orders(
    order_id: int | None = None,
    product_category: str | None = None,
    payment_method: str | None = None,
    limit: int = 25,
    offset: int = 0,
):
    df = _get_orders_df()
    if order_id is not None:
        df = df[df["order_id"] == order_id]
    if product_category:
        df = df[df["product_category"] == product_category]
    if payment_method:
        df = df[df["payment_method"] == payment_method]

    total = len(df)
    limit = max(1, min(limit, 100))
    page = df.iloc[offset : offset + limit]
    # pandas' own to_json (not the stdlib json module) is what correctly
    # turns a missing rating_given (NaN) into JSON null instead of raising.
    import json as _json

    orders = _json.loads(page.to_json(orient="records"))
    return {"total": total, "limit": limit, "offset": offset, "orders": orders}


@app.get("/api/orders/{order_id}")
def order_detail(order_id: int):
    features = lookup_order(order_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"No order with id {order_id}.")
    risk = check_return_risk(features)
    return {"order_id": order_id, "order_features": features, "risk": risk}


@app.get("/api/status")
def status():
    components = {}

    try:
        components["knowledge_base"] = {"ready": len(load_policy_documents()) >= 12}
    except Exception as exc:  # noqa: BLE001
        components["knowledge_base"] = {"ready": False, "detail": str(exc)}

    try:
        from part3.retrieval import _load

        index, chunks = _load()
        components["faiss_index"] = {"ready": index.ntotal == len(chunks)}
    except Exception as exc:  # noqa: BLE001
        components["faiss_index"] = {"ready": False, "detail": str(exc)}

    try:
        from part1.train_return_risk import MODEL_PATH as RISK_MODEL_PATH

        components["return_risk_model"] = {"ready": RISK_MODEL_PATH.exists()}
    except Exception as exc:  # noqa: BLE001
        components["return_risk_model"] = {"ready": False, "detail": str(exc)}

    try:
        from part2.config import MODEL_PATH as IMAGE_MODEL_PATH

        components["image_classifier_model"] = {"ready": IMAGE_MODEL_PATH.exists()}
    except Exception as exc:  # noqa: BLE001
        components["image_classifier_model"] = {"ready": False, "detail": str(exc)}

    return {
        "components": components,
        "mode": "live_llm" if USE_LIVE_LLM else "mock_llm",
        "groundedness_threshold": GROUNDEDNESS_THRESHOLD,
        "intent_routing_floor": INTENT_ROUTING_FLOOR,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "live_llm" if USE_LIVE_LLM else "mock_llm"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
