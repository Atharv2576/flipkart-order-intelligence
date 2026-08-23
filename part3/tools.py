"""The two real tools the agent can call. Neither is a hardcoded stand-in:
check_return_risk loads Part 1's saved Random Forest pipeline and calls its
predict_proba; classify_product_image is Part 2's own inference function,
imported directly rather than copied.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from part1.common import FEATURES
from part2.model import classify_product_image  # noqa: F401 -- re-exported for Part 3 callers

ROOT = Path(__file__).resolve().parents[1]
RETURN_MODEL_PATH = ROOT / "models" / "return_risk_model.pkl"
RETURN_METADATA_PATH = ROOT / "models" / "return_risk_metadata.json"
ORDERS_PATH = ROOT / "orders_dataset.csv"

_return_model = None
_return_metadata = None
_orders_df = None


def _get_return_model():
    global _return_model
    if _return_model is None:
        _return_model = joblib.load(RETURN_MODEL_PATH)
    return _return_model


def _get_return_metadata() -> dict:
    global _return_metadata
    if _return_metadata is None:
        _return_metadata = json.loads(RETURN_METADATA_PATH.read_text())
    return _return_metadata


def _get_orders_df() -> pd.DataFrame:
    global _orders_df
    if _orders_df is None:
        _orders_df = pd.read_csv(ORDERS_PATH)
    return _orders_df


def lookup_order(order_id: int) -> dict | None:
    """Real order features from the committed dataset, with the label
    stripped -- the tool must never see the true `returned` value.
    """
    df = _get_orders_df()
    row = df.loc[df["order_id"] == order_id]
    if row.empty:
        return None
    features = row.iloc[0][FEATURES].to_dict()
    return features


def risk_bucket(probability: float, threshold_rf: float) -> str:
    if probability < threshold_rf:
        return "Low"
    if probability < threshold_rf + 0.15:
        return "Medium"
    return "High"


def check_return_risk(order_features: dict) -> dict:
    """Loads models/return_risk_model.pkl and calls its predict_proba. Risk
    bucket cut points are anchored to t*_rf -- the F1-maximising threshold
    computed on this saved Random Forest's own predict_proba output -- not a
    fixed 0.3/0.6 split and not the Logistic Regression's threshold, since
    two equally valid Random Forests can produce probability distributions
    concentrated in completely different ranges.
    """
    model = _get_return_model()
    metadata = _get_return_metadata()
    threshold_rf = metadata["threshold_rf"]

    frame = pd.DataFrame([{k: order_features.get(k) for k in FEATURES}])
    probability = float(model.predict_proba(frame)[0][1])

    return {
        "return_probability": round(probability, 4),
        "risk_bucket": risk_bucket(probability, threshold_rf),
        "threshold_rf": threshold_rf,
    }
