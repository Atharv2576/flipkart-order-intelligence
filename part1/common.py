"""Shared building blocks for the return-risk pipeline: feature contract,
leakage-free preprocessing, the train/test split, and the threshold sweep
utility shared by the Logistic Regression and Random Forest stages.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "orders_dataset.csv"

TARGET = "returned"
ID_COLUMN = "order_id"

NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# order_id is a row identifier, not a signal, and is deliberately excluded.
# returned is the label and must never appear on the feature side.

THRESHOLD_GRID = np.round(np.arange(0.10, 0.9001, 0.02), 2)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def sweep_thresholds(y_true, y_proba, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Precision/recall/F1 for the positive class at every threshold in grid."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    rows = []
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(sweep_df: pd.DataFrame) -> pd.Series:
    """The F1-maximising row of a threshold sweep — never hand-picked."""
    return sweep_df.loc[sweep_df["f1"].idxmax()]


def feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
