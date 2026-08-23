"""Reload the saved Part 1 artifact from disk (no retraining) and re-verify
its metrics against the held-out test split -- a sanity check that the
committed models/return_risk_model.pkl is exactly what the reports describe.

Run as: python3 -m part1.evaluate_return_risk
"""
import json

import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from part1.common import load_dataset, split_data
from part1.train_return_risk import MODEL_PATH, METADATA_PATH


def main():
    df = load_dataset()
    _, X_test, _, y_test = split_data(df)

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    t_star_rf = metadata["threshold_rf"]

    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= t_star_rf).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "precision_at_t_star": precision_score(y_test, y_pred, zero_division=0),
        "recall_at_t_star": recall_score(y_test, y_pred, zero_division=0),
        "f1_at_t_star": f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"Loaded {MODEL_PATH.name}, t*_rf = {t_star_rf}")
    for key, value in metrics.items():
        print(f"  {key}: {round(value, 4)}")

    expected = metadata["test_metrics_at_threshold_rf"]
    assert abs(metrics["precision_at_t_star"] - expected["precision"]) < 1e-9
    assert abs(metrics["recall_at_t_star"] - expected["recall"]) < 1e-9
    print("Reloaded artifact matches models/return_risk_metadata.json exactly.")
    return metrics


if __name__ == "__main__":
    main()
