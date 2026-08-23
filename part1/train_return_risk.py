"""Part 1 Tasks 3-6, 9: leakage-free preprocessing, a DummyClassifier baseline,
a tuned Logistic Regression with a threshold sweep, a GridSearchCV-tuned
Random Forest, and persistence of the winning artifact with its own
F1-maximising probability threshold t*_rf.

Run as: python3 -m part1.train_return_risk
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from part1.common import (
    best_threshold,
    build_preprocessor,
    load_dataset,
    split_data,
    sweep_thresholds,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "return_risk_model.pkl"
METADATA_PATH = MODELS_DIR / "return_risk_metadata.json"


def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_class1": f1_score(y_test, y_pred, zero_division=0),
        "recall_class1": recall_score(y_test, y_pred, zero_division=0),
    }


def train_logistic(preprocessor, X_train, y_train, X_test, y_test) -> dict:
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)

    default_metrics = {
        "accuracy": accuracy_score(y_test, pred_default),
        "precision": precision_score(y_test, pred_default, zero_division=0),
        "recall": recall_score(y_test, pred_default, zero_division=0),
        "f1": f1_score(y_test, pred_default, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    sweep = sweep_thresholds(y_test, proba)
    best = best_threshold(sweep)

    return {
        "pipeline": pipeline,
        "default_metrics": default_metrics,
        "sweep": sweep,
        "best_threshold": float(best["threshold"]),
        "best_metrics": {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
        },
    }


def train_random_forest(preprocessor, X_train, y_train, X_test, y_test) -> dict:
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [6, 10, None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    test_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_proba)

    return {
        "search": search,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "test_roc_auc": float(test_roc_auc),
        "gap": float(abs(search.best_score_ - test_roc_auc)),
    }


def save_artifact(rf_result: dict, X_test, y_test) -> dict:
    """Persist first, reload, verify identical predict_proba, and only then
    compute t*_rf from the *reloaded* artifact's own probabilities -- so the
    saved threshold provably belongs to the file Part 3 will load.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pipeline = rf_result["search"].best_estimator_
    joblib.dump(best_pipeline, MODEL_PATH)

    reloaded = joblib.load(MODEL_PATH)
    in_memory_proba = best_pipeline.predict_proba(X_test)[:, 1]
    reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
    assert np.allclose(in_memory_proba, reloaded_proba), "reloaded model diverges from saved model"

    sweep = sweep_thresholds(y_test, reloaded_proba)
    best = best_threshold(sweep)
    t_star_rf = float(best["threshold"])

    metadata = {
        "threshold_rf": t_star_rf,
        "bucket_definition": {
            "low": f"probability < {t_star_rf}",
            "medium": f"{t_star_rf} <= probability < {round(t_star_rf + 0.15, 2)}",
            "high": f"probability >= {round(t_star_rf + 0.15, 2)}",
        },
        "test_metrics_at_threshold_rf": {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
        },
        "best_params": rf_result["best_params"],
        "best_cv_roc_auc": rf_result["best_cv_roc_auc"],
        "test_roc_auc": rf_result["test_roc_auc"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()

    baseline = train_baseline(X_train, y_train, X_test, y_test)
    logistic = train_logistic(preprocessor, X_train, y_train, X_test, y_test)
    rf_result = train_random_forest(preprocessor, X_train, y_train, X_test, y_test)
    metadata = save_artifact(rf_result, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logistic["sweep"].to_csv(REPORTS_DIR / "part1_threshold_sweep_logreg.csv", index=False)

    print("=== Baseline (DummyClassifier, most_frequent) ===")
    print(baseline)
    print("\n=== Logistic Regression @ 0.50 ===")
    print(logistic["default_metrics"])
    print(f"\nt*_logistic = {logistic['best_threshold']}")
    print(logistic["best_metrics"])
    print("\n=== Random Forest + GridSearchCV ===")
    print("best params:", rf_result["best_params"])
    print("best CV ROC-AUC:", round(rf_result["best_cv_roc_auc"], 4))
    print("test ROC-AUC:", round(rf_result["test_roc_auc"], 4))
    print("gap:", round(rf_result["gap"], 4))
    print("\n=== Saved artifact ===")
    print(f"t*_rf = {metadata['threshold_rf']}")
    print(metadata["test_metrics_at_threshold_rf"])

    return {
        "baseline": baseline,
        "logistic": logistic,
        "rf_result": rf_result,
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
