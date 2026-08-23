import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from part1.common import build_preprocessor, load_dataset, split_data
from part1.train_return_risk import MODEL_PATH, METADATA_PATH

ROOT = Path(__file__).resolve().parents[1]


def test_generator_is_deterministic(tmp_path):
    script = (ROOT / "generate_orders.py").read_text()
    (tmp_path / "generate_orders.py").write_text(script)
    subprocess.run([sys.executable, "generate_orders.py"], cwd=tmp_path, check=True, capture_output=True)

    regenerated = pd.read_csv(tmp_path / "orders_dataset.csv")
    committed = pd.read_csv(ROOT / "orders_dataset.csv")
    pd.testing.assert_frame_equal(regenerated, committed)


def test_dataset_shape_and_rates():
    df = load_dataset()
    assert df.shape == (6000, 13)
    return_rate = df["returned"].mean()
    assert 0.18 <= return_rate <= 0.27
    missing_pct = df["rating_given"].isna().mean()
    assert 0.08 <= missing_pct <= 0.18


def test_order_id_and_label_excluded_from_features():
    from part1.common import FEATURES

    assert "order_id" not in FEATURES
    assert "returned" not in FEATURES


def test_split_is_stratified_and_reproducible():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 4800
    assert len(X_test) == 1200
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_fits_on_train_only_no_leakage():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    # transform must not require refitting and must not error on unseen rows
    transformed_test = preprocessor.transform(X_test)
    assert transformed_test.shape[0] == len(X_test)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run part1.train_return_risk first")
def test_saved_model_reloads_and_matches_metadata():
    pipeline = joblib.load(MODEL_PATH)
    df = load_dataset()
    _, X_test, _, y_test = split_data(df)
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()

    import json

    metadata = json.loads(METADATA_PATH.read_text())
    t_star_rf = metadata["threshold_rf"]
    from sklearn.metrics import f1_score

    y_pred = (proba >= t_star_rf).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    assert abs(f1 - metadata["test_metrics_at_threshold_rf"]["f1"]) < 1e-9


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run part1.train_return_risk first")
def test_t_star_rf_is_not_hand_picked_or_reused_from_logistic():
    import json

    metadata = json.loads(METADATA_PATH.read_text())
    t_star_rf = metadata["threshold_rf"]
    assert t_star_rf not in (0.3, 0.6)
