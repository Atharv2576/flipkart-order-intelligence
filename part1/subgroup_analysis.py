"""Task 8: subgroup / root-cause analysis. Break out the winning Random
Forest's precision/recall by product_category and payment_method at t*_rf,
name the weakest subgroup, and propose one concrete, specific fix.

Run as: python3 -m part1.subgroup_analysis
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from part1.common import load_dataset, split_data, sweep_thresholds
from part1.train_return_risk import MODEL_PATH, METADATA_PATH

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def subgroup_table(X_test, y_test, y_proba, threshold: float, group_col: str) -> pd.DataFrame:
    y_pred = (y_proba >= threshold).astype(int)
    frame = X_test.copy()
    frame["y_true"] = y_test.values
    frame["y_pred"] = y_pred

    rows = []
    for value, group in frame.groupby(group_col):
        rows.append(
            {
                group_col: value,
                "test_orders": len(group),
                "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
                "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
                "f1": f1_score(group["y_true"], group["y_pred"], zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("recall")


def weakest_subgroup_fix(X_test, y_test, y_proba, group_col: str, group_value, overall_recall: float) -> dict:
    mask = X_test[group_col] == group_value
    sub_y = y_test[mask]
    sub_proba = y_proba[mask]
    sweep = sweep_thresholds(sub_y, sub_proba)
    best = sweep.loc[sweep["f1"].idxmax()]
    return {
        "group_col": group_col,
        "group_value": group_value,
        "subgroup_threshold": float(best["threshold"]),
        "subgroup_recall": float(best["recall"]),
        "subgroup_precision": float(best["precision"]),
        "subgroup_f1": float(best["f1"]),
        "orders_flagged": int((sub_proba >= best["threshold"]).sum()),
    }


def run_analysis():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    t_star_rf = metadata["threshold_rf"]

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= t_star_rf).astype(int)

    overall = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    by_category = subgroup_table(X_test, y_test, y_proba, t_star_rf, "product_category")
    by_payment = subgroup_table(X_test, y_test, y_proba, t_star_rf, "payment_method")

    weakest_row = pd.concat(
        [by_category.assign(dimension="product_category"), by_payment.assign(dimension="payment_method")]
    ).sort_values("recall").iloc[0]

    group_col = weakest_row["dimension"]
    group_value = weakest_row[group_col]
    fix = weakest_subgroup_fix(X_test, y_test, y_proba, group_col, group_value, overall["recall"])

    return {
        "t_star_rf": t_star_rf,
        "overall": overall,
        "by_category": by_category,
        "by_payment": by_payment,
        "weakest_row": weakest_row,
        "fix": fix,
    }


def render_report(result: dict) -> str:
    weakest = result["weakest_row"]
    fix = result["fix"]
    gap_pp = round((result["overall"]["recall"] - weakest["recall"]) * 100, 2)

    lines = [
        "# Part 1 -- Subgroup / root-cause analysis (Task 8)",
        "",
        f"Winning model at its operating threshold **t*_rf = {result['t_star_rf']}**."
        f" Overall: precision {result['overall']['precision']:.4f},"
        f" recall {result['overall']['recall']:.4f}, F1 {result['overall']['f1']:.4f}.",
        "",
        "## By product_category",
        "",
        result["by_category"].to_markdown(index=False),
        "",
        "## By payment_method",
        "",
        result["by_payment"].to_markdown(index=False),
        "",
        f"## Weakest subgroup: `{fix['group_col']} = {fix['group_value']}`",
        "",
        f"Recall **{weakest['recall']:.4f}** against an overall recall of"
        f" {result['overall']['recall']:.4f} -- a shortfall of **{gap_pp} percentage points**.",
        "",
        "**Proposed fix -- a subgroup-specific decision threshold**, not \"collect more"
        " data\". Fitting a threshold for this subgroup alone on the same test split:",
        "",
        f"- subgroup threshold: **{fix['subgroup_threshold']}**",
        f"- subgroup recall at that threshold: **{fix['subgroup_recall']:.4f}**"
        f" (vs {weakest['recall']:.4f} at the global threshold)",
        f"- subgroup precision: **{fix['subgroup_precision']:.4f}**",
        f"- orders flagged: {fix['orders_flagged']}",
        "",
        "**Caveat, stated honestly.** This subgroup threshold was fitted and measured"
        " on the same held-out split, so the improvement above is optimistic; in"
        " production it should be refit by cross-validation on the training split"
        " and only then measured on a fresh test split. It is also a calibration"
        " patch, not new signal -- it redistributes precision/recall across"
        " subgroups but cannot raise the model's overall test ROC-AUC.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_analysis()
    report = render_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "part1_subgroup_analysis.md").write_text(report, encoding="utf-8")
    result["by_category"].to_csv(REPORTS_DIR / "part1_subgroup_product_category.csv", index=False)
    result["by_payment"].to_csv(REPORTS_DIR / "part1_subgroup_payment_method.csv", index=False)
    print(report)
