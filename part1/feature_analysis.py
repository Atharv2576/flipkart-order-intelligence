"""Task 7: explain the winning Random Forest via impurity-based feature
importance and permutation importance on the held-out test split, and
compare the two rankings.

Run as: python3 -m part1.feature_analysis
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from part1.common import load_dataset, split_data
from part1.train_return_risk import MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def impurity_importance(pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_report(pipeline, X_test, y_test, n_repeats=10, random_state=42) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats, random_state=random_state
    )
    return (
        pd.DataFrame({"feature": X_test.columns, "importance_mean": result.importances_mean})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def run_analysis():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = joblib.load(MODEL_PATH)

    impurity_top = impurity_importance(pipeline).head(5)
    permutation_full = permutation_importance_report(pipeline, X_test, y_test)

    return {"impurity_top5": impurity_top, "permutation_full": permutation_full}


def render_report(result: dict) -> str:
    impurity = result["impurity_top5"]
    permutation = result["permutation_full"]

    lines = [
        "# Part 1 -- Feature importance (Task 7)",
        "",
        "## Top 5 by impurity-based `.feature_importances_`",
        "",
        impurity.to_markdown(index=False),
        "",
        "## Permutation importance on the held-out test split (n_repeats=10, scoring=roc_auc)",
        "",
        permutation.to_markdown(index=False),
        "",
        "## Why impurity-based importance can overrate a noisy continuous column",
        "",
        "`.feature_importances_` totals how much every split on a column reduced"
        " Gini impurity on the *training* data. A continuous column with many"
        " distinct values offers many candidate split points, so the forest gets"
        " more chances to find a cut that separates a node's training rows by"
        " chance -- and that lucky split still banks impurity-reduction credit."
        " A one-hot flag has exactly one possible split and no such advantage."
        " Permutation importance instead measures how much *test-set* ROC-AUC"
        " actually drops when a column is shuffled, which is immune to that bias"
        " because it never looks at how the tree was built, only at what the"
        " column contributes to unseen-data performance.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_analysis()
    report = render_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "part1_feature_importance.md").write_text(report, encoding="utf-8")
    comparison = result["impurity_top5"].merge(
        result["permutation_full"], on="feature", how="left"
    )
    comparison.to_csv(REPORTS_DIR / "part1_importance_comparison.csv", index=False)
    print(report)
