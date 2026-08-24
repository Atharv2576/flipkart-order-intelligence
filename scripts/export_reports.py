"""Exports the real, already-committed Part 1/2/3 reports and metadata into
frontend/public/reports/*.json for the Insights screen. Every file here is
generated fresh from the actual artifacts -- nothing in frontend/ is a
second source of truth.

Run as: python3 scripts/export_reports.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS_DIR = ROOT / "reports"
OUT_DIR = ROOT / "frontend" / "public" / "reports"


def export_part1():
    sweep = pd.read_csv(REPORTS_DIR / "part1_threshold_sweep_logreg.csv")
    (OUT_DIR / "part1_threshold_sweep.json").write_text(sweep.to_json(orient="records"))

    importance = pd.read_csv(REPORTS_DIR / "part1_importance_comparison.csv")
    (OUT_DIR / "part1_importance.json").write_text(importance.to_json(orient="records"))

    by_category = pd.read_csv(REPORTS_DIR / "part1_subgroup_product_category.csv")
    by_payment = pd.read_csv(REPORTS_DIR / "part1_subgroup_payment_method.csv")
    (OUT_DIR / "part1_subgroups.json").write_text(
        json.dumps(
            {
                "by_product_category": json.loads(by_category.to_json(orient="records")),
                "by_payment_method": json.loads(by_payment.to_json(orient="records")),
            }
        )
    )

    metadata = json.loads((ROOT / "models" / "return_risk_metadata.json").read_text())
    (OUT_DIR / "part1_model.json").write_text(json.dumps(metadata))


def export_part2():
    from part2.config import CLASS_NAMES

    matrix = pd.read_csv(REPORTS_DIR / "part2_confusion_matrix.csv")
    (OUT_DIR / "part2_confusion_matrix.json").write_text(
        json.dumps({"class_names": CLASS_NAMES, "matrix": matrix.values.tolist()})
    )

    per_class = pd.read_csv(REPORTS_DIR / "part2_per_class_metrics.csv")
    (OUT_DIR / "part2_per_class.json").write_text(per_class.to_json(orient="records"))

    summary = json.loads((REPORTS_DIR / "part2_evaluation_summary.json").read_text())
    (OUT_DIR / "part2_summary.json").write_text(json.dumps(summary))


def export_part3():
    from part3.evaluate_retrieval import run_evaluation

    result = run_evaluation()
    (OUT_DIR / "part3_retrieval.json").write_text(json.dumps(result))


def export_project_meta():
    orders = pd.read_csv(ROOT / "orders_dataset.csv")
    return_risk_metadata = json.loads((ROOT / "models" / "return_risk_metadata.json").read_text())
    part2_summary = json.loads((REPORTS_DIR / "part2_evaluation_summary.json").read_text())

    meta = {
        "orders_total": int(len(orders)),
        "return_rate": round(float(orders["returned"].mean()), 4),
        "return_risk_test_roc_auc": return_risk_metadata["test_roc_auc"],
        "return_risk_threshold": return_risk_metadata["threshold_rf"],
        "image_classifier_test_accuracy": part2_summary["test_accuracy"],
        "policy_documents": len(list((ROOT / "part3" / "knowledge_base").glob("POL*.md"))),
    }
    (OUT_DIR / "project_meta.json").write_text(json.dumps(meta))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_part1()
    export_part2()
    export_part3()
    export_project_meta()
    print(f"exported reports to {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
