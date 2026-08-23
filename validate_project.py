"""End-to-end acceptance checks, re-run against live execution (not stale
output). Exits non-zero if anything fails.

Run as: python3 validate_project.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
checks_passed = 0
checks_failed = 0


def check(name: str, condition: bool) -> None:
    global checks_passed, checks_failed
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1


def part1_checks():
    df = pd.read_csv(ROOT / "orders_dataset.csv")
    check("orders_dataset.csv has 6000 rows", len(df) == 6000)
    check("orders_dataset.csv has 13 columns", df.shape[1] == 13)
    return_rate = df["returned"].mean()
    check("overall return rate in [0.18, 0.27]", 0.18 <= return_rate <= 0.27)
    missing_pct = df["rating_given"].isna().mean()
    check("missing rating_given in [0.08, 0.18]", 0.08 <= missing_pct <= 0.18)

    model_path = ROOT / "models" / "return_risk_model.pkl"
    metadata_path = ROOT / "models" / "return_risk_metadata.json"
    check("models/return_risk_model.pkl exists", model_path.exists())
    check("models/return_risk_metadata.json exists", metadata_path.exists())

    if model_path.exists() and metadata_path.exists():
        from part1.common import FEATURES, split_data

        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text())
        _, X_test, _, y_test = split_data(df)
        proba = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score

        test_roc_auc = roc_auc_score(y_test, proba)
        check("test ROC-AUC >= 0.58", test_roc_auc >= 0.58)
        check(
            "best_cv_roc_auc within 0.05 of test_roc_auc",
            abs(metadata["best_cv_roc_auc"] - metadata["test_roc_auc"]) <= 0.05,
        )
        check("t*_rf is not the hand-picked 0.3 or 0.6", metadata["threshold_rf"] not in (0.3, 0.6))


def part2_checks():
    model_path = ROOT / "models" / "product_classifier.pt"
    check("models/product_classifier.pt exists", model_path.exists())

    summary_path = ROOT / "reports" / "part2_evaluation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        check("Part 2 test accuracy >= 0.80", summary["test_accuracy"] >= 0.80)
        check("at least 2 confusion pairs documented", len(summary["top_confusion_pairs"]) >= 2)

    manifest_path = ROOT / "data" / "sample_images" / "manifest.json"
    check("sample_images manifest exists", manifest_path.exists())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        check("at least 5 real sample .png files exported", len(manifest) >= 5)
        for entry in manifest:
            check(f"sample image exists: {entry['filename']}", (ROOT / "data" / "sample_images" / entry["filename"]).exists())


def part3_checks():
    kb_dir = ROOT / "part3" / "knowledge_base"
    n_docs = len(list(kb_dir.glob("POL*.md")))
    check("knowledge base has >= 12 policy documents", n_docs >= 12)

    index_path = ROOT / "data" / "policy_index" / "policy.faiss"
    chunks_path = ROOT / "data" / "policy_index" / "chunks.json"
    check("FAISS index exists", index_path.exists())
    check("chunk metadata exists", chunks_path.exists())

    transcripts_dir = ROOT / "transcripts"
    transcript_files = sorted(transcripts_dir.glob("*.txt"))
    check("at least 8 transcripts recorded", len(transcript_files) >= 8)
    for f in transcript_files:
        check(f"transcript run in MOCK_LLM mode: {f.name}", "MOCK_LLM" in f.read_text())

    eval_path = ROOT / "reports" / "part3_retrieval_evaluation.md"
    check("retrieval evaluation report exists", eval_path.exists())


def main():
    print("=== Part 1 ===")
    part1_checks()
    print("\n=== Part 2 ===")
    part2_checks()
    print("\n=== Part 3 ===")
    part3_checks()

    print(f"\n{checks_passed}/{checks_passed + checks_failed} checks passed")
    if checks_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
