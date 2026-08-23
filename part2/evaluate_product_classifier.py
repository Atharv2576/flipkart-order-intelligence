"""Final test-set evaluation: accuracy, confusion matrix, per-class
precision/recall/F1, and the highest-confusion category pairs read directly
off the matrix.

Run as: python3 -m part2.evaluate_product_classifier
"""
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from part2.config import BATCH_SIZE, CLASS_NAMES, MODEL_PATH, get_device
from part2.data import load_datasets
from part2.model import build_model

ROOT_REPORTS = MODEL_PATH.parent.parent / "reports"


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        preds = model(inputs).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def top_confusion_pairs(cm: np.ndarray, k: int = 4) -> list[dict]:
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:k]


def main():
    device = get_device()
    _, _, test = load_datasets()
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model = model.to(device)

    preds, labels = predict_all(model, test_loader, device)
    accuracy = float((preds == labels).mean())

    cm = confusion_matrix(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, zero_division=0)

    per_class = [
        {
            "class": CLASS_NAMES[i],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(CLASS_NAMES))
    ]

    confusion_pairs = top_confusion_pairs(cm)

    ROOT_REPORTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        ROOT_REPORTS / "part2_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
        header=",".join(CLASS_NAMES), comments="",
    )

    import csv
    with open(ROOT_REPORTS / "part2_per_class_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(per_class)

    result = {
        "test_accuracy": round(accuracy, 4),
        "per_class": per_class,
        "top_confusion_pairs": confusion_pairs,
    }
    (ROOT_REPORTS / "part2_evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"test accuracy: {accuracy:.4f}")
    print("top confusion pairs:")
    for pair in confusion_pairs:
        print(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")

    return result


if __name__ == "__main__":
    main()
