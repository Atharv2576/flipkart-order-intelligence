"""Train the classifier head on cached frozen-backbone features (fast), and
conditionally fine-tune layer4 end-to-end only if that isn't enough to clear
80% validation accuracy.

Run as: python3 -m part2.train_product_classifier
"""
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from part2.config import (
    BATCH_SIZE,
    FEATURE_CACHE_DIR,
    FINETUNE_EPOCHS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_TRIGGER_ACC,
    HEAD_EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_CLASSES,
    get_device,
)
from part2.data import load_datasets
from part2.model import build_model, freeze_backbone, unfreeze_late_layers

ROOT_REPORTS = MODEL_PATH.parent.parent / "reports"


def load_cached(name: str):
    payload = torch.load(FEATURE_CACHE_DIR / f"{name}.pt")
    return payload["features"], payload["labels"]


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_head(device) -> tuple[nn.Linear, float, list[float]]:
    train_feats, train_labels = load_cached("train")
    val_feats, val_labels = load_cached("val")

    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=BATCH_SIZE, shuffle=False)

    head = nn.Linear(train_feats.shape[1], NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    val_acc_by_epoch = []
    for epoch in range(HEAD_EPOCHS):
        head.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(head(inputs), labels)
            loss.backward()
            optimizer.step()
        val_acc = accuracy(head, val_loader, device)
        val_acc_by_epoch.append(val_acc)

    return head, val_acc_by_epoch[-1], val_acc_by_epoch


def finetune(model, device) -> float:
    """Unfreeze layer4 and continue training end-to-end on raw images at a
    lower learning rate -- only invoked if feature-extraction alone falls
    short of the 80% validation-accuracy bar.
    """
    unfreeze_late_layers(model)
    train, val, _ = load_datasets()
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=FINETUNE_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    return accuracy(model, val_loader, device)


def main():
    device = get_device()
    print(f"device: {device}")

    started = time.time()
    head, head_val_acc, val_acc_by_epoch = train_head(device)
    head_elapsed = time.time() - started
    print(f"head training: val_acc={head_val_acc:.4f} in {head_elapsed:.1f}s")

    model = build_model(pretrained=True)
    freeze_backbone(model)
    model.fc.load_state_dict(head.state_dict())
    model = model.to(device)

    finetune_triggered = head_val_acc < FINETUNE_TRIGGER_ACC
    finetune_val_acc = None
    if finetune_triggered:
        print(f"head val_acc {head_val_acc:.4f} < {FINETUNE_TRIGGER_ACC}, fine-tuning layer4...")
        finetune_val_acc = finetune(model, device)
        print(f"after fine-tuning: val_acc={finetune_val_acc:.4f}")
    else:
        print(f"head val_acc {head_val_acc:.4f} >= {FINETUNE_TRIGGER_ACC}, fine-tuning skipped")

    model = model.to("cpu")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    log = {
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "head_epochs": HEAD_EPOCHS,
        "head_val_acc_by_epoch": val_acc_by_epoch,
        "head_val_acc_final": head_val_acc,
        "head_training_seconds": round(head_elapsed, 1),
        "finetune_triggered": finetune_triggered,
        "finetune_trigger_threshold": FINETUNE_TRIGGER_ACC,
        "finetune_val_acc": finetune_val_acc,
        "finetune_epochs": FINETUNE_EPOCHS if finetune_triggered else 0,
        "finetune_learning_rate": FINETUNE_LEARNING_RATE if finetune_triggered else None,
    }
    ROOT_REPORTS.mkdir(parents=True, exist_ok=True)
    (ROOT_REPORTS / "part2_training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    return log


if __name__ == "__main__":
    main()
