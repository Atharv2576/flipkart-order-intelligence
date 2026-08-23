"""Speed-up step: the backbone is frozen, so its output for a given image is
constant across every training epoch. Running it once and caching the
512-d feature vectors turns 12 epochs x 60,000 forward passes into a single
extraction pass, collapsing an hour-scale CPU training loop into minutes.

Run as: python3 -m part2.cache_features
"""
import time

import torch
from torch.utils.data import DataLoader

from part2.config import BATCH_SIZE, FEATURE_CACHE_DIR, get_device
from part2.data import load_datasets
from part2.model import backbone_only, build_model, freeze_backbone


@torch.no_grad()
def extract_features(backbone, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    features, labels = [], []
    for images, targets in loader:
        images = images.to(device)
        out = backbone(images).view(images.size(0), -1)
        features.append(out.cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def main():
    device = get_device()
    print(f"device: {device}")

    train, val, test = load_datasets()
    model = build_model(pretrained=True)
    freeze_backbone(model)
    backbone = backbone_only(model).to(device)

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for name, dataset in (("train", train), ("val", val), ("test", test)):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        features, labels = extract_features(backbone, loader, device)
        torch.save({"features": features, "labels": labels}, FEATURE_CACHE_DIR / f"{name}.pt")
        print(f"cached {name}: {features.shape[0]} images -> {features.shape[1]}-d vectors")

    elapsed = time.time() - started
    print(f"feature extraction took {elapsed:.1f}s")


if __name__ == "__main__":
    main()
