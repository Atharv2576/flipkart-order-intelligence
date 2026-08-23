"""Fashion-MNIST loading, the stratified validation carve-out, and the
preprocessing transform for a pretrained ResNet-18 backbone.
"""
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms

from part2.config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    RANDOM_SEED,
    VAL_SIZE,
)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_raw_datasets():
    """The canonical Fashion-MNIST train/test split, downloaded once and cached."""
    transform = build_transform()
    train_full = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
    return train_full, test


def stratified_train_val_split(train_full, val_size: int = VAL_SIZE, seed: int = RANDOM_SEED):
    """Carve a stratified validation split out of the 60k train half, leaving
    the canonical test split completely untouched.
    """
    targets = train_full.targets.numpy()
    indices = list(range(len(train_full)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=targets
    )
    return Subset(train_full, train_idx), Subset(train_full, val_idx)


def load_datasets():
    train_full, test = load_raw_datasets()
    train, val = stratified_train_val_split(train_full)
    return train, val, test
