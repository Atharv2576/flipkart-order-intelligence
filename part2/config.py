"""Preprocessing contract, split sizes, and hyperparameters for Part 2's
Fashion-MNIST -> ResNet-18 transfer-learning categoriser.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fashion_mnist"
FEATURE_CACHE_DIR = ROOT / "data" / "feature_cache"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "product_classifier.pt"
METADATA_PATH = MODELS_DIR / "product_classifier_metadata.json"
SAMPLE_IMAGES_DIR = ROOT / "data" / "sample_images"

# Fashion-MNIST's fixed label order (Zalando Research).
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)

VAL_SIZE = 6000  # stratified, 600 per class, carved out of the 60k train half
INPUT_SIZE = 224  # ResNet-18's native ImageNet resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HEAD_EPOCHS = 12
FINETUNE_EPOCHS = 6
FINETUNE_LEARNING_RATE = 1e-4
FINETUNE_TRIGGER_ACC = 0.80  # unfreeze late layers only if feature-extraction falls short
RANDOM_SEED = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
