"""The ResNet-18 transfer-learning architecture: build/freeze/unfreeze
helpers, and classify_product_image() -- the single-image inference
function Part 3's tool imports directly.
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

from part2.config import CLASS_NAMES, MODEL_PATH, NUM_CLASSES, get_device
from part2.data import build_transform


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_late_layers(model: nn.Module) -> None:
    """Fine-tuning fallback: unfreeze layer4 (and keep fc trainable),
    leaving conv1/bn1/layer1-3 frozen.
    """
    for name, param in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            param.requires_grad = True


def backbone_only(model: nn.Module) -> nn.Module:
    """Everything up to (and including) the average pool -- the frozen
    feature extractor used by the caching speed-up.
    """
    return nn.Sequential(*list(model.children())[:-1])


_inference_model = None


def _load_inference_model() -> nn.Module:
    global _inference_model
    if _inference_model is None:
        model = build_model(pretrained=False)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _inference_model = model
    return _inference_model


def classify_product_image(image_path: str) -> dict:
    """Loads models/product_classifier.pt, runs the training preprocessing on
    a single real image file, and returns the predicted category plus the
    model's confidence. Never reads the filename.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    model = _load_inference_model()
    transform = build_transform()
    image = Image.open(path).convert("L")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item())
    top3_idx = torch.topk(probabilities, k=3).indices.tolist()

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 4),
        "top3": [
            {"class": CLASS_NAMES[i], "confidence": round(float(probabilities[i]), 4)}
            for i in top3_idx
        ],
        "image_path": str(path),
    }
