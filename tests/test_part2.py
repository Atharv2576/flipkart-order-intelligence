import json
import shutil

import numpy as np
import pytest

from part2.config import CLASS_NAMES, MODEL_PATH, SAMPLE_IMAGES_DIR
from part2.model import build_model, freeze_backbone

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run part2.train_product_classifier first")


def test_backbone_is_frozen_except_fc():
    model = build_model(pretrained=False)
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_sample_images_exist_and_are_real_files():
    manifest = json.loads((SAMPLE_IMAGES_DIR / "manifest.json").read_text())
    assert len(manifest) >= 5
    for entry in manifest:
        assert (SAMPLE_IMAGES_DIR / entry["filename"]).exists()
        assert entry["class"] in CLASS_NAMES


def test_classifier_matches_manifest_labels():
    from part2.model import classify_product_image

    manifest = json.loads((SAMPLE_IMAGES_DIR / "manifest.json").read_text())
    correct = 0
    for entry in manifest:
        result = classify_product_image(str(SAMPLE_IMAGES_DIR / entry["filename"]))
        correct += result["predicted_class"] == entry["class"]
    # all 10 committed samples are correctly classified test-split images
    assert correct == len(manifest)


def test_prediction_ignores_the_filename(tmp_path):
    from part2.model import classify_product_image

    manifest = json.loads((SAMPLE_IMAGES_DIR / "manifest.json").read_text())
    sneaker_entry = next(e for e in manifest if e["class"] == "Sneaker")
    original = SAMPLE_IMAGES_DIR / sneaker_entry["filename"]
    renamed = tmp_path / "99_definitely_a_handbag.png"
    shutil.copy(original, renamed)

    result_original = classify_product_image(str(original))
    result_renamed = classify_product_image(str(renamed))
    assert result_original["predicted_class"] == result_renamed["predicted_class"]
    assert result_original["confidence"] == result_renamed["confidence"]


def test_confusion_matrix_shape_and_row_sums():
    reports_dir = MODEL_PATH.parent.parent / "reports"
    cm = np.loadtxt(reports_dir / "part2_confusion_matrix.csv", delimiter=",", skiprows=1)
    assert cm.shape == (10, 10)
    row_sums = cm.sum(axis=1)
    assert row_sums.sum() == 10000


def test_missing_image_raises_file_not_found():
    from part2.model import classify_product_image

    with pytest.raises(FileNotFoundError):
        classify_product_image("data/sample_images/does_not_exist.png")
