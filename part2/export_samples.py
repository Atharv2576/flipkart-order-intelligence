"""Task 8: export real test-split images as actual .png files, since
torchvision stores Fashion-MNIST as raw IDX binaries and Part 3's tool needs
real image files on disk to point at -- never raw IDX data, never a
hardcoded label.

Run as: python3 -m part2.export_samples
"""
import json

from torchvision import datasets

from part2.config import CLASS_NAMES, DATA_DIR, SAMPLE_IMAGES_DIR


def slugify(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


def main():
    # No transform here -- we want the raw PIL images to write out as PNGs.
    test_raw = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)

    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen_classes = set()

    for index in range(len(test_raw)):
        image, label = test_raw[index]
        if label in seen_classes:
            continue
        seen_classes.add(label)

        filename = f"{label:02d}_{slugify(CLASS_NAMES[label])}.png"
        out_path = SAMPLE_IMAGES_DIR / filename
        image.save(out_path)

        manifest.append(
            {
                "filename": filename,
                "class": CLASS_NAMES[label],
                "class_index": label,
                "source": "Fashion-MNIST test split",
                "test_split_index": index,
            }
        )
        if len(seen_classes) == len(CLASS_NAMES):
            break

    (SAMPLE_IMAGES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(manifest)} images to {SAMPLE_IMAGES_DIR}")
    for entry in manifest:
        print(f"  {entry['filename']}  (test index {entry['test_split_index']})")

    return manifest


if __name__ == "__main__":
    main()
