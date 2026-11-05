"""Prepare a YOLO dataset from labeled CoatingVision surface-crack images.

The source images are split by original image first, then each split receives
the same offline augmentations.  This prevents augmented copies of one source
image from leaking between train and validation sets.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_labels(path: Path) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields, got {len(parts)}")
        class_id = int(float(parts[0]))
        box = [float(value) for value in parts[1:]]
        if not all(0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"{path}:{line_number}: normalized box value out of range")
        classes.append(class_id)
        boxes.append(box)
    return boxes, classes


def write_yolo_labels(path: Path, boxes: list[list[float]]) -> None:
    lines = ["0 " + " ".join(f"{value:.6f}" for value in box) for box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_transforms() -> list[tuple[str, A.Compose | None]]:
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2)
    return [
        ("orig", None),
        (
            "hflip",
            A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
        ),
        (
            "color",
            A.Compose(
                [
                    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=12, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=1.0),
                ],
                bbox_params=bbox_params,
            ),
        ),
        (
            "affine",
            A.Compose(
                [
                    A.Affine(
                        scale=(0.90, 1.10),
                        translate_percent=(-0.08, 0.08),
                        rotate=(-8, 8),
                        shear=(-2, 2),
                        fit_output=False,
                        p=1.0,
                    )
                ],
                bbox_params=bbox_params,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    labels_dir = source / "labels"
    if not source.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Source must contain an image directory and labels subdirectory: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite a non-empty output directory: {output}")

    images = {
        path.stem: path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    paired: list[tuple[str, Path, Path, list[list[float]]]] = []
    skipped_empty: list[str] = []
    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.lower()):
        if label_path.name.lower() == "classes.txt" or label_path.stem not in images:
            continue
        boxes, _ = read_yolo_labels(label_path)
        if not boxes:
            skipped_empty.append(label_path.stem)
            continue
        paired.append((label_path.stem, images[label_path.stem], label_path, boxes))
    if not paired:
        raise SystemExit("No non-empty image/YOLO-label pairs were found")

    rng = random.Random(args.seed)
    rng.shuffle(paired)
    train_count = int(round(len(paired) * args.train_ratio))
    train_count = min(max(train_count, 1), len(paired) - 1)
    splits = {"train": paired[:train_count], "val": paired[train_count:]}
    transforms = build_transforms()
    records: list[dict[str, object]] = []

    for split, samples in splits.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem, image_path, _, boxes in samples:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for suffix, transform in transforms:
                if transform is None:
                    augmented_image = image
                    augmented_boxes = boxes
                else:
                    transformed = transform(image=image, bboxes=boxes, class_labels=[0] * len(boxes))
                    augmented_image = transformed["image"]
                    augmented_boxes = [list(map(float, box)) for box in transformed["bboxes"]]
                if not augmented_boxes:
                    continue
                output_stem = f"{stem}_{suffix}"
                output_image = image_dir / f"{output_stem}.jpg"
                output_label = label_dir / f"{output_stem}.txt"
                bgr = cv2.cvtColor(np.asarray(augmented_image), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(output_image), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not write image: {output_image}")
                write_yolo_labels(output_label, augmented_boxes)
                records.append(
                    {
                        "split": split,
                        "source": image_path.name,
                        "output": output_image.name,
                        "augmentation": suffix,
                        "boxes": len(augmented_boxes),
                    }
                )

    (output / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Surface_Crack\n",
        encoding="utf-8",
    )
    (output / "generation_report.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "seed": args.seed,
                "source_labeled_images": len(paired),
                "skipped_empty_labels": skipped_empty,
                "source_split": {split: len(samples) for split, samples in splits.items()},
                "generated_images": {split: sum(item["split"] == split for item in records) for split in splits},
                "augmentations": [name for name, _ in transforms],
                "class_mapping": {"all_source_classes": [15], "output_class": 0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "source_split": {split: len(samples) for split, samples in splits.items()}, "generated": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
