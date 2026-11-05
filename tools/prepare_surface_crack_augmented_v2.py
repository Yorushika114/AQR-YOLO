"""Create an improved surface-crack dataset: augment first, split second."""

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


def read_labels(path: Path) -> list[list[float]]:
    boxes = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        box = [float(value) for value in parts[1:]]
        if not all(0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"{path}:{line_number}: box value outside [0, 1]")
        boxes.append(box)
    return boxes


def write_labels(path: Path, boxes: list[list[float]]) -> None:
    path.write_text(
        "".join("0 " + " ".join(f"{value:.6f}" for value in box) + "\n" for box in boxes),
        encoding="utf-8",
    )


def make_transforms(seed: int) -> list[tuple[str, A.Compose | None]]:
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.15)
    return [
        ("orig", None),
        # Enlarge local crack structure while retaining at least one labeled box.
        (
            "local_zoom",
            A.Compose(
                [
                    A.AtLeastOneBBoxRandomCrop(height=360, width=480, erosion_factor=0.0, p=1.0),
                    A.Resize(height=480, width=640, interpolation=cv2.INTER_CUBIC, p=1.0),
                ],
                bbox_params=bbox_params,
                seed=seed + 1,
            ),
        ),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params, seed=seed + 2)),
        (
            "detail",
            A.Compose(
                [
                    A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.9),
                    A.UnsharpMask(blur_limit=3, sigma_limit=(0.1, 0.5), alpha=(0.15, 0.35), threshold=5, p=0.8),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.18, p=0.8),
                    A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0.0, 0.0), p=0.35),
                ],
                bbox_params=bbox_params,
                seed=seed + 3,
            ),
        ),
        (
            "affine",
            A.Compose(
                [
                    A.Affine(
                        scale=(0.88, 1.16),
                        translate_percent=(-0.06, 0.06),
                        rotate=(-7, 7),
                        shear=(-2, 2),
                        fit_output=False,
                        keep_ratio=True,
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    )
                ],
                bbox_params=bbox_params,
                seed=seed + 4,
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
        raise SystemExit(f"Source must contain images and labels/: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")

    images = {
        path.stem: path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    sources = []
    skipped_empty = []
    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.lower()):
        if label_path.name.lower() == "classes.txt" or label_path.stem not in images:
            continue
        boxes = read_labels(label_path)
        if not boxes:
            skipped_empty.append(label_path.stem)
            continue
        sources.append((label_path.stem, images[label_path.stem], boxes))
    if not sources:
        raise SystemExit("No non-empty labeled images found")

    transforms = make_transforms(args.seed)
    pool = []
    for source_index, (stem, image_path, boxes) in enumerate(sources):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for suffix, transform in transforms:
            if transform is None:
                transformed_image, transformed_boxes = image, boxes
            else:
                result = transform(image=image, bboxes=boxes, class_labels=[0] * len(boxes))
                transformed_image = result["image"]
                transformed_boxes = [list(map(float, box)) for box in result["bboxes"]]
            if not transformed_boxes:
                continue
            pool.append((stem, suffix, image_path.name, transformed_image, transformed_boxes, source_index))

    random.Random(args.seed).shuffle(pool)
    train_count = int(round(len(pool) * args.train_ratio))
    train_count = min(max(train_count, 1), len(pool) - 1)
    split_items = {"train": pool[:train_count], "val": pool[train_count:]}
    records = []

    for split, items in split_items.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem, suffix, source_name, image, boxes, source_index in items:
            output_stem = f"{stem}_{suffix}"
            image_path = image_dir / f"{output_stem}.jpg"
            label_path = label_dir / f"{output_stem}.txt"
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write {image_path}")
            write_labels(label_path, boxes)
            records.append({"split": split, "source": source_name, "output": image_path.name, "augmentation": suffix, "source_index": source_index})

    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Surface_Crack\n",
        encoding="utf-8",
    )
    (output / "generation_report.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "seed": args.seed,
                "augmentation_before_split": True,
                "source_labeled_images": len(sources),
                "skipped_empty_labels": skipped_empty,
                "augmented_pool_images": len(pool),
                "final_split": {split: len(items) for split, items in split_items.items()},
                "augmentations": [name for name, _ in transforms],
                "class_mapping": {"all_source_classes": [15], "output_class": 0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "pool": len(pool), "final_split": {split: len(items) for split, items in split_items.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
