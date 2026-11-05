"""Quality-filter and split the current surface-crack dataset.

This intentionally uses only the original variants as sources, removes the
lowest-quality tail, stratifies the source-level split, and augments training
images only. Validation therefore contains no transformed copy of a training
image.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def read_boxes(path: Path) -> list[list[float]]:
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


def write_boxes(path: Path, boxes: list[list[float]]) -> None:
    path.write_text(
        "".join("0 " + " ".join(f"{value:.6f}" for value in box) + "\n" for box in boxes),
        encoding="utf-8",
    )


def make_train_transforms(seed: int) -> list[tuple[str, A.Compose]]:
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.15)
    return [
        (
            "hflip",
            A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params, seed=seed + 1),
        ),
        (
            "detail",
            A.Compose(
                [
                    A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.9),
                    A.UnsharpMask(blur_limit=3, sigma_limit=(0.1, 0.5), alpha=(0.15, 0.35), threshold=5, p=0.8),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.18, p=0.8),
                    A.GaussNoise(std_range=(0.01, 0.025), mean_range=(0.0, 0.0), p=0.25),
                ],
                bbox_params=bbox_params,
                seed=seed + 2,
            ),
        ),
        (
            "affine",
            A.Compose(
                [
                    A.Affine(
                        scale=(0.92, 1.10),
                        translate_percent=(-0.04, 0.04),
                        rotate=(-5, 5),
                        shear=(-1.5, 1.5),
                        keep_ratio=True,
                        fit_output=False,
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    )
                ],
                bbox_params=bbox_params,
                seed=seed + 3,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-percent", type=float, default=10.0)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")

    records = []
    for split in ("train", "val"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        for image_path in image_dir.glob("*_orig.jpg"):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            boxes = read_boxes(label_path)
            if not boxes:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            records.append(
                {
                    "stem": image_path.stem.removesuffix("_orig"),
                    "image_path": image_path,
                    "label_path": label_path,
                    "boxes": boxes,
                    "laplacian": laplacian,
                    "mean": float(gray.mean()),
                    "gray_std": float(gray.std()),
                    "box_count": len(boxes),
                }
            )
    if not records:
        raise SystemExit("No original labeled images were found")

    scores = np.asarray([record["laplacian"] for record in records], dtype=float)
    # Keep the upper (100 - drop_percent)% quality tail; the threshold is the
    # requested lower-tail percentile, not the upper-tail percentile.
    threshold = max(1.5, float(np.percentile(scores, args.drop_percent)))
    kept = [record for record in records if record["laplacian"] >= threshold and record["gray_std"] >= 2.5]
    rejected = [record for record in records if record not in kept]
    if len(kept) < 10:
        raise SystemExit(f"Quality filter left too few images: {len(kept)}")

    # Stratify by sharpness and annotation density, then take every fifth item
    # as validation. This keeps both distributions close while avoiding source
    # leakage between train and validation.
    q_edges = np.quantile([record["laplacian"] for record in kept], [0.25, 0.5, 0.75])
    groups = defaultdict(list)
    for record in kept:
        qbin = int(np.digitize(record["laplacian"], q_edges, right=True))
        dbin = min(record["box_count"] // 3, 3)
        groups[(qbin, dbin)].append(record)
    rng = random.Random(args.seed)
    ordered = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop())
    val_records = [record for index, record in enumerate(ordered) if index % 5 == 0]
    val_ids = {id(record) for record in val_records}
    train_records = [record for record in ordered if id(record) not in val_ids]

    transforms = make_train_transforms(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    generated = []
    for split, source_records in (("train", train_records), ("val", val_records)):
        for record in source_records:
            image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read {record['image_path']}")
            if split == "val":
                variants = [("orig", image, record["boxes"])]
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                variants = [("orig", rgb, record["boxes"])]
                for suffix, transform in transforms:
                    result = transform(image=rgb, bboxes=record["boxes"], class_labels=[0] * record["box_count"])
                    if result["bboxes"]:
                        variants.append((suffix, result["image"], [list(map(float, box)) for box in result["bboxes"]]))
            for suffix, variant_image, boxes in variants:
                output_stem = f"{record['stem']}_{suffix}"
                output_image = output / "images" / split / f"{output_stem}.jpg"
                output_label = output / "labels" / split / f"{output_stem}.txt"
                if suffix == "orig":
                    shutil.copy2(record["image_path"], output_image)
                else:
                    bgr = cv2.cvtColor(np.asarray(variant_image), cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(output_image), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError(f"Could not write {output_image}")
                write_boxes(output_label, boxes)
                generated.append({"split": split, "source": record["image_path"].name, "output": output_image.name, "augmentation": suffix})

    (output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Surface_Crack\n",
        encoding="utf-8",
    )
    (output / "generation_report.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "quality_filter": "original variants only; lowest-quality tail removed",
                "laplacian_threshold": threshold,
                "drop_percent_requested": args.drop_percent,
                "source_original_images": len(records),
                "quality_rejected": [record["image_path"].name for record in rejected],
                "source_split": {"train": len(train_records), "val": len(val_records)},
                "generated_images": {split: sum(item["split"] == split for item in generated) for split in ("train", "val")},
                "train_augmentations": ["orig", "hflip", "detail", "affine"],
                "validation_is_original_only": True,
                "class_mapping": {"all_source_classes": [15], "output_class": 0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "quality_rejected": len(rejected), "source_split": {"train": len(train_records), "val": len(val_records)}, "generated": len(generated)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
