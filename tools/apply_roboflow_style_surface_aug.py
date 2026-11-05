# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Create a leakage-safe Surface Crack dataset with Roboflow-style augmentations."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_KINDS = [None, "hflip", "affine", "appearance", "detail"]
VALIDATION_KINDS = [None]


def read_image(path: Path) -> np.ndarray:
    """Read an image without failing on Windows paths containing non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image without failing on Windows paths containing non-ASCII characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def read_yolo_label_file(path: Path) -> tuple[list[list[float]], list[int]]:
    """Read YOLO labels and return boxes plus class IDs."""
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number} 不是五列 YOLO 标签")
        class_id = int(float(values[0]))
        box = [float(v) for v in values[1:]]
        classes.append(class_id)
        boxes.append(box)
    return boxes, classes


def validate_yolo_label_file(path: Path) -> list[str]:
    """Return validation errors for a YOLO label file."""
    errors: list[str] = []
    try:
        boxes, classes = read_yolo_label_file(path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    for index, (box, class_id) in enumerate(zip(boxes, classes, strict=True)):
        if class_id != 0:
            errors.append(f"第 {index} 个框类别不是 0: {class_id}")
        if not all(np.isfinite(value) for value in box):
            errors.append(f"第 {index} 个框包含非有限数值")
        if len(box) != 4 or not all(0.0 <= value <= 1.0 for value in box):
            errors.append(f"第 {index} 个框超出 [0, 1]: {box}")
        if len(box) == 4 and (box[2] <= 0.0 or box[3] <= 0.0):
            errors.append(f"第 {index} 个框宽高无效: {box}")
    return errors


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.15, clip=True)


def build_pipeline(kind: str, seed: int = 20260731) -> A.Compose:
    """Build a deterministic augmentation pipeline for training or validation."""
    if kind == "hflip":
        transforms = [A.HorizontalFlip(p=1.0)]
    elif kind == "affine":
        raise ValueError("affine 使用手工几何路径，以保证每个输入框最多对应一个输出框")
    elif kind == "appearance":
        transforms = [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
                    A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                ],
                p=1.0,
            )
        ]
    elif kind == "detail":
        transforms = [
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.04), mean_range=(0.0, 0.0), p=1.0),
                    A.Sharpen(alpha=(0.08, 0.18), lightness=(0.9, 1.1), p=1.0),
                ],
                p=1.0,
            )
        ]
    elif kind == "val_mild":
        transforms = [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=1.0),
                    A.HorizontalFlip(p=1.0),
                ],
                p=1.0,
            )
        ]
    else:
        raise ValueError(f"未知增强类型: {kind}")
    image_only = kind in {"appearance", "detail"}
    pipeline = A.Compose(transforms, **({} if image_only else {"bbox_params": _bbox_params()}))
    pipeline.set_random_seed(seed)
    return pipeline


def _apply_manual_affine(
    image: np.ndarray, boxes: list[list[float]], classes: list[int], seed: int
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """Apply one affine transform while keeping a one-to-one box mapping."""
    height, width = image.shape[:2]
    rng = random.Random(seed)
    scale = rng.uniform(0.92, 1.08)
    angle = math.radians(rng.uniform(-10.0, 10.0))
    shear_x = math.tan(math.radians(rng.uniform(-4.0, 4.0)))
    shear_y = math.tan(math.radians(rng.uniform(-4.0, 4.0)))
    translate_x = rng.uniform(-0.04, 0.04) * width
    translate_y = rng.uniform(-0.04, 0.04) * height
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    translate_to_center = np.array([[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]])
    rotate_scale = np.array(
        [[scale * cos_angle, -scale * sin_angle, 0.0], [scale * sin_angle, scale * cos_angle, 0.0], [0.0, 0.0, 1.0]]
    )
    shear = np.array([[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]])
    translate_back = np.array([[1.0, 0.0, center_x + translate_x], [0.0, 1.0, center_y + translate_y], [0.0, 0.0, 1.0]])
    matrix = translate_back @ rotate_scale @ shear @ translate_to_center
    affine = matrix[:2].astype(np.float32)
    transformed_image = cv2.warpAffine(
        image,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    transformed_boxes: list[list[float]] = []
    transformed_classes: list[int] = []
    for box, class_id in zip(boxes, classes, strict=True):
        x_center, y_center, box_width, box_height = box
        corners = np.array(
            [
                [x_center - box_width / 2.0, y_center - box_height / 2.0, 1.0],
                [x_center + box_width / 2.0, y_center - box_height / 2.0, 1.0],
                [x_center + box_width / 2.0, y_center + box_height / 2.0, 1.0],
                [x_center - box_width / 2.0, y_center + box_height / 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        corners[:, 0] *= width
        corners[:, 1] *= height
        transformed = corners @ matrix.T
        x_min = max(0.0, float(transformed[:, 0].min()))
        y_min = max(0.0, float(transformed[:, 1].min()))
        x_max = min(float(width), float(transformed[:, 0].max()))
        y_max = min(float(height), float(transformed[:, 1].max()))
        new_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        old_area = max(1.0, box_width * width * box_height * height)
        if new_area <= 1.0 or new_area / old_area < 0.15:
            continue
        transformed_boxes.append(
            [
                ((x_min + x_max) / 2.0) / width,
                ((y_min + y_max) / 2.0) / height,
                (x_max - x_min) / width,
                (y_max - y_min) / height,
            ]
        )
        transformed_classes.append(int(class_id))
    return transformed_image, transformed_boxes, transformed_classes


def apply_pipeline(image: np.ndarray, boxes: list[list[float]], classes: list[int], kind: str, seed: int) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """Apply one augmentation and return the transformed image and labels."""
    if kind == "affine":
        return _apply_manual_affine(image, boxes, classes, seed)
    if kind in {"appearance", "detail"}:
        transformed_image = build_pipeline(kind, seed)(image=image)["image"]
        return transformed_image, boxes, classes
    pipeline = build_pipeline(kind, seed)
    result = pipeline(image=image, bboxes=boxes, class_labels=classes)
    transformed_boxes = [[float(value) for value in box] for box in result["bboxes"]]
    transformed_classes = [int(value) for value in result["class_labels"]]
    return result["image"], transformed_boxes, transformed_classes


def write_yolo_labels(path: Path, boxes: Iterable[Iterable[float]], classes: Iterable[int]) -> None:
    """Write clipped, valid YOLO labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box, class_id in zip(boxes, classes, strict=True):
        clipped = [min(1.0, max(0.0, float(value))) for value in box]
        if clipped[2] > 0.0 and clipped[3] > 0.0:
            lines.append(f"{int(class_id)} " + " ".join(f"{value:.6f}" for value in clipped))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _original_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.endswith("_orig"))


def _source_key(path: Path) -> str:
    return path.stem.removesuffix("_orig")


def _copy_or_augment(source_image: Path, source_label: Path, output_image: Path, output_label: Path, kind: str | None, seed: int) -> int:
    image = read_image(source_image)
    boxes, classes = read_yolo_label_file(source_label)
    if kind is not None:
        image, boxes, classes = apply_pipeline(image, boxes, classes, kind, seed)
    write_image(output_image, image)
    write_yolo_labels(output_label, boxes, classes)
    return len(boxes)


def build_dataset(source_root: Path, output_root: Path, seed: int = 20260731) -> dict:
    """Build the augmented train/validation folders and return a generation report."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖请先指定新目录: {output_root}")
    train_source = source_root / "images" / "train"
    train_labels = source_root / "labels" / "train"
    val_source = source_root / "images" / "val"
    val_labels = source_root / "labels" / "val"
    train_images = _original_images(train_source)
    val_images = _original_images(val_source)
    if not train_images or not val_images:
        raise FileNotFoundError("质量筛选目录中没有找到 *_orig 图片")

    counts = {"train": {"images": 0, "boxes": 0}, "val": {"images": 0, "boxes": 0}}
    source_train = {_source_key(path) for path in train_images}
    source_val = {_source_key(path) for path in val_images}
    if source_train & source_val:
        raise ValueError(f"训练/验证源文件发生交集: {sorted(source_train & source_val)[:5]}")

    for split, image_paths, label_dir, kinds, output_split in [
        ("train", train_images, train_labels, TRAIN_KINDS, "train"),
        ("val", val_images, val_labels, VALIDATION_KINDS, "val"),
    ]:
        for source_index, source_image in enumerate(image_paths):
            source_label = label_dir / f"{source_image.stem}.txt"
            if not source_label.exists():
                raise FileNotFoundError(source_label)
            for kind_index, kind in enumerate(kinds):
                suffix = "orig" if kind is None else kind
                output_name = f"{source_image.stem.removesuffix('_orig')}_{suffix}.jpg"
                output_image = output_root / "images" / output_split / output_name
                output_label = output_root / "labels" / output_split / f"{Path(output_name).stem}.txt"
                boxes = _copy_or_augment(source_image, source_label, output_image, output_label, kind, seed + source_index * 101 + kind_index)
                counts[output_split]["images"] += 1
                counts[output_split]["boxes"] += boxes

    output_root.mkdir(parents=True, exist_ok=True)
    base_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Surface_Crack"},
    }
    import yaml

    (output_root / "data.yaml").write_text(yaml.safe_dump(base_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "source_split": {"train_original": len(train_images), "val_original": len(val_images)},
        "train_augmentations": ["orig", "hflip", "affine", "appearance", "detail"],
        "validation_augmentations": ["orig"],
        "validation_is_original_only": True,
        "counts": counts,
        "train_val_source_overlap": sorted(source_train & source_val),
        "class_mapping": {"output_class": 0, "name": "Surface_Crack"},
        "notes": [
            "验证集只保留原始图片，不使用离线增强。",
            "几何增强使用 YOLO 框同步变换和反射边界填充。",
            "未使用强裁剪、Mosaic、CutMix 或大幅模糊。",
        ],
    }
    (output_root / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"C:\Users\23223\Desktop\CoatingVision_Surface_Crack_augmented_v2_augment_first\quality_filtered_split"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Users\23223\Desktop\CoatingVision_Surface_Crack_augmented_v2_augment_first\roboflow_style_train_only"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    report = build_dataset(args.source, args.output, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
