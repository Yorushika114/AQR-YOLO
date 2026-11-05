from pathlib import Path

import numpy as np

from tools.apply_roboflow_style_surface_aug import apply_pipeline
from tools.apply_roboflow_style_surface_aug import VALIDATION_KINDS
from tools.apply_roboflow_style_surface_aug import validate_yolo_label_file


def test_invalid_yolo_box_is_rejected(tmp_path: Path):
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5 1.2 0.2\n", encoding="utf-8")
    assert validate_yolo_label_file(label)


def test_affine_keeps_at_most_one_box_for_each_source_box():
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    boxes = [
        [0.296875, 0.542708, 0.293750, 0.914583],
        [0.670312, 0.469792, 0.193750, 0.422917],
        [0.871094, 0.118750, 0.179688, 0.233333],
        [0.958594, 0.488542, 0.070312, 0.422917],
        [0.676562, 0.871875, 0.118750, 0.256250],
        [0.967187, 0.891667, 0.065625, 0.208333],
    ]
    classes = [0] * len(boxes)
    _, transformed_boxes, transformed_classes = apply_pipeline(image, boxes, classes, "affine", seed=7)
    assert 0 < len(transformed_boxes) <= len(boxes)
    assert len(transformed_boxes) == len(transformed_classes)
    assert all(0.0 <= value <= 1.0 for box in transformed_boxes for value in box)


def test_appearance_does_not_change_boxes():
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    boxes = [[0.25, 0.35, 0.2, 0.3]]
    classes = [0]
    _, transformed_boxes, transformed_classes = apply_pipeline(image, boxes, classes, "appearance", seed=7)
    assert np.allclose(transformed_boxes, boxes, atol=1e-5)
    assert transformed_classes == classes


def test_validation_policy_keeps_original_images_only():
    assert VALIDATION_KINDS == [None]
