# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Coordinate-safe crop views and cross-view candidate fusion for CBSC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropView:
    """Metadata describing one crop in the original image coordinate system."""

    offset: tuple[int, int]
    scale: tuple[float, float]
    orig_shape: tuple[int, int]
    crop_shape: tuple[int, int]
    is_boundary: bool = False


def _positions(length: int, crop: int, step: int) -> list[int]:
    """Return regular crop starts plus the final edge-aligned start."""
    last = max(length - crop, 0)
    starts = list(range(0, last + 1, step))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def generate_overlapping_crops(
    image_shape: tuple[int, int] | list[int],
    crop_size: int | tuple[int, int],
    overlap: float = 0.25,
    max_crops: int = 16,
) -> list[CropView]:
    """Generate deterministic edge-aligned overlapping crop metadata."""
    height, width = int(image_shape[0]), int(image_shape[1])
    crop_height, crop_width = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if min(height, width, crop_height, crop_width) <= 0:
        raise ValueError("image_shape and crop_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if max_crops < 1:
        raise ValueError("max_crops must be positive")

    step_y = max(int(round(crop_height * (1 - overlap))), 1)
    step_x = max(int(round(crop_width * (1 - overlap))), 1)
    views = []
    for top in _positions(height, crop_height, step_y):
        for left in _positions(width, crop_width, step_x):
            actual_height = min(crop_height, height - top)
            actual_width = min(crop_width, width - left)
            views.append(
                CropView(
                    offset=(left, top),
                    scale=(1.0, 1.0),
                    orig_shape=(height, width),
                    crop_shape=(actual_height, actual_width),
                    is_boundary=top + actual_height == height or left + actual_width == width,
                )
            )
            if len(views) >= max_crops:
                return views
    return views


def map_boxes_to_crop(boxes: torch.Tensor, view: CropView) -> torch.Tensor:
    """Map global xyxy boxes to clipped crop coordinates."""
    boxes = boxes.clone()
    offset_x, offset_y = view.offset
    scale_x, scale_y = view.scale
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("crop view scales must be positive")
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - offset_x) / scale_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - offset_y) / scale_y
    boxes[:, 0] = boxes[:, 0].clamp(0, view.crop_shape[1])
    boxes[:, 2] = boxes[:, 2].clamp(0, view.crop_shape[1])
    boxes[:, 1] = boxes[:, 1].clamp(0, view.crop_shape[0])
    boxes[:, 3] = boxes[:, 3].clamp(0, view.crop_shape[0])
    return boxes


def map_boxes_to_global(boxes: torch.Tensor, view: CropView) -> torch.Tensor:
    """Map crop xyxy boxes back to clipped global coordinates."""
    boxes = boxes.clone()
    offset_x, offset_y = view.offset
    scale_x, scale_y = view.scale
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("crop view scales must be positive")
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + offset_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + offset_y
    boxes[:, 0] = boxes[:, 0].clamp(0, view.orig_shape[1])
    boxes[:, 2] = boxes[:, 2].clamp(0, view.orig_shape[1])
    boxes[:, 1] = boxes[:, 1].clamp(0, view.orig_shape[0])
    boxes[:, 3] = boxes[:, 3].clamp(0, view.orig_shape[0])
    return boxes


def build_cbsc_training_views(
    image: torch.Tensor,
    bboxes: torch.Tensor,
    cls: torch.Tensor,
    batch_idx: torch.Tensor,
    crop_size: int | tuple[int, int],
    overlap: float = 0.25,
    max_crops: int = 16,
) -> dict[str, object]:
    """Build resized crop views and crop-local labels for CBSC training.

    Args:
        image (torch.Tensor): Image batch with shape ``(B, C, H, W)``.
        bboxes (torch.Tensor): Normalized global ``xywh`` labels.
        cls (torch.Tensor): Class labels with shape ``(N, 1)`` or ``(N,)``.
        batch_idx (torch.Tensor): Source image index for each target.
        crop_size (int | tuple[int, int]): Crop output size.
        overlap (float): Fractional crop overlap.
        max_crops (int): Maximum crops per image.

    Returns:
        (dict[str, object]): Crop tensor, crop labels, metadata, and source indices.
    """
    if image.ndim != 4:
        raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    crop_height, crop_width = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    views = generate_overlapping_crops((height, width), crop_size, overlap, max_crops)
    crop_images = []
    crop_boxes = []
    crop_classes = []
    crop_batch_idx = []
    crop_source_indices = []
    crop_boundary = []
    metadata = []

    for source_index in range(image.shape[0]):
        source_targets = torch.where(batch_idx == source_index)[0]
        for view in views:
            left, top = view.offset
            actual_height, actual_width = view.crop_shape
            crop = image[source_index : source_index + 1, :, top : top + actual_height, left : left + actual_width]
            crop_images.append(F.interpolate(crop, size=(crop_height, crop_width), mode="bilinear").squeeze(0))
            resized_view = CropView(
                offset=view.offset,
                scale=(actual_width / crop_width, actual_height / crop_height),
                orig_shape=view.orig_shape,
                crop_shape=(crop_height, crop_width),
                is_boundary=view.is_boundary,
            )
            metadata.append(resized_view)
            crop_source_indices.append(source_index)
            crop_boundary.append(resized_view.is_boundary)
            crop_index = len(crop_images) - 1
            if len(source_targets):
                global_boxes = bboxes[source_targets].clone()
                normalized_targets = bboxes[source_targets]
                global_boxes[:, 0] = (normalized_targets[:, 0] - normalized_targets[:, 2] / 2) * width
                global_boxes[:, 2] = (normalized_targets[:, 0] + normalized_targets[:, 2] / 2) * width
                global_boxes[:, 1] = (normalized_targets[:, 1] - normalized_targets[:, 3] / 2) * height
                global_boxes[:, 3] = (normalized_targets[:, 1] + normalized_targets[:, 3] / 2) * height
                local_boxes = map_boxes_to_crop(global_boxes, resized_view)
                local_widths = (local_boxes[:, 2] - local_boxes[:, 0]).clamp_min(0)
                local_heights = (local_boxes[:, 3] - local_boxes[:, 1]).clamp_min(0)
                keep = (local_widths > 1) & (local_heights > 1)
                if keep.any():
                    kept = local_boxes[keep]
                    normalized = torch.stack(
                        [
                            (kept[:, 0] + kept[:, 2]) / (2 * crop_width),
                            (kept[:, 1] + kept[:, 3]) / (2 * crop_height),
                            (kept[:, 2] - kept[:, 0]) / crop_width,
                            (kept[:, 3] - kept[:, 1]) / crop_height,
                        ],
                        dim=1,
                    )
                    crop_boxes.append(normalized)
                    crop_classes.append(cls[source_targets][keep].reshape(-1, 1))
                    crop_batch_idx.append(torch.full((int(keep.sum()),), crop_index, dtype=torch.long, device=batch_idx.device))

    if crop_boxes:
        boxes = torch.cat(crop_boxes, dim=0)
        classes = torch.cat(crop_classes, dim=0)
        target_batch_idx = torch.cat(crop_batch_idx, dim=0)
    else:
        boxes = bboxes.new_empty((0, 4))
        classes = cls.reshape(-1, 1)[:0]
        target_batch_idx = batch_idx.new_empty((0,), dtype=torch.long)
    is_identity = len(metadata) == image.shape[0] and all(
        view.offset == (0, 0)
        and view.scale == (1.0, 1.0)
        and view.orig_shape == view.crop_shape == (height, width)
        for view in metadata
    )
    return {
        "img_cbsc": torch.stack(crop_images, dim=0),
        "bboxes_cbsc": boxes,
        "cls_cbsc": classes,
        "batch_idx_cbsc": target_batch_idx,
        "cbsc_views": metadata,
        "cbsc_source_indices": torch.tensor(crop_source_indices, device=image.device, dtype=torch.long),
        "cbsc_boundary": crop_boundary,
        "cbsc_is_identity": is_identity,
    }


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU for candidate fusion."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-9)


def fuse_view_candidates(
    candidates: torch.Tensor,
    source: torch.Tensor | list[int],
    match_iou: float = 0.3,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Fuse same-class candidates from different views using confidence-weighted boxes."""
    if candidates.ndim != 2 or candidates.shape[1] < 6:
        raise ValueError("candidates must have shape (N, 6+) in xyxy, confidence, class format")
    source = torch.as_tensor(source, device=candidates.device).reshape(-1)
    if len(source) != len(candidates):
        raise ValueError("source must contain one view index per candidate")
    if not 0 <= match_iou <= 1:
        raise ValueError("match_iou must be between 0 and 1")
    if not len(candidates):
        return candidates.clone(), {"before": 0, "after": 0, "merged_groups": 0}

    ious = _box_iou(candidates[:, :4], candidates[:, :4])
    parent = list(range(len(candidates)))

    def union(left: int, right: int) -> None:
        while parent[left] != left:
            left = parent[left]
        while parent[right] != right:
            right = parent[right]
        if left != right:
            parent[right] = left

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if source[left] == source[right] or candidates[left, 5] != candidates[right, 5]:
                continue
            if ious[left, right] >= match_iou:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        root = index
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        groups.setdefault(root, []).append(index)

    fused = []
    merged_groups = 0
    for group in groups.values():
        selected = candidates[group]
        if len(group) == 1:
            fused.append(selected[0].clone())
            continue
        weights = selected[:, 4].clamp_min(1e-6)
        row = selected[weights.argmax()].clone()
        row[:4] = (selected[:, :4] * weights[:, None]).sum(0) / weights.sum()
        row[4] = selected[:, 4].max()
        fused.append(row)
        merged_groups += 1
    fused = torch.stack(fused, dim=0)
    fused = fused[torch.argsort(fused[:, 4], descending=True)]
    return fused, {"before": int(len(candidates)), "after": int(len(fused)), "merged_groups": merged_groups}

