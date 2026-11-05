# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Deterministic counterfactual views for defect detection training."""

from __future__ import annotations

import random

import torch


def _box_to_pixels(box: torch.Tensor, height: int, width: int) -> tuple[int, int, int, int]:
    """Convert a normalized xywh box to a clipped integer xyxy box."""
    cx, cy, bw, bh = box.tolist()
    x1 = max(0, min(width - 1, int(round((cx - bw / 2) * width))))
    y1 = max(0, min(height - 1, int(round((cy - bh / 2) * height))))
    x2 = max(x1 + 1, min(width, int(round((cx + bw / 2) * width))))
    y2 = max(y1 + 1, min(height, int(round((cy + bh / 2) * height))))
    return x1, y1, x2, y2


def _pixel_box_to_normalized(box: tuple[int, int, int, int], height: int, width: int) -> torch.Tensor:
    """Convert an integer xyxy box to normalized xywh coordinates."""
    x1, y1, x2, y2 = box
    return torch.tensor(
        [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height],
        dtype=torch.float32,
    )


def _has_overlap(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    """Return whether a pixel box overlaps any existing box."""
    x1, y1, x2, y2 = box
    return any(x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1 for bx1, by1, bx2, by2 in boxes)


def _find_insert_box(source_box: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]], height: int, width: int):
    """Find a deterministic non-overlapping destination for an inserted patch."""
    _, _, x2, y2 = source_box
    patch_width = x2 - source_box[0]
    patch_height = y2 - source_box[1]
    max_x = max(width - patch_width, 0)
    max_y = max(height - patch_height, 0)
    candidates = [
        (0, 0, patch_width, patch_height),
        (max_x, 0, width, patch_height),
        (0, max_y, patch_width, height),
        (max_x, max_y, width, height),
    ]
    for candidate in candidates:
        if not _has_overlap(candidate, existing):
            return candidate
    return candidates[0]


def _fill_box(image: torch.Tensor, box: tuple[int, int, int, int]) -> None:
    """Fill a box with the mean color of its immediate surrounding pixels."""
    x1, y1, x2, y2 = box
    height, width = image.shape[-2:]
    samples = []
    if y1:
        samples.append(image[:, y1 - 1 : y1, x1:x2])
    if y2 < height:
        samples.append(image[:, y2 : y2 + 1, x1:x2])
    if x1:
        samples.append(image[:, y1:y2, x1 - 1 : x1])
    if x2 < width:
        samples.append(image[:, y1:y2, x2 : x2 + 1])
    if samples:
        values = torch.cat([sample.reshape(image.shape[0], -1) for sample in samples], dim=1)
        fill = values.float().mean(dim=1, keepdim=True).view(-1, 1, 1).to(image.dtype)
    else:
        fill = image.float().mean(dim=(-2, -1), keepdim=True).to(image.dtype)
    image[:, y1:y2, x1:x2] = fill


def build_counterfactual_views(
    image: torch.Tensor,
    bboxes: torch.Tensor,
    cls: torch.Tensor,
    batch_idx: torch.Tensor,
    seed: int = 0,
    erase_prob: float = 0.5,
    insert_prob: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Build deterministic erase and insert views from normalized detection labels.

    Args:
        image (torch.Tensor): Batch of images with shape ``(B, C, H, W)``.
        bboxes (torch.Tensor): Normalized ``xywh`` boxes with shape ``(N, 4)``.
        cls (torch.Tensor): Class labels with shape ``(N, 1)`` or ``(N,)``.
        batch_idx (torch.Tensor): Image index for each target.
        seed (int): Seed used to select one target per image.
        erase_prob (float): Probability of constructing the erase edit.
        insert_prob (float): Probability of constructing the insert edit.

    Returns:
        (dict[str, torch.Tensor]): Original-compatible branch images, labels, and edit masks.
    """
    if image.ndim != 4:
        raise ValueError(f"image must have shape (B, C, H, W), got {tuple(image.shape)}")
    if not 0 <= erase_prob <= 1 or not 0 <= insert_prob <= 1:
        raise ValueError("erase_prob and insert_prob must be between 0 and 1")

    cls = cls.reshape(-1, 1)
    bboxes = bboxes.reshape(-1, 4)
    batch_idx = batch_idx.reshape(-1).long()
    erase_image = image.clone()
    insert_image = image.clone()
    erase_mask = torch.zeros(image.shape[0], *image.shape[-2:], dtype=torch.bool, device=image.device)
    insert_mask = torch.zeros_like(erase_mask)
    erase_boxes = torch.zeros((image.shape[0], 4), dtype=bboxes.dtype, device=bboxes.device)
    insert_boxes = torch.zeros_like(erase_boxes)
    erase_classes = torch.zeros((image.shape[0], 1), dtype=cls.dtype, device=cls.device)
    insert_classes = torch.zeros_like(erase_classes)
    erase_valid = torch.zeros(image.shape[0], dtype=torch.bool, device=image.device)
    insert_valid = torch.zeros_like(erase_valid)
    erase_keep = torch.ones(bboxes.shape[0], dtype=torch.bool, device=bboxes.device)
    inserted_boxes = [bboxes]
    insert_cls = [cls]
    insert_batch_idx = [batch_idx]
    rng = random.Random(seed)

    for batch_number in range(image.shape[0]):
        target_indices = torch.where(batch_idx == batch_number)[0].tolist()
        if not target_indices:
            continue
        target_index = rng.choice(target_indices)
        pixel_boxes = [_box_to_pixels(bboxes[i].detach().cpu(), *image.shape[-2:]) for i in target_indices]
        source_box = _box_to_pixels(bboxes[target_index].detach().cpu(), *image.shape[-2:])
        x1, y1, x2, y2 = source_box

        if rng.random() < erase_prob:
            _fill_box(erase_image[batch_number], source_box)
            erase_mask[batch_number, y1:y2, x1:x2] = True
            erase_boxes[batch_number] = bboxes[target_index]
            erase_classes[batch_number] = cls[target_index]
            erase_valid[batch_number] = True
            erase_keep[target_index] = False

        if rng.random() < insert_prob:
            destination = _find_insert_box(source_box, pixel_boxes, image.shape[-2], image.shape[-1])
            dx1, dy1, dx2, dy2 = destination
            patch = image[batch_number, :, y1:y2, x1:x2].clone()
            insert_image[batch_number, :, dy1:dy2, dx1:dx2] = patch
            insert_mask[batch_number, dy1:dy2, dx1:dx2] = True
            new_box = _pixel_box_to_normalized(destination, *image.shape[-2:]).to(bboxes.device, bboxes.dtype)
            insert_boxes[batch_number] = new_box
            insert_classes[batch_number] = cls[target_index]
            insert_valid[batch_number] = True
            inserted_boxes.append(new_box.view(1, 4))
            insert_cls.append(cls[target_index : target_index + 1])
            insert_batch_idx.append(batch_idx[target_index : target_index + 1])

    inserted_bboxes = torch.cat(inserted_boxes, dim=0)
    inserted_classes = torch.cat(insert_cls, dim=0)
    inserted_batch_indices = torch.cat(insert_batch_idx, dim=0)
    order = torch.argsort(inserted_batch_indices)
    return {
        "img_erase": erase_image,
        "img_insert": insert_image,
        "bboxes_erase": bboxes[erase_keep],
        "cls_erase": cls[erase_keep],
        "batch_idx_erase": batch_idx[erase_keep],
        "bboxes_insert": inserted_bboxes[order],
        "cls_insert": inserted_classes[order],
        "batch_idx_insert": inserted_batch_indices[order],
        "edit_mask": erase_mask | insert_mask,
        "didc_erase_mask": erase_mask,
        "didc_insert_mask": insert_mask,
        "didc_erase_box": erase_boxes,
        "didc_insert_box": insert_boxes,
        "didc_erase_cls": erase_classes,
        "didc_insert_cls": insert_classes,
        "didc_erase_valid": erase_valid,
        "didc_insert_valid": insert_valid,
    }

