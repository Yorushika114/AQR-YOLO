# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Differentiable DIDC consistency and target-response losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _anchor_centers(preds: dict[str, torch.Tensor], strides: torch.Tensor) -> torch.Tensor:
    """Return concatenated feature-grid centers in input-image pixels."""
    centers = []
    for feature, stride in zip(preds["feats"], strides):
        height, width = feature.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5,
            torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5,
            indexing="ij",
        )
        centers.append(torch.stack((x.reshape(-1), y.reshape(-1)), dim=1) * stride.to(feature.dtype))
    return torch.cat(centers, dim=0)


def _region_response(
    preds: dict[str, torch.Tensor],
    boxes: torch.Tensor,
    classes: torch.Tensor,
    valid: torch.Tensor,
    image_size: tuple[int, int],
    strides: torch.Tensor,
) -> torch.Tensor:
    """Measure differentiable class response around one normalized box per image."""
    scores = preds["scores"].sigmoid()
    centers = _anchor_centers(preds, strides).to(scores.device)
    height, width = image_size
    response = scores.sum(dim=(1, 2)) * 0
    for batch_index in range(scores.shape[0]):
        if not bool(valid[batch_index]):
            continue
        x_center, y_center, box_width, box_height = boxes[batch_index]
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        inside = (centers[:, 0] >= x1) & (centers[:, 0] <= x2) & (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
        if not inside.any():
            distance = ((centers - torch.stack(((x1 + x2) / 2, (y1 + y2) / 2))) ** 2).sum(dim=1)
            inside[distance.argmin()] = True
        class_index = classes[batch_index].long().clamp(0, scores.shape[1] - 1)
        response[batch_index] = scores[batch_index, class_index, inside].mean()
    return response


def region_response_for_targets(
    preds: dict[str, torch.Tensor],
    boxes: torch.Tensor,
    classes: torch.Tensor,
    batch_indices: torch.Tensor,
    image_size: tuple[int, int],
    strides: torch.Tensor,
) -> torch.Tensor:
    """Measure one differentiable class response for every target row."""
    scores = preds["scores"].sigmoid()
    centers = _anchor_centers(preds, strides).to(scores.device)
    height, width = image_size
    responses = []
    for box, class_index, batch_index in zip(boxes, classes.reshape(-1), batch_indices.reshape(-1)):
        x_center, y_center, box_width, box_height = box
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        inside = (centers[:, 0] >= x1) & (centers[:, 0] <= x2) & (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
        if not inside.any():
            distance = ((centers - torch.stack(((x1 + x2) / 2, (y1 + y2) / 2))) ** 2).sum(dim=1)
            inside[distance.argmin()] = True
        class_index = class_index.long().clamp(0, scores.shape[1] - 1)
        responses.append(scores[batch_index.long(), class_index, inside].mean())
    return torch.stack(responses) if responses else scores.sum().reshape(1)[:0]


def didc_auxiliary_losses(
    original: dict[str, torch.Tensor],
    erased: dict[str, torch.Tensor],
    inserted: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DIDC outside-region stability and target-response direction losses.

    Args:
        original (dict[str, torch.Tensor]): Original detector predictions.
        erased (dict[str, torch.Tensor]): Predictions for the erased view.
        inserted (dict[str, torch.Tensor]): Predictions for the inserted view.
        batch (dict[str, torch.Tensor]): Batch containing DIDC masks and target metadata.
        model (torch.nn.Module): Detection model providing head strides and DIDC margin.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): ``(outside_loss, delta_loss)``.
    """
    if "edit_mask" not in batch:
        zero = original["scores"].sum() * 0 if isinstance(original, dict) else original.sum() * 0
        return zero, zero
    edit_mask = batch["edit_mask"].float().unsqueeze(1)
    outside_loss = original["scores"].sum() * 0
    offset = 0
    for original_feature, erased_feature, inserted_feature in zip(
        original["feats"], erased["feats"], inserted["feats"]
    ):
        height, width = original_feature.shape[-2:]
        count = height * width
        outside = ~F.interpolate(edit_mask, size=(height, width), mode="nearest").flatten(1).bool()
        original_scores = original["scores"][:, :, offset : offset + count]
        erased_scores = erased["scores"][:, :, offset : offset + count]
        inserted_scores = inserted["scores"][:, :, offset : offset + count]
        difference = F.smooth_l1_loss(erased_scores, original_scores, reduction="none")
        difference = difference + F.smooth_l1_loss(inserted_scores, original_scores, reduction="none")
        outside_loss = outside_loss + (difference * outside.unsqueeze(1)).sum() / outside.sum().clamp_min(1)
        offset += count
    outside_loss = outside_loss / max(len(original["feats"]), 1)

    strides = model.model[-1].stride
    image_size = tuple(batch["img"].shape[-2:])
    erase_box = batch["didc_erase_box"]
    insert_box = batch["didc_insert_box"]
    erase_cls = batch["didc_erase_cls"]
    insert_cls = batch["didc_insert_cls"]
    erase_valid = batch["didc_erase_valid"]
    insert_valid = batch["didc_insert_valid"]
    original_erase = _region_response(original, erase_box, erase_cls, erase_valid, image_size, strides)
    erased_response = _region_response(erased, erase_box, erase_cls, erase_valid, image_size, strides)
    original_insert = _region_response(original, insert_box, insert_cls, insert_valid, image_size, strides)
    inserted_response = _region_response(inserted, insert_box, insert_cls, insert_valid, image_size, strides)
    margin = max(float(getattr(getattr(model, "args", None), "didc_delta_margin", 0.05)), 0.0)
    erase_violation = F.relu(erased_response - original_erase + margin)
    insert_violation = F.relu(original_insert - inserted_response + margin)
    delta_valid = erase_valid.float() + insert_valid.float()
    delta_loss = (erase_violation * erase_valid.float() + insert_violation * insert_valid.float()).sum()
    delta_loss = delta_loss / delta_valid.sum().clamp_min(1)
    return outside_loss, delta_loss


def didc_single_view_auxiliary_losses(
    original: dict[str, torch.Tensor],
    branch: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    branch_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DIDC auxiliary losses for one selected erase or insert view.

    Args:
        original (dict[str, torch.Tensor]): Predictions for the original image.
        branch (dict[str, torch.Tensor]): Predictions for the selected counterfactual view.
        batch (dict[str, torch.Tensor]): Batch containing DIDC masks and target metadata.
        model (torch.nn.Module): Detection model providing head strides and DIDC margin.
        branch_name (str): Either ``"erase"`` or ``"insert"``.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): Outside-region and directional losses.
    """
    if branch_name not in {"erase", "insert"}:
        raise ValueError(f"branch_name must be 'erase' or 'insert', got {branch_name!r}")
    if "edit_mask" not in batch:
        zero = original.sum() * 0 if isinstance(original, torch.Tensor) else original["scores"].sum() * 0
        return zero, zero

    mask_key = f"didc_{branch_name}_mask"
    edit_mask = batch.get(mask_key, batch["edit_mask"]).float().unsqueeze(1)
    outside_loss = original["scores"].sum() * 0
    offset = 0
    for original_feature, branch_feature in zip(original["feats"], branch["feats"]):
        height, width = original_feature.shape[-2:]
        count = height * width
        outside = ~F.interpolate(edit_mask, size=(height, width), mode="nearest").flatten(1).bool()
        original_scores = original["scores"][:, :, offset : offset + count]
        branch_scores = branch["scores"][:, :, offset : offset + count]
        difference = F.smooth_l1_loss(branch_scores, original_scores, reduction="none")
        outside_loss = outside_loss + (difference * outside.unsqueeze(1)).sum() / outside.sum().clamp_min(1)
        offset += count
    outside_loss = outside_loss / max(len(original["feats"]), 1)

    strides = model.model[-1].stride
    image_size = tuple(batch["img"].shape[-2:])
    box = batch[f"didc_{branch_name}_box"]
    classes = batch[f"didc_{branch_name}_cls"]
    valid = batch[f"didc_{branch_name}_valid"]
    original_response = _region_response(original, box, classes, valid, image_size, strides)
    branch_response = _region_response(branch, box, classes, valid, image_size, strides)
    margin = max(float(getattr(getattr(model, "args", None), "didc_delta_margin", 0.05)), 0.0)
    if branch_name == "erase":
        violation = F.relu(branch_response - original_response + margin)
    else:
        violation = F.relu(original_response - branch_response + margin)
    delta_loss = (violation * valid.float()).sum() / valid.float().sum().clamp_min(1)
    return outside_loss, delta_loss

