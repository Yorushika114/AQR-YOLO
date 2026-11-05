# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Differentiable cross-view consistency losses for CBSC."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from ultralytics.data.cbsc import CropView, fuse_view_candidates, generate_overlapping_crops, map_boxes_to_global
from ultralytics.utils.didc import _anchor_centers, region_response_for_targets
from ultralytics.utils import nms
from ultralytics.utils.mcrd import mcrd_config_from_args


def _xywhn_to_xyxy(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert normalized xywh boxes to pixel xyxy boxes."""
    height, width = image_size
    result = torch.empty_like(boxes)
    result[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    result[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    result[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    result[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return result


def _xyxy_to_xywhn(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert pixel xyxy boxes to normalized xywh boxes."""
    height, width = image_size
    return torch.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / (2 * width),
            (boxes[:, 1] + boxes[:, 3]) / (2 * height),
            (boxes[:, 2] - boxes[:, 0]) / width,
            (boxes[:, 3] - boxes[:, 1]) / height,
        ],
        dim=1,
    )


def _class_distribution(
    preds: dict[str, torch.Tensor],
    boxes: torch.Tensor,
    batch_indices: torch.Tensor,
    image_size: tuple[int, int],
    strides: torch.Tensor,
) -> torch.Tensor:
    """Return average class distributions around target boxes."""
    logits = preds["scores"].log_softmax(dim=1)
    centers = _anchor_centers(preds, strides).to(logits.device)
    height, width = image_size
    distributions = []
    for box, batch_index in zip(boxes, batch_indices):
        x_center, y_center, box_width, box_height = box
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        inside = (centers[:, 0] >= x1) & (centers[:, 0] <= x2) & (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
        if not inside.any():
            distance = ((centers - torch.stack(((x1 + x2) / 2, (y1 + y2) / 2))) ** 2).sum(dim=1)
            inside[distance.argmin()] = True
        distributions.append(logits[batch_index.long(), :, inside].mean(dim=1))
    return torch.stack(distributions) if distributions else logits.sum().reshape(1)[:0]


def cbsc_consistency_loss(
    full_preds: dict[str, torch.Tensor],
    crop_preds: dict[str, torch.Tensor],
    batch: dict[str, object],
    model: torch.nn.Module,
) -> torch.Tensor:
    """Match full-image and crop-view class responses after global coordinate mapping.

    Args:
        full_preds (dict[str, torch.Tensor]): Predictions for the original image batch.
        crop_preds (dict[str, torch.Tensor]): Predictions for the CBSC crop batch.
        batch (dict[str, object]): Batch containing crop labels and metadata.
        model (torch.nn.Module): Detection model providing head strides.

    Returns:
        (torch.Tensor): Differentiable response and class-distribution consistency loss.
    """
    boxes = batch["bboxes_cbsc"]
    if not len(boxes):
        return full_preds["scores"].sum() * 0
    crop_indices = batch["batch_idx_cbsc"]
    source_indices = batch["cbsc_source_indices"][crop_indices]
    views = batch["cbsc_views"]
    crop_size = tuple(batch["img_cbsc"].shape[-2:])
    full_size = tuple(batch["img"].shape[-2:])
    crop_pixel_boxes = _xywhn_to_xyxy(boxes, crop_size)
    global_pixel_boxes = torch.stack(
        [map_boxes_to_global(crop_pixel_boxes[index : index + 1], views[int(crop_indices[index])])[0] for index in range(len(boxes))]
    )
    global_boxes = _xyxy_to_xywhn(global_pixel_boxes, full_size).clamp(0, 1)
    strides = model.model[-1].stride
    crop_response = region_response_for_targets(
        crop_preds, boxes, batch["cls_cbsc"], crop_indices, crop_size, strides
    )
    full_response = region_response_for_targets(
        full_preds, global_boxes, batch["cls_cbsc"], source_indices, full_size, strides
    )
    response_loss = F.smooth_l1_loss(crop_response, full_response, reduction="none")
    crop_distribution = _class_distribution(crop_preds, boxes, crop_indices, crop_size, strides)
    full_distribution = _class_distribution(full_preds, global_boxes, source_indices, full_size, strides)
    class_loss = F.kl_div(crop_distribution, full_distribution.detach().softmax(dim=1), reduction="none").sum(dim=1)
    boundary = torch.tensor(
        [1.5 if views[int(index)].is_boundary else 1.0 for index in crop_indices],
        device=response_loss.device,
        dtype=response_loss.dtype,
    )
    return ((response_loss + 0.25 * class_loss) * boundary).sum() / boundary.sum().clamp_min(1)


def _match_model_input_dtype(model, tensor: torch.Tensor) -> torch.Tensor:
    """Move a CBSC crop batch to the device and floating dtype expected by the model."""
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return tensor

    if parameter.is_floating_point():
        return tensor.to(device=parameter.device, dtype=parameter.dtype)
    return tensor.to(device=parameter.device)


def _limit_fusion_candidates(
    candidates: torch.Tensor, sources: torch.Tensor, max_candidates: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the highest-confidence candidates before quadratic cross-view fusion."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if len(candidates) <= max_candidates:
        return candidates, sources
    keep = torch.topk(candidates[:, 4], k=max_candidates, largest=True, sorted=True).indices
    return candidates[keep], sources[keep]


def postprocess_cbsc(model, preds, img: torch.Tensor, args, iou: float, relation_decoder=None):
    """Run full-image and crop-view inference, fusion, MCRD, and final NMS."""
    full_preds = nms.non_max_suppression(
        preds,
        args.conf,
        iou,
        args.classes,
        args.agnostic_nms,
        max_det=args.max_det,
        nc=0,
        end2end=False,
        rotated=False,
        mcrd=False,
    )
    height, width = img.shape[-2:]
    views = generate_overlapping_crops(
        (height, width),
        getattr(args, "cbsc_crop_size", 640),
        getattr(args, "cbsc_overlap", 0.25),
        getattr(args, "cbsc_max_crops", 16),
    )
    views = [view for view in views if view.offset != (0, 0) or view.crop_shape != (height, width)]
    cbsc_metrics = {"views": len(views), "before_fusion": 0, "after_fusion": 0, "fusion_pruned": 0}
    if not views:
        return full_preds, cbsc_metrics, {}

    crop_height, crop_width = (
        (getattr(args, "cbsc_crop_size", 640),) * 2
        if isinstance(getattr(args, "cbsc_crop_size", 640), int)
        else tuple(getattr(args, "cbsc_crop_size"))
    )
    crops = []
    batch_views: list[list[CropView]] = []
    for batch_index in range(img.shape[0]):
        image_views = []
        for view in views:
            left, top = view.offset
            actual_height, actual_width = view.crop_shape
            crop = img[batch_index : batch_index + 1, :, top : top + actual_height, left : left + actual_width]
            crops.append(torch.nn.functional.interpolate(crop, size=(crop_height, crop_width), mode="bilinear"))
            image_views.append(
                replace(
                    view,
                    scale=(actual_width / crop_width, actual_height / crop_height),
                    crop_shape=(crop_height, crop_width),
                )
            )
        batch_views.append(image_views)

    crop_input = _match_model_input_dtype(model, torch.cat(crops, dim=0))
    crop_preds = nms.non_max_suppression(
        model(crop_input),
        args.conf,
        iou,
        args.classes,
        args.agnostic_nms,
        max_det=args.max_det,
        nc=0,
        end2end=False,
        rotated=False,
        mcrd=False,
    )
    mcrd_config = mcrd_config_from_args(args, relation_decoder)
    mcrd_metrics = {}
    outputs = []
    for batch_index, full_candidate in enumerate(full_preds):
        candidate_parts = [full_candidate]
        source_parts = [torch.zeros(len(full_candidate), dtype=torch.long, device=img.device)]
        for view_index, view in enumerate(batch_views[batch_index]):
            crop_candidate = crop_preds[batch_index * len(views) + view_index]
            if len(crop_candidate):
                mapped = crop_candidate.clone()
                mapped[:, :4] = map_boxes_to_global(mapped[:, :4], view)
                candidate_parts.append(mapped)
                source_parts.append(torch.full((len(mapped),), view_index + 1, dtype=torch.long, device=img.device))
        candidates = torch.cat(candidate_parts, dim=0)
        sources = torch.cat(source_parts, dim=0)
        cbsc_metrics["before_fusion"] += int(len(candidates))
        fusion_limit = max(int(getattr(args, "cbsc_max_candidates", args.max_det)), 1)
        limited_candidates, limited_sources = _limit_fusion_candidates(candidates, sources, fusion_limit)
        cbsc_metrics["fusion_pruned"] += int(len(candidates) - len(limited_candidates))
        fused, fusion_metrics = fuse_view_candidates(
            limited_candidates, limited_sources, match_iou=getattr(args, "cbsc_match_iou", 0.3)
        )
        cbsc_metrics["after_fusion"] += fusion_metrics["after"]
        outputs.append(
            nms.nms_candidates(
                fused,
                iou_thres=iou,
                max_det=args.max_det,
                agnostic=args.agnostic_nms,
                mcrd=getattr(args, "mcrd", False),
                mcrd_config=mcrd_config,
                mcrd_metrics=mcrd_metrics,
            )
        )
    return outputs, cbsc_metrics, mcrd_metrics

