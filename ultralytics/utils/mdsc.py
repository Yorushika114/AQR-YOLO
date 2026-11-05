# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Single-view morphology-density score calibration for detection candidates."""

from __future__ import annotations

import torch


def mdsc_config_from_args(args) -> dict[str, float | int | bool]:
    """Read MDSC options from a Ultralytics configuration namespace."""
    return {
        "shape_weight": getattr(args, "mdsc_shape_weight", 0.05),
        "density_weight": getattr(args, "mdsc_density_weight", 0.03),
        "small_object_weight": getattr(args, "mdsc_small_object_weight", 0.05),
        "radius": getattr(args, "mdsc_radius", 64),
        "topk": getattr(args, "mdsc_topk", 8),
    }


def mdsc_calibrate_candidates(
    candidates: torch.Tensor,
    image_shape: tuple[int, int] | list[int],
    enabled: bool = True,
    shape_weight: float = 0.05,
    density_weight: float = 0.03,
    small_object_weight: float = 0.05,
    radius: float = 64,
    topk: int = 8,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Calibrate candidate confidence with inexpensive geometry and local density cues.

    The operation is candidate-only: it does not crop images, call the model, or use labels. The logit-space update is
    deliberately small so that MDSC changes ranking without replacing the detector's learned confidence.

    Args:
        candidates (torch.Tensor): Candidate rows in ``xyxy, confidence, class`` format.
        image_shape (tuple[int, int] | list[int]): Image height and width.
        enabled (bool): Whether to apply calibration.
        shape_weight (float): Weight for the long-side/short-side morphology cue.
        density_weight (float): Weight for local candidate density support.
        small_object_weight (float): Weight for relative small-object support.
        radius (float): Local center-distance radius in pixels.
        topk (int): Maximum neighbors counted for each candidate.

    Returns:
        (tuple[torch.Tensor, dict]): Calibrated candidates and lightweight metrics.
    """
    if candidates.ndim != 2 or candidates.shape[1] < 6:
        raise ValueError("candidates must have shape (N, 6+) in xyxy, confidence, class format")
    if radius < 0 or topk < 1:
        raise ValueError("radius must be non-negative and topk must be positive")
    metrics: dict[str, float | int | bool] = {
        "enabled": bool(enabled),
        "before": int(len(candidates)),
        "after": int(len(candidates)),
        "mean_abs_delta": 0.0,
    }
    if not enabled or not len(candidates):
        return candidates.clone(), metrics

    boxes = candidates[:, :4]
    wh = (boxes[:, 2:4] - boxes[:, :2]).clamp_min(1e-6)
    areas = wh.prod(dim=1)
    aspect = wh.max(dim=1).values / wh.min(dim=1).values
    median_area = areas.detach().median().clamp_min(1e-6)

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distances = torch.cdist(centers, centers)
    neighbor_limit = min(int(topk), max(len(candidates) - 1, 0))
    density = candidates.new_zeros(len(candidates))
    if neighbor_limit:
        nearest = torch.argsort(distances, dim=1)[:, 1 : neighbor_limit + 1]
        density = (distances.gather(1, nearest) <= radius).float().mean(dim=1)

    # Center all cues so the update remains a rank/calibration adjustment rather than a confidence replacement.
    shape_signal = ((aspect - 1) / (aspect + 1)).clamp(0, 1) - 0.25
    small_signal = (torch.log(median_area) - torch.log(areas))
    small_signal = small_signal.tanh() * 0.5
    density_signal = density - density.mean() if len(candidates) > 1 else density * 0
    adjustment = shape_weight * shape_signal + small_object_weight * small_signal + density_weight * density_signal

    confidence = candidates[:, 4].clamp(1e-4, 1 - 1e-4)
    logits = torch.log(confidence) - torch.log1p(-confidence)
    calibrated = torch.sigmoid(logits + adjustment).to(candidates.dtype)
    output = candidates.clone()
    output[:, 4] = calibrated
    metrics["mean_abs_delta"] = float((calibrated - candidates[:, 4]).abs().mean().detach())
    metrics["mean_density"] = float(density.mean().detach())
    metrics["image_area"] = int(max(int(image_shape[0]), 1) * max(int(image_shape[1]), 1))
    return output, metrics

