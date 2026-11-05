"""Regression tests for the active innovation switches and combined classification loss."""

import sys

import torch

import train
from ultralytics.utils.loss import FrequencyConditionedFocalLoss


def test_cli_without_flags_uses_the_documented_default_innovations(monkeypatch):
    """CLI defaults must not silently override train_model's documented experiment defaults."""
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = train.parse_args()

    assert (args.mcsr, args.shde, args.scol, args.fcfl) == (False, True, True, True)


def test_fcfl_accepts_scol_anchor_weight_for_small_positive_anchors():
    """FCFL must preserve SCOL's per-anchor emphasis when both switches are enabled."""
    logits = torch.zeros(1, 2, 3)
    targets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    classes = torch.tensor([0])
    anchor_weight = torch.tensor([[[2.0], [1.0]]])

    plain = FrequencyConditionedFocalLoss(num_classes=3, device=logits.device, beta=0.0)
    weighted = FrequencyConditionedFocalLoss(num_classes=3, device=logits.device, beta=0.0)

    plain_loss = plain(logits, targets, classes)
    weighted_loss = weighted(logits, targets, classes, anchor_weight=anchor_weight)

    assert weighted_loss > plain_loss
