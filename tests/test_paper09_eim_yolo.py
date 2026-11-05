"""Regression tests for the EIM-YOLO paper reproduction entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

import paper09_eim_yolo as eim


def test_inner_siou_is_symmetric_and_penalizes_center_offset() -> None:
    """Equal left/right offsets must receive the same non-negative geometric penalty."""
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    right_shift = torch.tensor([[0.5, 0.0, 2.5, 2.0]])
    left_shift = torch.tensor([[-0.5, 0.0, 1.5, 2.0]])

    right_score = eim.inner_siou(target, right_shift)
    left_score = eim.inner_siou(target, left_shift)

    assert torch.allclose(right_score, left_score, atol=1e-6)
    assert torch.all(right_score < 1.0)


def test_rfa_conv_accepts_a_single_image_in_training_mode() -> None:
    """The global attention branch must not apply batch normalization to a 1x1 single-image tensor."""
    layer = eim.RFAConv(8, 8, k=3, s=2).train()

    output = layer(torch.randn(1, 8, 16, 16))

    assert output.shape == (1, 8, 8, 8)
    assert torch.isfinite(output).all()


def test_committed_yaml_is_a_valid_full_four_head_eim_model() -> None:
    """The checked-in configuration must match the paper's four-head EIM model."""
    config = yaml.safe_load((Path(__file__).parents[1] / "paper09_eim_52sbzxqm.yaml").read_text(encoding="utf-8"))

    assert config["head"][-1][0] == [25, 16, 19, 22]


def test_training_entry_does_not_disable_empty_gt_or_hide_nonfinite_losses(monkeypatch) -> None:
    """Background batches and non-finite losses must be handled by the real criterion, not masked."""
    calls: list[str] = []
    train_options: dict[str, object] = {}

    class _Model:
        class _Inner:
            @staticmethod
            def info(**kwargs):
                return 1, 1, 1, 1.0

        model = _Inner()

        @staticmethod
        def add_callback(*args, **kwargs):
            calls.append("callback")

        @staticmethod
        def train(**kwargs):
            train_options.update(kwargs)
            calls.append("train")

    monkeypatch.setattr(eim, "_patch_inner_siou", lambda: calls.append("inner_siou"))
    monkeypatch.setattr(eim, "_patch_empty_gt_loss", lambda: calls.append("empty_gt"), raising=False)
    monkeypatch.setattr(eim, "_patch_val_loss_nan_guard", lambda: calls.append("nan_guard"), raising=False)
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(
        eim, "copy_training_result", lambda project, name: copied.append((project, name)), raising=False
    )
    model_options: dict[str, object] = {}

    def _build_model(**kwargs):
        model_options.update(kwargs)
        return _Model()

    monkeypatch.setattr(eim, "build_model", _build_model)
    monkeypatch.setattr(sys, "argv", ["paper09_eim_yolo.py", "--epochs", "1", "--device", "0"])

    eim.main()

    assert calls == ["inner_siou", "train"]
    assert model_options["fourd"] is True
    assert train_options["project"] == eim.TRAIN_RESULTS_DIR
    assert copied == [(eim.TRAIN_RESULTS_DIR, "paper09_eim_fixed_75ep")]
