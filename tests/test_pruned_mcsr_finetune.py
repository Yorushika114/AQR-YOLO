from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from tools.finetune_pruned_mcsr import (
    DEFAULT_DATA,
    DEFAULT_WEIGHTS,
    FineTuneConfig,
    annotate_finetuned_checkpoint,
    build_parser,
    build_structure_preserving_trainer,
    build_train_overrides,
    load_and_audit_pruned_model,
    verify_finetuned_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml"
PRETRAINED = ROOT / "yolo11n.pt"


def save_marked_mcsr_checkpoint(tmp_path: Path, name: str = "marked_pruned.pt") -> tuple[Path, int]:
    """Save an MCSR checkpoint with internally consistent pruning metadata."""
    yolo = YOLO(MODEL_YAML).load(PRETRAINED)
    current_params = sum(parameter.numel() for parameter in yolo.model.parameters())
    yolo.ckpt = {
        "pruning": {
            "method": "test pruning",
            "before_params": current_params + 8,
            "after_params": current_params,
        }
    }
    path = tmp_path / name
    yolo.save(path)
    return path, current_params


def test_load_and_audit_accepts_consistent_pruned_checkpoint(tmp_path):
    path, expected_params = save_marked_mcsr_checkpoint(tmp_path)

    yolo, audit = load_and_audit_pruned_model(path)

    assert audit.source == path.resolve()
    assert audit.current_params == audit.after_params == expected_params
    assert audit.before_params == expected_params + 8
    assert audit.class_count == yolo.model.model[-1].nc
    assert yolo.model.model[-1].router is not None


def test_load_and_audit_rejects_checkpoint_without_pruning_metadata():
    with pytest.raises(ValueError, match="pruning metadata"):
        load_and_audit_pruned_model(PRETRAINED)


def test_build_train_overrides_uses_recovery_defaults(tmp_path):
    config = FineTuneConfig(weights=tmp_path / "pruned.pt", data=tmp_path / "data.yaml")

    overrides = build_train_overrides(config)

    assert overrides["model"] == str(config.weights)
    assert overrides["data"] == str(config.data)
    assert overrides["optimizer"] == "AdamW"
    assert overrides["lr0"] == 0.001
    assert overrides["mcsr"] is True
    assert overrides["sa_tal"] is True
    assert overrides["sa_size"] == 40.0
    assert overrides["pretrained"] is False
    assert overrides["resume"] is False


@pytest.mark.parametrize("device", ([0, 1], "0,1", (0, 1)))
def test_build_train_overrides_rejects_multi_device_training(tmp_path, device):
    config = FineTuneConfig(weights=tmp_path / "pruned.pt", data=tmp_path / "data.yaml", device=device)

    with pytest.raises(ValueError, match="single-device"):
        build_train_overrides(config)


def test_detection_trainer_setup_preserves_injected_module(tmp_path):
    path, _ = save_marked_mcsr_checkpoint(tmp_path)
    yolo, _ = load_and_audit_pruned_model(path)
    trainer = object.__new__(DetectionTrainer)
    trainer.model = yolo.model

    result = trainer.setup_model()

    assert result is None
    assert trainer.model is yolo.model
    assert isinstance(trainer.model, nn.Module)


def test_build_structure_preserving_trainer_injects_exact_model(tmp_path):
    path, _ = save_marked_mcsr_checkpoint(tmp_path)
    yolo, _ = load_and_audit_pruned_model(path)
    overrides = build_train_overrides(FineTuneConfig(weights=path, data=tmp_path / "data.yaml"))

    class RecordingTrainer:
        def __init__(self, overrides, _callbacks):
            self.overrides = overrides
            self.callbacks = _callbacks
            self.model = "would-be-rebuilt-model"
            self.data = {"nc": yolo.model.model[-1].nc}

    trainer = build_structure_preserving_trainer(yolo, overrides, trainer_cls=RecordingTrainer)

    assert trainer.model is yolo.model
    assert trainer.overrides is overrides
    assert trainer.callbacks is yolo.callbacks


def test_build_structure_preserving_trainer_rejects_dataset_class_mismatch(tmp_path):
    path, _ = save_marked_mcsr_checkpoint(tmp_path)
    yolo, _ = load_and_audit_pruned_model(path)
    overrides = build_train_overrides(FineTuneConfig(weights=path, data=tmp_path / "data.yaml"))

    class MismatchedTrainer:
        def __init__(self, overrides, _callbacks):
            self.data = {"nc": yolo.model.model[-1].nc + 1}
            self.model = "would-be-rebuilt-model"

    with pytest.raises(ValueError, match="class-count mismatch"):
        build_structure_preserving_trainer(yolo, overrides, trainer_cls=MismatchedTrainer)


def test_verify_finetuned_checkpoint_accepts_unchanged_structure(tmp_path):
    path, expected_params = save_marked_mcsr_checkpoint(tmp_path)

    actual_params = verify_finetuned_checkpoint(path, expected_params)

    assert actual_params == expected_params


def test_verify_finetuned_checkpoint_rejects_parameter_restoration(tmp_path):
    path, expected_params = save_marked_mcsr_checkpoint(tmp_path)

    with pytest.raises(RuntimeError, match="parameter count changed"):
        verify_finetuned_checkpoint(path, expected_params - 1)


def test_verify_finetuned_checkpoint_rejects_missing_mcsr_router():
    stock = YOLO(PRETRAINED)
    stock_params = sum(parameter.numel() for parameter in stock.model.parameters())

    with pytest.raises(RuntimeError, match="MCSR router"):
        verify_finetuned_checkpoint(PRETRAINED, stock_params)


def test_annotate_finetuned_checkpoint_preserves_model_and_pruning_metadata(tmp_path):
    path, expected_params = save_marked_mcsr_checkpoint(tmp_path)
    _, audit = load_and_audit_pruned_model(path)
    config = FineTuneConfig(weights=path, data=tmp_path / "data.yaml", sa_tal=True, sa_size=48.0)

    annotate_finetuned_checkpoint(path, audit, config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = YOLO(path)

    assert checkpoint["pruning"]["method"] == "test pruning"
    assert checkpoint["pruning_finetune"]["source"] == str(path.resolve())
    assert checkpoint["pruning_finetune"]["parameters_before_finetune"] == expected_params
    assert checkpoint["pruning_finetune"]["parameters_after_finetune"] == expected_params
    assert checkpoint["pruning_finetune"]["sa_tal"] is True
    assert checkpoint["pruning_finetune"]["sa_size"] == 48.0
    assert sum(parameter.numel() for parameter in restored.model.parameters()) == expected_params
    assert restored.model.model[-1].router is not None


def test_finetune_defaults_are_editable_local_paths():
    config = FineTuneConfig()

    assert config.weights == DEFAULT_WEIGHTS
    assert config.data == DEFAULT_DATA
    assert config.weights == ROOT / "runs/prune/mcsr_scale_pruned.pt"
    assert config.data == Path(r"D:\浏览器下载位置\detection_yolo_v2\data.yaml")


def test_cli_exposes_paths_recovery_settings_and_boolean_sa_tal():
    parser = build_parser()
    help_text = parser.format_help()

    assert "--weights" in help_text
    assert "--data" in help_text
    assert "--epochs" in help_text
    assert "--sa-tal" in help_text
    assert "--no-sa-tal" in help_text

    disabled = parser.parse_args(["--no-sa-tal"])
    assert disabled.sa_tal is False


def test_direct_script_execution_reaches_checkpoint_audit(tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: {0: defect}\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    missing_weights = tmp_path / "missing.pt"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/finetune_pruned_mcsr.py"),
            "--weights",
            str(missing_weights),
            "--data",
            str(data),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pruned weights not found" in result.stderr
    assert "NameError" not in result.stderr


def test_direct_script_uses_local_ultralytics_for_custom_mcsr_checkpoint(tmp_path):
    weights, _ = save_marked_mcsr_checkpoint(tmp_path)
    (tmp_path / "images/train").mkdir(parents=True)
    (tmp_path / "images/val").mkdir(parents=True)
    data = tmp_path / "data.yaml"
    data.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\nnames: {{0: defect}}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/finetune_pruned_mcsr.py"),
            "--weights",
            str(weights),
            "--data",
            str(data),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "class-count mismatch" in result.stderr
    assert "Can't get attribute 'AdaptiveDefectDetect'" not in result.stderr
