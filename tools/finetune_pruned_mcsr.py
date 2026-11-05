# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fine-tune a structurally pruned YOLO11 MCSR checkpoint without rebuilding it from YAML."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


DEFAULT_WEIGHTS = ROOT / "runs/prune/mcsr_scale_pruned.pt"
DEFAULT_DATA = Path(r"D:\浏览器下载位置\detection_yolo_v2\data.yaml")
DEFAULT_PROJECT = ROOT / "runs/prune_finetune"


@dataclass(frozen=True)
class PrunedModelAudit:
    """Verified structural facts read from a pruned MCSR checkpoint."""

    source: Path
    before_params: int
    after_params: int
    current_params: int
    class_count: int
    mcsr: bool = True  # whether the pruned architecture carries the MCSR router


@dataclass(frozen=True)
class FineTuneConfig:
    """Configuration for recovery fine-tuning of one pruned MCSR checkpoint."""

    weights: Path = DEFAULT_WEIGHTS
    data: Path = DEFAULT_DATA
    epochs: int = 20
    batch: int = 16
    workers: int = 4
    imgsz: int = 640
    device: str | int | list[int] | tuple[int, ...] = 0
    project: Path = DEFAULT_PROJECT
    name: str = "mcsr_recovery"
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    warmup_epochs: float = 1.0
    close_mosaic: int = 5
    sa_tal: bool = True
    sa_size: float = 40.0
    mcsr: bool = True  # whether the pruned structure carries the MCSR router
    seed: int = 0
    unpruned: bool = False  # treat --weights as an unpruned (plain) YOLO11 checkpoint


@dataclass(frozen=True)
class FineTuneResult:
    """Paths and structural evidence produced by recovery fine-tuning."""

    source: Path
    save_dir: Path
    best: Path | None
    last: Path | None
    selected: Path
    parameters_before: int
    parameters_after: int


def _validate_single_device(device: str | int | list[int] | tuple[int, ...]) -> None:
    """Reject DDP configurations because child processes cannot inherit the injected module."""
    if isinstance(device, (list, tuple)) or (isinstance(device, str) and "," in device):
        raise ValueError("pruned recovery fine-tuning currently supports single-device training only")


def build_train_overrides(config: FineTuneConfig) -> dict[str, Any]:
    """Translate the recovery configuration into Ultralytics trainer overrides."""
    _validate_single_device(config.device)
    return {
        "model": str(config.weights),
        "data": str(config.data),
        "task": "detect",
        "mode": "train",
        "epochs": config.epochs,
        "batch": config.batch,
        "workers": config.workers,
        "imgsz": config.imgsz,
        "device": config.device,
        "project": str(config.project),
        "name": config.name,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "warmup_epochs": config.warmup_epochs,
        "close_mosaic": config.close_mosaic,
        "seed": config.seed,
        "deterministic": True,
        "save": True,
        "resume": False,
        "pretrained": False,
        "val": True,
        "mcsr": config.mcsr,
        "sa_tal": config.sa_tal,
        "sa_size": config.sa_size,
    }


def build_structure_preserving_trainer(
    yolo: YOLO,
    overrides: dict[str, Any],
    trainer_cls: type = DetectionTrainer,
):
    """Create a trainer and inject the exact pruned module before setup_model runs."""
    trainer = trainer_cls(overrides=overrides, _callbacks=yolo.callbacks)
    model_nc = int(yolo.model.model[-1].nc)
    dataset_nc = getattr(trainer, "data", {}).get("nc")
    if dataset_nc is not None and int(dataset_nc) != model_nc:
        raise ValueError(
            f"pruned model and dataset class-count mismatch: model nc={model_nc}, dataset nc={dataset_nc}; "
            "prune a checkpoint that was already trained for this dataset"
        )
    trainer.model = yolo.model
    return trainer


def verify_finetuned_checkpoint(checkpoint: str | Path, expected_params: int, mcsr: bool = True) -> int:
    """Reload a fine-tuned checkpoint and prove that its pruned structure survived.

    With ``mcsr=True`` the router must be present; with ``mcsr=False`` the plain
    YOLO11 Detect head must carry no router.
    """
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fine-tuned checkpoint not found: {path}")
    restored = YOLO(path)
    actual_params = sum(parameter.numel() for parameter in restored.model.parameters())
    if actual_params != expected_params:
        raise RuntimeError(
            f"fine-tuned parameter count changed: expected {expected_params}, found {actual_params}; "
            "the model may have been rebuilt from YAML"
        )
    head = restored.model.model[-1]
    has_router = getattr(head, "router", None) is not None
    if has_router != mcsr:
        raise RuntimeError(
            f"fine-tuned checkpoint head/router mismatch: mcsr={mcsr} but router={'present' if has_router else 'absent'}"
        )
    return actual_params


def annotate_finetuned_checkpoint(
    checkpoint: str | Path,
    audit: PrunedModelAudit,
    config: FineTuneConfig,
) -> Path:
    """Attach source-pruning and recovery metadata without replacing trained tensors."""
    path = Path(checkpoint).expanduser().resolve()
    parameters_after = verify_finetuned_checkpoint(path, audit.after_params, audit.mcsr)
    source_checkpoint = torch_load(audit.source, map_location=torch.device("cpu"))
    target_checkpoint = torch_load(path, map_location=torch.device("cpu"))
    if not isinstance(source_checkpoint, dict) or not isinstance(target_checkpoint, dict):
        raise ValueError("Ultralytics checkpoints must be dictionaries")

    pruning_meta = source_checkpoint.get("pruning")
    if pruning_meta is None:
        # --unpruned source carried no pruning metadata; record an explicit "unpruned" audit
        pruning_meta = {
            "method": "none",
            "unpruned": True,
            "before_params": audit.after_params,
            "after_params": audit.after_params,
            "mcsr": audit.mcsr,
        }
    target_checkpoint["pruning"] = dict(pruning_meta)
    target_checkpoint["pruning_finetune"] = {
        "source": str(audit.source),
        "parameters_before_finetune": audit.after_params,
        "parameters_after_finetune": parameters_after,
        "structure_preserved": parameters_after == audit.after_params,
        "mcsr": audit.mcsr,
        "sa_tal": config.sa_tal,
        "sa_size": config.sa_size,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
    }
    torch.save(target_checkpoint, path)
    return path


def finetune_pruned_model(
    config: FineTuneConfig,
    trainer_cls: type = DetectionTrainer,
) -> FineTuneResult:
    """Fine-tune an injected pruned model, then verify and annotate best/last weights."""
    data = Path(config.data).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data}")

    yolo, audit = load_and_audit_pruned_model(config.weights, unpruned=config.unpruned)
    overrides = build_train_overrides(config)
    overrides["data"] = str(data)
    trainer = build_structure_preserving_trainer(yolo, overrides, trainer_cls=trainer_cls)
    trainer.train()

    best = Path(trainer.best).resolve() if Path(trainer.best).is_file() else None
    last = Path(trainer.last).resolve() if Path(trainer.last).is_file() else None
    selected = best or last
    if selected is None:
        raise FileNotFoundError(f"fine-tuning created no best.pt or last.pt under {trainer.save_dir}")

    for checkpoint in (best, last):
        if checkpoint is not None:
            verify_finetuned_checkpoint(checkpoint, audit.after_params, audit.mcsr)
            annotate_finetuned_checkpoint(checkpoint, audit, config)
            verify_finetuned_checkpoint(checkpoint, audit.after_params, audit.mcsr)

    return FineTuneResult(
        source=audit.source,
        save_dir=Path(trainer.save_dir).resolve(),
        best=best,
        last=last,
        selected=selected,
        parameters_before=audit.after_params,
        parameters_after=verify_finetuned_checkpoint(selected, audit.after_params, audit.mcsr),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser while keeping all defaults editable above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="structurally pruned MCSR .pt checkpoint")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO dataset YAML")
    parser.add_argument("--epochs", type=int, default=20, help="recovery fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=16, help="training batch size")
    parser.add_argument("--workers", type=int, default=4, help="data-loader worker processes")
    parser.add_argument("--imgsz", type=int, default=640, help="square training image size")
    parser.add_argument("--device", default=0, help="one device only, e.g. 0, cuda:0, or cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="training output root")
    parser.add_argument("--name", default="mcsr_recovery", help="training run name")
    parser.add_argument("--optimizer", default="AdamW", help="recovery optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial recovery learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="final learning-rate fraction")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="warmup epochs")
    parser.add_argument("--close-mosaic", type=int, default=5, help="epochs without mosaic at the end")
    parser.add_argument(
        "--sa-tal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable SA-TAL during recovery fine-tuning",
    )
    parser.add_argument("--sa-size", type=float, default=40.0, help="SA-TAL small-box diagonal threshold in pixels")
    parser.add_argument(
        "--mcsr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fine-tune the MCSR structure (requires router); --no-mcsr keeps the plain YOLO11 Detect head",
    )
    parser.add_argument(
        "--unpruned",
        action="store_true",
        help="treat --weights as an unpruned YOLO11 checkpoint: no pruning metadata required, before==after audit (MCSR structure still derived from the loaded head)",
    )
    parser.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    return parser


def parse_args(argv: list[str] | None = None) -> FineTuneConfig:
    """Parse command-line options into the immutable fine-tuning configuration."""
    args = build_parser().parse_args(argv)
    return FineTuneConfig(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        sa_tal=args.sa_tal,
        sa_size=args.sa_size,
        mcsr=args.mcsr,
        seed=args.seed,
        unpruned=args.unpruned,
    )


def main() -> None:
    """Run recovery fine-tuning and print the structure-preservation evidence."""
    result = finetune_pruned_model(parse_args())
    print(f"Source: {result.source}")
    print(f"Parameters: {result.parameters_before:,} -> {result.parameters_after:,} (structure preserved)")
    print(f"Run directory: {result.save_dir}")
    print(f"Selected weights: {result.selected}")


def load_and_audit_pruned_model(weights: str | Path, unpruned: bool = False) -> tuple[YOLO, PrunedModelAudit]:
    """Load a pruned MCSR checkpoint and reject metadata or structure mismatches.

    With ``unpruned=True`` the weights are treated as a plain (unpruned) YOLO11
    checkpoint: no pruning metadata is required, the audit records ``before == after``
    as the live parameter count, and ``mcsr=False``.
    """
    source = Path(weights).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"pruned weights not found: {source}")

    yolo = YOLO(source)

    if unpruned:
        # Unpruned control: before == after (no parameter reduction), but the MCSR
        # structure may still be present (unpruned_mcsr / unpruned_both) or absent
        # (unpruned_none / unpruned_satal). Derive mcsr from the loaded head.
        current_params = sum(parameter.numel() for parameter in yolo.model.parameters())
        head = yolo.model.model[-1]
        has_router = getattr(head, "router", None) is not None
        return yolo, PrunedModelAudit(
            source=source,
            before_params=current_params,
            after_params=current_params,
            current_params=current_params,
            class_count=int(head.nc),
            mcsr=has_router,
        )

    pruning = (yolo.ckpt or {}).get("pruning")
    if not isinstance(pruning, dict):
        raise ValueError(f"checkpoint has no pruning metadata: {source}")

    try:
        before_params = int(pruning["before_params"])
        after_params = int(pruning["after_params"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pruning metadata must contain integer before_params and after_params") from exc

    mcsr = bool(pruning.get("mcsr", True))

    current_params = sum(parameter.numel() for parameter in yolo.model.parameters())
    if before_params <= after_params:
        raise ValueError(
            f"invalid pruning metadata: before_params={before_params} must exceed after_params={after_params}"
        )
    if current_params != after_params:
        raise ValueError(
            f"pruned structure mismatch: checkpoint has {current_params} parameters, metadata expects {after_params}"
        )

    head = yolo.model.model[-1]
    has_router = getattr(head, "router", None) is not None
    if has_router != mcsr:
        raise ValueError(
            f"checkpoint structure/ metadata mismatch: pruning.mcsr={mcsr} but "
            f"router={'present' if has_router else 'absent'}"
        )

    return yolo, PrunedModelAudit(
        source=source,
        before_params=before_params,
        after_params=after_params,
        current_params=current_params,
        class_count=int(head.nc),
        mcsr=mcsr,
    )


if __name__ == "__main__":
    main()
