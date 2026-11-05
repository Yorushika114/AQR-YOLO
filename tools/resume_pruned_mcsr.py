# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Resume an interrupted pruned-MCSR fine-tune run without rebuilding the pruned structure.

Ultralytics 8.4.94 native resume cannot be used for structurally-pruned checkpoints:
saved ckpts store ``model=None`` and ``setup_model()`` rebuilds from ``weights.yaml``,
which is the *unpruned* yaml (2,877,957 params). This script instead re-injects the
pruned model exactly like ``finetune_pruned_mcsr.py`` and restores training state
(EMA / optimizer / scaler / best_fitness / start_epoch) via an ``on_pretrain_routine_end``
callback, so the interrupted run continues at its correct epoch and mosaic schedule.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.patches import torch_load  # noqa: E402

import tools.finetune_pruned_mcsr as fp  # noqa: E402  (reuse structure-preserving helpers)


@dataclass(frozen=True)
class ResumeConfig:
    """Paths/state for resuming one pruned-MCSR recovery run."""

    weights: Path  # pruned source (structure + pruning metadata)
    last: Path  # interrupted run's weights/last.pt
    data: Path = fp.DEFAULT_DATA
    epochs: int = 75
    close_mosaic: int = 10  # must match the interrupted run's protocol
    device: int = 0
    sa_tal: bool = True
    sa_size: float = 40.0
    mcsr: bool = True
    seed: int = 0
    backup_suffix: str = "_resume_backup"


def resolve_run_dir(last: Path) -> Path:
    """The run directory owning ``last`` (i.e. ``<run>/weights/last.pt`` -> ``<run>``)."""
    return last.resolve().parent.parent


def _load_ckpt(last: Path) -> dict[str, Any]:
    """Raw checkpoint dict; passes straight to trainer._load_checkpoint_state."""
    return torch_load(last, map_location=torch.device("cpu"))


def _build_resume_overrides(cfg: ResumeConfig, run_dir: Path) -> dict[str, Any]:
    """Same training config as the original recovery run, but pinned to the existing run dir."""
    base = fp.build_train_overrides(
        fp.FineTuneConfig(
            weights=cfg.weights,
            data=cfg.data,
            epochs=cfg.epochs,
            close_mosaic=cfg.close_mosaic,
            device=cfg.device,
            sa_tal=cfg.sa_tal,
            sa_size=cfg.sa_size,
            mcsr=cfg.mcsr,
            seed=cfg.seed,
            name=run_dir.name,  # cosmetic; save_dir override wins below
        )
    )
    base["resume"] = False  # avoid check_resume arg-replacement; state restored via callback
    base["save_dir"] = str(run_dir)  # get_save_dir honours args.save_dir -> exact same run dir
    base["exist_ok"] = True
    return base


def _make_resume_callback(ckpt: dict[str, Any], start_epoch: int, epochs: int, close_mosaic: int):
    """Return an on_pretrain_routine_end callback that restores optimizer/EMA/scaler/best/epoch."""

    def _restore(trainer) -> None:
        # continue the epoch loop where the dead run stopped
        trainer.start_epoch = start_epoch
        trainer.epoch = start_epoch - 1
        trainer._load_checkpoint_state(ckpt)  # optimizer / scaler / EMA / best_fitness
        trainer.scheduler.last_epoch = start_epoch - 1  # keep lr schedule position
        # the dead run had already closed mosaic for its final epochs; keep it closed
        if close_mosaic and start_epoch >= epochs - close_mosaic:
            trainer._close_dataloader_mosaic()
            trainer.train_loader.reset()
        from ultralytics.utils import LOGGER

        LOGGER.info(
            f"[resume] restored epoch={start_epoch}/{epochs} "
            f"best_fitness={trainer.best_fitness} ema_updates={trainer.ema.updates}"
        )

    return _restore


def _union_csv(csv_a: Path, csv_b: Path) -> pd.DataFrame | None:
    """Union of two results csv rows keyed by epoch (later segment wins a tie)."""
    import pandas as pd

    frames = [pd.read_csv(p) for p in (csv_a, csv_b) if p.exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def _backup_run_artifacts(run_dir: Path, suffix: str) -> None:
    """Accumulate history into the backup so repeated kill->resume segments never lose rows."""
    bdir = run_dir / f".{suffix}"
    bdir.mkdir(parents=True, exist_ok=True)
    bak_csv = bdir / "results.csv"
    live_csv = run_dir / "results.csv"
    from ultralytics.utils import LOGGER

    merged = _union_csv(bak_csv, live_csv)
    if merged is not None:
        merged.to_csv(bak_csv, index=False)
        LOGGER.info(f"[resume] accumulated {len(merged)} history rows -> .{suffix}/results.csv")
    for name in ("weights/best.pt", "weights/last.pt"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, bdir / src.name)
            LOGGER.info(f"[resume] backed up {name} -> .{suffix}/")


def _merge_results(run_dir: Path, backup_suffix: str) -> None:
    """Union the accumulated history with the resumed segment into the final csv."""
    from ultralytics.utils import LOGGER

    bak_csv = run_dir / f".{backup_suffix}" / "results.csv"
    csv = run_dir / "results.csv"
    if not bak_csv.exists():
        return
    merged = _union_csv(bak_csv, csv)
    if merged is None:
        return
    merged.to_csv(csv, index=False)
    LOGGER.info(f"[resume] final results.csv now has {len(merged)} rows (epochs {merged['epoch'].min()}-{merged['epoch'].max()})")


def resume_pruned_model(cfg: ResumeConfig) -> fp.FineTuneResult:
    """Continue an interrupted pruned-MCSR fine-tune and re-annotate pruning metadata."""
    run_dir = resolve_run_dir(cfg.last)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    _backup_run_artifacts(run_dir, cfg.backup_suffix)

    # 1. pruned structure + audit from the source checkpoint (authoritative mcsr flag)
    yolo, audit = fp.load_and_audit_pruned_model(cfg.weights)
    if cfg.mcsr != audit.mcsr:
        cfg = dataclasses.replace(cfg, mcsr=audit.mcsr)
    # 2. apply the interrupted run's EMA weights onto the SAME pruned structure
    ckpt = _load_ckpt(cfg.last)
    start_epoch = int(ckpt["epoch"]) + 1
    ema_state = ckpt["ema"].float().state_dict()
    yolo.model.load_state_dict(ema_state, strict=True)
    yolo.model.float()

    overrides = _build_resume_overrides(cfg, run_dir)
    trainer = fp.build_structure_preserving_trainer(yolo, overrides)
    if str(Path(trainer.save_dir).resolve()) != str(run_dir):
        raise RuntimeError(
            f"resume would target {trainer.save_dir}, expected {run_dir}; refusing to scatter outputs"
        )
    trainer.add_callback(
        "on_pretrain_routine_end",
        _make_resume_callback(ckpt, start_epoch, trainer.epochs, int(trainer.args.close_mosaic)),
    )
    trainer.train()

    best = Path(trainer.best).resolve() if Path(trainer.best).is_file() else None
    last = Path(trainer.last).resolve() if Path(trainer.last).is_file() else None
    selected = best or last
    if selected is None:
        raise FileNotFoundError(f"resume produced no best.pt/last.pt under {trainer.save_dir}")

    for checkpoint in (best, last):
        if checkpoint is not None:
            fp.verify_finetuned_checkpoint(checkpoint, audit.after_params, audit.mcsr)
            fp.annotate_finetuned_checkpoint(checkpoint, audit, _cfg_to_finetune(cfg))
            fp.verify_finetuned_checkpoint(checkpoint, audit.after_params, audit.mcsr)

    _merge_results(run_dir, cfg.backup_suffix)

    return fp.FineTuneResult(
        source=audit.source,
        save_dir=Path(trainer.save_dir).resolve(),
        best=best,
        last=last,
        selected=selected,
        parameters_before=audit.after_params,
        parameters_after=fp.verify_finetuned_checkpoint(selected, audit.after_params, audit.mcsr),
    )


def _cfg_to_finetune(cfg: ResumeConfig) -> fp.FineTuneConfig:
    """Reuse the finetune metadata annotation with the resume's hyper-parameters."""
    return fp.FineTuneConfig(
        weights=cfg.weights,
        data=cfg.data,
        epochs=cfg.epochs,
        close_mosaic=cfg.close_mosaic,
        device=cfg.device,
        sa_tal=cfg.sa_tal,
        sa_size=cfg.sa_size,
        mcsr=cfg.mcsr,
        seed=cfg.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser mirroring finetune_pruned_mcsr.py plus the interrupted last.pt."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True, help="pruned source checkpoint (structure + metadata)")
    p.add_argument("--last", type=Path, required=True, help="interrupted run's weights/last.pt")
    p.add_argument("--data", type=Path, default=fp.DEFAULT_DATA, help="dataset YAML")
    p.add_argument("--epochs", type=int, default=75, help="total epochs of the original run (continue to this)")
    p.add_argument("--close-mosaic", type=int, default=10, help="mosaic-free tail epochs (must match original run)")
    p.add_argument("--device", default=0, help="single device only")
    p.add_argument("--sa-tal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sa-size", type=float, default=40.0)
    p.add_argument("--mcsr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backup-suffix", default="_resume_backup", help="suffix for pre-resume backups")
    return p


def main() -> None:
    """Run the resume and print structural preservation evidence."""
    args = build_parser().parse_args()
    result = resume_pruned_model(
        ResumeConfig(
            weights=args.weights,
            last=args.last,
            data=args.data,
            epochs=args.epochs,
            close_mosaic=args.close_mosaic,
            device=args.device,
            sa_tal=args.sa_tal,
            sa_size=args.sa_size,
            mcsr=args.mcsr,
            seed=args.seed,
            backup_suffix=args.backup_suffix,
        )
    )
    print(f"Source: {result.source}")
    print(f"Parameters: {result.parameters_before:,} -> {result.parameters_after:,} (structure preserved)")
    print(f"Run directory: {result.save_dir}")
    print(f"Selected weights: {result.selected}")


if __name__ == "__main__":
    main()
