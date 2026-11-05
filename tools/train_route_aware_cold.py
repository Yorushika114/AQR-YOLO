# -*- coding: utf-8 -*-
"""冷启动训练"Route-Aware 剪枝后的 mcsr+sa-tal (both)"模型。

与 train_pruned_cold.py 同口径: 剪枝臂也走公平冷启动协议, 100ep / lr0=0.01 / warmup=1.0 /
close_mosaic=10 / cls_remap / randaugment / erasing=0.4 / batch16 / imgsz640 / seed0 /
deterministic / optimizer=auto。剪枝源不同的地方仅是 weights 指向新剪枝产物:

  runs/prune/route_aware_expC_pruned.pt (Route-Aware, 2,537,509 params, -11.83%,
  p3=0/p4=0.10/p5=0.30 关闭R调制, 层13与MCSR入口保护)

冷起点: initialize_weights 全随机 -> yolo.load('yolo11n.pt') 按名+形状迁移(剪伤层随机)。
结构保持: 依赖 finetune_pruned_mcsr.build_structure_preserving_trainer, 把剪枝结构直接
注入 trainer(不 rebuild from yaml)。训练后 verify checkpint 参数数 == 剪枝结构 after_params。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds, initialize_weights  # noqa: E402

from tools.finetune_pruned_mcsr import (  # noqa: E402
    FineTuneConfig,
    build_structure_preserving_trainer,
    build_train_overrides,
    load_and_audit_pruned_model,
    verify_finetuned_checkpoint,
)

ROUTE_PRUNED = ROOT / "runs/prune/route_aware_expC_pruned.pt"
PRETRAINED = "yolo11n.pt"
DATASET_YAML = Path(r"D:/浏览器下载位置/detection_yolo_v2/data.yaml")


def build_cold_route_model(seed: int = 0, mcsr: bool = True, weights: Path = ROUTE_PRUNED) -> tuple[YOLO, int, int]:
    """构建 Route-Aware 剪枝结构的冷起点模型, 返回 (yolo, expected_params, before_params)."""
    yolo, audit = load_and_audit_pruned_model(weights)
    expected = audit.after_params
    before = audit.before_params

    init_seeds(seed, deterministic=True)
    initialize_weights(yolo.model)  # 全网络随机重置(冷起点)
    yolo.load(PRETRAINED)  # 按名+形状迁移 yolo11n.pt(匹配层迁移, 剪伤层随机)

    actual = sum(p.numel() for p in yolo.model.parameters())
    if actual != expected:
        raise RuntimeError(
            f"cold route-pruned structure mismatch: got {actual:,} params, expected {expected:,}"
        )
    return yolo, expected, before


def main() -> None:
    config = FineTuneConfig(
        weights=ROUTE_PRUNED,
        data=str(DATASET_YAML),
        epochs=100,
        batch=16,
        workers=4,
        imgsz=640,
        device=0,
        project=str(ROOT / "runs/detect"),
        name="route_aware_C_e100",
        optimizer="auto",
        lr0=0.01,                 # 公平冷启动(非 warm 0.001)
        lrf=0.01,
        warmup_epochs=1.0,
        close_mosaic=10,          # 与 unpruned 冷臂一致
        sa_tal=True,
        sa_size=40.0,
        mcsr=True,                # Route-Aware 剪枝结构带 MCSR 路由
        seed=0,
        unpruned=False,
    )
    overrides = build_train_overrides(config)
    overrides["data"] = str(DATASET_YAML)
    overrides["cls_remap"] = True
    overrides["auto_augment"] = "randaugment"
    overrides["erasing"] = 0.4

    yolo, expected, before = build_cold_route_model(seed=config.seed, mcsr=config.mcsr)
    print(
        f"[cold-route] pruning source before={before:,} after={expected:,}; "
        f"cold start: initialize_weights + load(yolo11n.pt); lr0=0.01/100ep"
    )

    trainer = build_structure_preserving_trainer(yolo, overrides)
    trainer.train()

    best = Path(trainer.best).resolve() if Path(trainer.best).is_file() else None
    last = Path(trainer.last).resolve() if Path(trainer.last).is_file() else None
    for ck in (best, last):
        if ck is not None:
            verify_finetuned_checkpoint(ck, expected, mcsr=config.mcsr)
            print(f"[cold-route] verified structure preserved: {ck}")
    print(f"[cold-route] save_dir={Path(trainer.save_dir).resolve()}")
    print(f"[cold-route] best={best} last={last}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # Windows spawn: 防止子进程在 bootstrap 前重入模块
    main()
