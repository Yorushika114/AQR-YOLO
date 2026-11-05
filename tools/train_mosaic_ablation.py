# -*- coding: utf-8 -*-
"""Mosaic 流程优化验证试验: YOLO11n (Mosaic 开) vs YOLO11n (Mosaic 关)。

单变量对照设计
--------------
两臂**只有 mosaic 一个变量不同**, 其余训练设置与本地既有 100ep 公平协议完全同口径:

    yolo11n.pt 冷启动 / epochs=100 / batch=16 / imgsz=640 / seed=0 / deterministic /
    optimizer=auto / lr0=0.01 / lrf=0.01 / warmup_epochs=1.0 / close_mosaic=10 /
    auto_augment=randaugment / erasing=0.4 / mixup=0 / copy_paste=0

    arm A  mosaic_on_e100   mosaic=1.0   (标准流程, 与 baseline-2 一致)
    arm B  mosaic_off_e100  mosaic=0.0   (Mosaic 关闭)

说明
----
* 两臂均显式传 close_mosaic=10, 使 args.yaml 仅 mosaic 一行不同; mosaic=0.0 时
  close_mosaic 逻辑不生效(本就没有 mosaic 可关), 因此不构成额外变量。
* 结论口径: 对比两臂 epoch-100 的 mAP50 / mAP50-95 / 召回率 与逐类 AP
  (Pinhole / Surface_Crack / Delamination), 判定"关闭 Mosaic"是否合理。

用法
----
    python tools/train_mosaic_ablation.py --arm on
    python tools/train_mosaic_ablation.py --arm off
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
MODEL_YAML = "ultralytics/cfg/models/11/yolo11.yaml"
PRETRAINED = "yolo11n.pt"
PROJECT = str(ROOT / "runs/detect")

# 单变量: 两臂仅 mosaic 不同。
ARMS: dict[str, dict] = {
    "on": {"name": "mosaic_on_e100", "mosaic": 1.0},
    "off": {"name": "mosaic_off_e100", "mosaic": 0.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True, help="on=Mosaic 开, off=Mosaic 关")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = ARMS[args.arm]
    init_seeds(args.seed, deterministic=True)
    model = YOLO(MODEL_YAML).load(PRETRAINED)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(
        f"[mosaic-ablation] arm={args.arm} name={spec['name']} "
        f"mosaic={spec['mosaic']} params={n_params:,}",
        flush=True,
    )

    model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        # 两臂一致(mosaic=0 时不生效), 保证 args.yaml 仅 mosaic 一行不同。
        close_mosaic=10,
        mosaic=spec["mosaic"],
        mixup=0.0,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.4,
        amp=True,
        val=True,
        save=True,
        plots=True,
        exist_ok=True,
        project=PROJECT,
        name=spec["name"],
    )
    print(f"[mosaic-ablation] arm={args.arm} DONE -> runs/detect/{spec['name']}", flush=True)


if __name__ == "__main__":
    main()
