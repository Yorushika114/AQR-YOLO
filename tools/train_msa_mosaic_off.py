# -*- coding: utf-8 -*-
"""MSA-YOLO (mcsr + sa_tal) + Mosaic 关闭, 冷启动 100 轮。

为什么需要这一臂
----------------
本目录已有的两条真实曲线(协议完全同口径: 本地数据 / 冷启动 / 100轮 / lr0=0.01 /
close_mosaic=10 / seed0 / deterministic):

    mosaic_on_e100   纯 yolo11n, mosaic=1.0  ->  mAP50-95 = 0.83491
    mosaic_off_e100  纯 yolo11n, mosaic=0.0  ->  mAP50-95 = 0.85834
    both_e100        MSA-YOLO,  mosaic=1.0  ->  mAP50-95 = 0.84456  (创新点锚点)

结论: 仅仅把 Mosaic 关掉(+2.343) 就**大于**整套 MSA 模块在 mosaic 开时带来的收益。
也就是说 "MSA-YOLO vs 基线" 的主张目前站不住 —— 便宜的增强改动赢过了创新模块。

论文 §3.2 已判定"关闭 Mosaic 合理", 那么方法本体的正式配置就应当是 mosaic=0。
缺的正是这条:

    both_mosaic_off_e100   MSA-YOLO (mcsr+sa_tal), mosaic=0.0

跑完后才能给出正面对照:
    基线(纯 yolo11n, mosaic 关) 0.85834  vs  MSA-YOLO(mosaic 关) = ?

本脚本只增加 mosaic 一个变量(相对 both_e100), 其余 kwargs 与 both_e100/args.yaml 逐项一致。

用法
----
    D:/anaconda/envs/test/python.exe tools/train_msa_mosaic_off.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
# 必须用 MSA 结构 yaml: 只有它把检测头换成 AdaptiveDefectDetect(带 MCSR 路由)。
# 用普通 yolo11.yaml 时即使 train(mcsr=True) 也仍是普通 Detect 头 —— 结构不会变。
MODEL_YAML = "ultralytics/cfg/models/11/yolo11n-adaptive-defect.yaml"
PRETRAINED = "yolo11n.pt"
PROJECT = str(ROOT / "runs/detect")
NAME = "both_mosaic_off_e100"


def main() -> None:
    init_seeds(0, deterministic=True)
    model = YOLO(MODEL_YAML).load(PRETRAINED)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(
        f"[msa-mosaic-off] name={NAME} mosaic=0.0 mcsr=True sa_tal=True params={n_params:,}",
        flush=True,
    )

    model.train(
        data=DATASET_YAML,
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        workers=4,
        seed=0,
        deterministic=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        close_mosaic=10,          # 与 both_e100 / mosaic_off_e100 一致
        mosaic=0.0,               # 唯一变量
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
        name=NAME,
        mcsr=True,
        sa_tal=True,
        sa_size=40.0,
    )
    print(f"[msa-mosaic-off] DONE -> runs/detect/{NAME}", flush=True)


if __name__ == "__main__":
    main()
