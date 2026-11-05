# -*- coding: utf-8 -*-
"""公平冷启动对比图: 复现方法 + 剪枝方法(新 route-aware) vs 权威基线 baseline-2。

版式与 plot_map_recall_combined_cold.py 一致(2x4 网格,(a)(b) 上排并排、(c) 下排居中、
图例置顶单行、PercentFormatter)。剪枝臂 MSA-YOLO-DG 现在指向新的 route-aware 冷训练结果
(route_aware_C_e100, 0.84613), 替代旧 DG(msa_yolo_dg_pruned, 0.83623)。

系列: baseline-2(0.7616) / paper01 SD-YOLO / paper03 YOLOv8-GCE / paper09 EIM-YOLO /
MSA-YOLO-DG(新 prune)。图中数值为各臂 best mAP50-95。paper09 尾段发散, 用 best(ep 早些)而非末轮。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 公平对比文件夹(桌面)。剪枝臂已替换为 route_aware_C_e100。
BASE = Path(r"C:\Users\23223\Desktop\实验数据\对比实验\复现论文_冷启动100ep_公平")

SERIES = {
    "base": {"path": BASE / "baseline-2" / "results.csv",
             "label": "YOLO11n Baseline", "color": "#17becf", "ls": ":", "lw": 1.8},
    "paper01": {"path": BASE / "paper01_sd_yolo_100ep" / "results.csv",
                "label": "SD-YOLO", "color": "#d62728", "ls": "-", "lw": 1.6},
    "paper03": {"path": BASE / "paper03_gce_100ep" / "results.csv",
                "label": "YOLOv8-GCE", "color": "#e377c2", "ls": ":", "lw": 1.6},
    "paper09": {"path": BASE / "paper09_eim_100ep" / "results.csv",
                "label": "EIM-YOLO", "color": "#7f7f7f", "ls": ":", "lw": 1.6},
    "route": {"path": BASE / "route_aware_C_e100" / "results.csv",
              "label": "MSA-YOLO-RAP", "color": "#1f77b4", "ls": "-", "lw": 2.6},
}

PLOT_KEYS = ["base", "paper01", "paper03", "paper09", "route"]
OUTPUT = BASE / "fig_复现论文_冷启动100ep_公平对比.png"
TITLE = "复现论文 vs 权威基线 baseline-2 (全新冷启动 yolo11n.pt, lr0=0.01, 100-epoch)"

METRICS = [
    ("metrics/mAP50(B)", "(a)", "训练轮次", "mAP50/%"),
    ("metrics/mAP50-95(B)", "(b)", "训练轮次", "mAP50-95/%"),
    ("metrics/recall(B)", "(c)", "训练轮次", "召回率/%"),
]

MAX_EPOCH = 100


def load_data(keys: list[str]) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    for key in keys:
        spec = SERIES[key]
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {spec['label']}: {path}")
        rows: dict[int, dict[str, str]] = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip(): v for k, v in row.items()}
                try:
                    ep = int(row["epoch"])
                except (KeyError, ValueError):
                    continue
                rows[ep] = row
        epochs = sorted(e for e in rows if e <= MAX_EPOCH)
        data: dict[str, list] = {"epoch": epochs}
        for col, _, _, _ in METRICS:
            data[col] = [float(rows[e][col]) for e in epochs]
        out[key] = data
    return out


def to_percent(values: list[float]) -> list[float]:
    if values and max(values) <= 1.5:
        return [v * 100 for v in values]
    return values


def render() -> None:
    data = load_data(PLOT_KEYS)
    fig = plt.figure(figsize=(12.4, 7.8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.42, wspace=0.40)
    axes = [
        fig.add_subplot(grid[0, 0:2]),   # (a) 上左
        fig.add_subplot(grid[0, 2:4]),   # (b) 上右
        fig.add_subplot(grid[1, 1:3]),   # (c) 下中
    ]

    legend_handles, legend_labels = [], []
    for ax, (column, sub_caption, x_label, y_label) in zip(axes, METRICS):
        for key in PLOT_KEYS:
            spec = SERIES[key]
            d = data[key]
            line = ax.plot(
                d["epoch"], to_percent(d[column]),
                label=spec["label"], color=spec["color"],
                linestyle=spec["ls"], linewidth=spec["lw"],
            )[0]
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(spec["label"])
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(False)
        ax.set_xlim(1, MAX_EPOCH)
        ax.text(0.5, -0.24, sub_caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=13, fontweight="bold")

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(TITLE, y=1.02, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.90, wspace=0.40)
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Times New Roman", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    render()


if __name__ == "__main__":
    main()
