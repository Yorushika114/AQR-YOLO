# -*- coding: utf-8 -*-
"""Mosaic 流程优化验证试验 — 对比作图。

直接读取 runs/detect/mosaic_{on,off}_e100/results.csv 的真实训练产物, 不复制、
不改写任何 results.csv, 曲线可逐点追溯到原始训练, 满足数据完整性要求。

用法:
    python tools/plot_mosaic_ablation.py

产出:
    runs/detect/mosaic_ablation/fig_mosaic_on_off_e100.png
    stdout: markdown 汇总表(终点指标 + 峰值 + 达到峰值轮次)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs/detect/mosaic_ablation"
OUT_NAME = "fig_mosaic_on_off_e100.png"

# 单变量对照: 仅 mosaic 不同, 其余与本地 100ep 公平协议完全一致。
SERIES = {
    "on": {
        "path": ROOT / "runs/detect/mosaic_on_e100/results.csv",
        "label": "YOLO11n (Mosaic 开)",
        "color": "#1F77B4",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "off": {
        "path": ROOT / "runs/detect/mosaic_off_e100/results.csv",
        "label": "YOLO11n (Mosaic 关)",
        "color": "#E45756",
        "linestyle": "--",
        "linewidth": 2.0,
    },
}

METRICS = [
    ("metrics/mAP50(B)", "(a)", "训练轮次", "mAP50/%"),
    ("metrics/mAP50-95(B)", "(b)", "训练轮次", "mAP50-95/%"),
    ("metrics/recall(B)", "(c)", "训练轮次", "召回率/%"),
]

# 不绘图、但汇总表需要的列(否则 summarize 取不到会退化成 n/a)。
EXTRA_COLUMNS = ("metrics/precision(B)",)

MAX_EPOCH = 100


def load(path: Path) -> dict[str, list[float]]:
    """读单个 results.csv, 按 epoch 去重(保留最后一条)并截断到 MAX_EPOCH。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少训练产物: {path}")
    rows: dict[int, dict[str, str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in csv.DictReader(fh):
            row = {k.strip(): v for k, v in raw.items()}
            try:
                epoch = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue
            rows[epoch] = row
    epochs = sorted(e for e in rows if e <= MAX_EPOCH)
    if not epochs:
        raise ValueError(f"无可用的 epoch 行: {path}")
    data: dict[str, list[float]] = {"epoch": epochs}
    columns = [m[0] for m in METRICS] + [
        c for c in EXTRA_COLUMNS if c in rows[epochs[0]]
    ]
    for column in columns:
        data[column] = [float(rows[e][column]) for e in epochs]
    return data


def to_percent(values: list[float]) -> list[float]:
    """results.csv 里指标是 0-1, 统一转成百分数用于绘图。"""
    if values and max(values) <= 1.5:
        return [v * 100 for v in values]
    return values


def summarize(data: dict[str, dict[str, list[float]]]) -> None:
    """打印 markdown 汇总表: 终点值 / 峰值 / 峰值轮次。"""
    header = "| 指标 | " + " | ".join(SERIES[k]["label"] for k in data) + " |"
    sep = "|---|" + "---|" * len(data)
    print(header)
    print(sep)
    rows = [
        ("mAP50 (终点)", "metrics/mAP50(B)", "last"),
        ("mAP50 (峰值)", "metrics/mAP50(B)", "best"),
        ("mAP50-95 (终点)", "metrics/mAP50-95(B)", "last"),
        ("mAP50-95 (峰值)", "metrics/mAP50-95(B)", "best"),
        ("Recall (终点)", "metrics/recall(B)", "last"),
        ("Precision (终点)", "metrics/precision(B)", "last"),
    ]
    for title, column, mode in rows:
        cells = []
        for key in data:
            d = data[key]
            if column not in d:
                cells.append("n/a")
                continue
            vals = to_percent(d[column])
            if mode == "last":
                cells.append(f"{vals[-1]:.2f}")
            else:
                best = max(vals)
                ep = d["epoch"][vals.index(best)]
                cells.append(f"{best:.2f} (ep{ep})")
        print(f"| {title} | " + " | ".join(cells) + " |")
    print()
    for key in data:
        print(f"{SERIES[key]['label']}: 已跑到 epoch {data[key]['epoch'][-1]}")


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Times New Roman", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    data: dict[str, dict[str, list[float]]] = {}
    for key, spec in SERIES.items():
        if spec["path"].is_file():
            data[key] = load(spec["path"])
            print(f"Loaded {spec['label']}: {spec['path']}")
        else:
            print(f"[skip] 尚未出现(该臂未完成/未开始): {spec['path']}")
    if not data:
        raise SystemExit("两臂的 results.csv 都不存在, 无可绘制数据。")

    summarize(data)

    # 版式与论文其余对比图一致: (a)(b) 上排并排, (c) 下排居中, 图例置顶单行。
    fig = plt.figure(figsize=(12.0, 7.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.42, wspace=0.42)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]

    legend_handles, legend_labels = [], []
    max_epoch = max(d["epoch"][-1] for d in data.values())

    for ax, (column, sub_caption, x_label, y_label) in zip(axes, METRICS):
        ends: list[tuple[float, int, dict]] = []
        for key, d in data.items():
            spec = SERIES[key]
            ys = to_percent(d[column])
            line = ax.plot(
                d["epoch"],
                ys,
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=spec["linewidth"],
            )[0]
            ends.append((ys[-1], d["epoch"][-1], spec))
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(spec["label"])

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(False)
        # 右侧留白给终值标签, 顶部留白给错开后的高值标签。
        ax.set_xlim(1, MAX_EPOCH * 1.10)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.10)

        # 终值数值标注, 便于直接读数。
        # 两臂终值接近时(如 mAP50 的 94.01 / 94.56), 固定的 (4, 4) 偏移会把两个
        # 标签叠成一团 —— 论文图里这是硬伤。按数值排序后强制最小垂直间距, 并把
        # 标签统一放到曲线右侧的留白里。
        lo, hi = ax.get_ylim()
        min_gap = (hi - lo) * 0.050
        order = sorted(range(len(ends)), key=lambda i: ends[i][0])
        placed = [0.0] * len(ends)
        for rank, i in enumerate(order):
            y = ends[i][0]
            if rank:
                y = max(y, placed[order[rank - 1]] + min_gap)
            placed[i] = y
        for i, (v, ep, spec) in enumerate(ends):
            ax.annotate(
                f"{v:.2f}",
                xy=(ep, v),
                xytext=(ep + MAX_EPOCH * 0.015, placed[i]),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=8,
                color=spec["color"],
                fontweight="bold",
            )

        ax.text(
            0.5, -0.24, sub_caption,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=12, fontweight="bold",
        )

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=max(len(legend_labels), 1),
        fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.90, wspace=0.42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    if max_epoch < MAX_EPOCH:
        print(f"[注意] 训练尚未跑满 100 轮(当前最长 {max_epoch} 轮), 此图为中间态。")


if __name__ == "__main__":
    main()
