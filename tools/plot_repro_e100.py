from __future__ import annotations

"""复现方法 @100ep vs baseline-2 对比图(同一冷启动协议 house style)。

3 个复现方法(paper01 SD-YOLO / paper03 YOLOv8-GCE / paper09 EIM-YOLO)与权威基线
baseline-2 的 100-epoch 记录对比。全部为全新冷启动:yolo11n.pt, lr0=0.01,
warmup=1.0, close_mosaic10, batch16, imgsz640, seed0, deterministic, cls_remap。

注意:paper09 在末段(ep94+)训练损失发散、验证指标崩落,图例标注其 BEST(ep93)
而非末轮,曲线如实展示整段记录(含发散尾段)。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

BASE_UNPRUNED = Path(r"D:\浏览器下载位置\baseline-2")
REPO = Path(r"D:\个人文件整理\02_开发项目\ultralytics-8.4.94")

FILES = {
    "base": BASE_UNPRUNED / "results.csv",
    "paper01": REPO / "runs/detect/paper01_sd_yolo_100ep/results.csv",
    "paper03": REPO / "runs/detect/paper03_gce_100ep/results.csv",
    "paper09": REPO / "runs/detect/paper09_eim_100ep/results.csv",
}

# 图例:方法名 + 报告口径(全为 best,论文可部署最优检查点)
LABELS = {
    "base": "baseline-2 (fresh, 0.7616)",
    "paper01": "paper01 SD-YOLO (0.7879)",
    "paper03": "paper03 YOLOv8-GCE (0.8098)",
    "paper09": "paper09 EIM-YOLO (0.8090 @ep93; 尾部发散)",
}

COLORS = {
    "base": "#000000",
    "paper01": "#d62728",
    "paper03": "#2ca02c",
    "paper09": "#1f77b4",
}

LINESTYLES = {
    "base": ":",
    "paper01": "-",
    "paper03": "-",
    "paper09": "-",
}

MAX_EPOCH = 100
PLOT_KEYS = list(FILES.keys())
OUTPUT_NAME = "repro_e100_vs_baseline.png"
OUTPUT_PATH = REPO / OUTPUT_NAME

METRICS = [
    ("metrics/mAP50(B)", "(a)", "训练轮次", "mAP50/%"),
    ("metrics/mAP50-95(B)", "(b)", "训练轮次", "mAP50-95/%"),
    ("metrics/recall(B)", "(c)", "训练轮次", "召回率/%"),
]


def load_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for key in PLOT_KEYS:
        path = FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {LABELS[key]}: {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        required = {"epoch", *(m[0] for m in METRICS)}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        df = df.drop_duplicates(subset="epoch", keep="last").reset_index(drop=True)
        df = df[df["epoch"] <= MAX_EPOCH]
        data[key] = df
        print(f"Loaded {LABELS[key]}: {path} ({len(df)} rows)")
    return data


def to_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.max(skipna=True) <= 1.5:
        return values * 100
    return values


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Times New Roman", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    data = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    legend_handles = []
    legend_labels = []

    for ax, (column, sub_caption, x_label, y_label) in zip(axes, METRICS):
        for key in PLOT_KEYS:
            df = data[key]
            line = ax.plot(
                df["epoch"],
                to_percent(df[column]),
                label=LABELS[key],
                color=COLORS[key],
                linestyle=LINESTYLES[key],
                linewidth=1.5,
            )[0]
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(LABELS[key])

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(False)
        ax.set_xlim(1, MAX_EPOCH)
        ax.text(
            0.5, -0.32, sub_caption, transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold",
        )

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=4, fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, 0.97),
    )
    fig.suptitle(
        "复现方法 @100ep vs baseline-2 (全新冷启动 yolo11n.pt, lr0=0.01, 100-epoch)",
        y=1.04, fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.80, wspace=0.34)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
