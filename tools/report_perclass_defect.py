# -*- coding: utf-8 -*-
"""逐类对比的出图与表格 —— 论文 3.3 节。

读 perclass_results.json，产出：
  pr_curves_per_class.png    逐类 PR 曲线（3 类分图 + 总体）
  table_3_3.md               结果表（| 模型 | 类别 | P/% | R/% | AP50/% | AP50-95/% | 漏检数 | 误检数 |）

配色：蓝 #1F6FEB（YOLO11n 基线，实线）/ 橙 #E8590C（MSA-YOLO-RAP，虚线）。
已用 dataviz 校验器验证：CVD ΔE 30.5(protan)/34.3(tritan)，全项 PASS；
另加线型二次编码，保证灰度印刷可辨。

用法:
    python tools/report_perclass_defect.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

OUT_DIR = Path(r"C:/Users/23223/Desktop/新论文数据保存/逐类检测性能对比试验")
RESULTS_JSON = OUT_DIR / "perclass_results.json"

C_BASE = "#1F6FEB"   # YOLO11n
C_MSA = "#E8590C"    # MSA-YOLO-RAP

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
rcParams["axes.edgecolor"] = "#B0B0B0"
rcParams["axes.linewidth"] = 0.8
rcParams["grid.color"] = "#DDDDDD"
rcParams["grid.linewidth"] = 0.6

CN = {"Pinhole": "针孔缺陷 (Pinhole)", "Surface_Crack": "细长裂纹 (Surface_Crack)",
      "Delamination": "分层缺陷 (Delamination)"}


def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> None:
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    res = payload["results"]
    base, msa = res["baseline"], res["msa"]
    classes = base["class_order"]
    gt = payload["gt_counts"]

    # ------------------------------------------------------------ PR 曲线
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    panels = [(axes[0][0], classes[0]), (axes[0][1], classes[1]),
              (axes[1][0], classes[2]), (axes[1][1], None)]

    for ax, cls in panels:
        if cls is not None:
            for arm, color, ls, label in ((base, C_BASE, "-", base["label"]),
                                          (msa, C_MSA, "--", msa["label"])):
                cur = arm["pr_curves"].get(cls)
                if not cur:
                    continue
                ap = arm["per_class"][cls]["ap50_95"]
                ax.plot(cur["recall"], cur["precision"], color=color, linestyle=ls,
                        linewidth=2.0, label=f"{label}  (AP$_{{50-95}}$={ap:.2f}%)")
            ax.set_title(CN.get(cls, cls), fontsize=11)
        else:
            # 总体：三类平均 PR（按各类 AP 加权的直观汇总）
            for arm, color, ls, label in ((base, C_BASE, "-", base["label"]),
                                          (msa, C_MSA, "--", msa["label"])):
                curves = [arm["pr_curves"][c] for c in classes if c in arm["pr_curves"]]
                if not curves:
                    continue
                rec = curves[0]["recall"]
                mean_prec = [sum(c["precision"][i] for c in curves) / len(curves) for i in range(len(rec))]
                ax.plot(rec, mean_prec, color=color, linestyle=ls, linewidth=2.0,
                        label=f"{label}  (mAP$_{{50-95}}$={arm['overall']['mAP50_95']:.2f}%)")
            ax.set_title("三类平均 (Mean over classes)", fontsize=11)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    fig.suptitle("YOLO11n 与 MSA-YOLO-RAP 在锂电池表面缺陷三类别上的 PR 曲线", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pr_png = OUT_DIR / "pr_curves_per_class.png"
    fig.savefig(pr_png, dpi=300)
    plt.close(fig)
    print(f"[plot] {pr_png}")

    # ------------------------------------------------------------ 结果表
    head = ("| 模型 | 类别 | P/% | R/% | AP50/% | AP50-95/% | 漏检数 | 误检数 |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|")
    lines = [head]
    gains = {}

    for cls in classes:
        b, m = base["per_class"][cls], msa["per_class"][cls]
        lines.append(
            f"| YOLO11n | {cls} ({b['class_id']}) | {fmt(b['precision'])} | {fmt(b['recall'])} | "
            f"{fmt(b['ap50'])} | {fmt(b['ap50_95'])} | {b['fn']} | {b['fp']} |")
        lines.append(
            f"| MSA-YOLO-RAP | {cls} ({m['class_id']}) | {fmt(m['precision'])} | {fmt(m['recall'])} | "
            f"{fmt(m['ap50'])} | {fmt(m['ap50_95'])} | {m['fn']} | {m['fp']} |")
        # 提升幅度行已按要求移除
        gains[cls] = {"dP": m["precision"] - b["precision"], "dR": m["recall"] - b["recall"],
                      "dAP50": m["ap50"] - b["ap50"], "dAP50_95": m["ap50_95"] - b["ap50_95"],
                      "dFN": m["fn"] - b["fn"], "dFP": m["fp"] - b["fp"], "gt": b["gt_count"],
                      "base_fn": b["fn"], "msa_fn": m["fn"], "base_fp": b["fp"], "msa_fp": m["fp"]}

    # 总体行
    lines.append(
        f"| YOLO11n | 总体 (all) | {fmt(base['overall']['precision'])} | {fmt(base['overall']['recall'])} | "
        f"{fmt(base['overall']['mAP50'])} | {fmt(base['overall']['mAP50_95'])} | "
        f"{sum(base['per_class'][c]['fn'] for c in classes)} | "
        f"{sum(base['per_class'][c]['fp'] for c in classes)} |")
    lines.append(
        f"| MSA-YOLO-RAP | 总体 (all) | {fmt(msa['overall']['precision'])} | {fmt(msa['overall']['recall'])} | "
        f"{fmt(msa['overall']['mAP50'])} | {fmt(msa['overall']['mAP50_95'])} | "
        f"{sum(msa['per_class'][c]['fn'] for c in classes)} | "
        f"{sum(msa['per_class'][c]['fp'] for c in classes)} |")

    (OUT_DIR / "table_3_3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "gains.json").write_text(json.dumps(gains, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[save] {OUT_DIR / 'table_3_3.md'}")
    print(f"[save] {OUT_DIR / 'gains.json'}")


if __name__ == "__main__":
    main()
