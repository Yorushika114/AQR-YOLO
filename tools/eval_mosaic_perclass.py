# -*- coding: utf-8 -*-
"""Mosaic 消融 —— 逐类 AP 评估(Pinhole / Surface_Crack / Delamination)。

为什么单独做这一步: results.csv 只记录全类平均指标, 而"关闭 Mosaic 是否合理"
的机理假设是 *Mosaic 把 4 图拼 1 图 → 目标尺寸被缩小、缺陷形态被切割, 对微小
缺陷不利*。要验证这个假设, 必须看**逐类**差距, 尤其是小目标类别。总 mAP 打平
但小目标明显变好, 同样构成"关闭合理"的证据; 反之假设不成立。

直接对两臂的 best.pt 跑 val, 不修改任何 results.csv, 结果只读。

用法:
    python tools/eval_mosaic_perclass.py

产出:
    runs/detect/mosaic_ablation/perclass_eval.md
    runs/detect/mosaic_ablation/perclass_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
OUT_DIR = ROOT / "runs/detect/mosaic_ablation"

ARMS = {
    "on": {"run": "mosaic_on_e100", "label": "YOLO11n (Mosaic 开)"},
    "off": {"run": "mosaic_off_e100", "label": "YOLO11n (Mosaic 关)"},
}


def as_list(values) -> list[float]:
    """metric 属性可能是 tensor / ndarray / list, 统一成 python float 列表。

    注意: 末尾的 TypeError 分支会把**标量**静默包成单元素列表。这正是 2026-09-12
    审计里"把 Metric.mp(平均精确率)当成 mAP50 用"能瞒过去的原因 —— 标量请用
    as_scalar(), 不要用 as_list(...)[0]。
    """
    if values is None:
        return []
    try:
        return [float(v) for v in values]
    except TypeError:
        return [float(values)]


def as_scalar(value) -> float | None:
    """取 Metric 的标量属性(mp / mr / map50 / map), 缺失或非标量时返回 None。

    `float()` 对长度 >1 的 ndarray 会抛 TypeError, 故不会把数组悄悄压成一个数。
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate(
    arm_key: str,
    spec: dict,
    save_json: bool = False,
    project: str | None = None,
) -> dict:
    """对某臂 best.pt 跑一次 val, 返回总体 + 逐类指标。

    save_json=True 时同时落盘 predictions.json(供 COCO 口径评测复用)。
    刻意让 per-class 与 COCO 两套分析共用**同一次** val: 分开跑两次会因
    非确定性算子产生细微波差, 导致两个脚本对同一模型给出不同数值。
    """
    weights = ROOT / f"runs/detect/{spec['run']}/weights/best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"缺少权重: {weights}")

    # val 目录是 exist_ok=True, 若上一次跑残留了 predictions.json, 而本次推理恰好
    # 一条检测都没有(validator 只在 jdict 非空时写盘), 陈旧的预测会被当成本次结果
    # 静默复用 —— 那属于用错数据出数。开跑前先清掉。
    if save_json and project:
        stale = Path(project) / f"coco_{arm_key}" / "predictions.json"
        if stale.is_file():
            stale.unlink()
            print(f"[eval] 已丢弃陈旧 {stale}(防止被静默复用)", flush=True)

    model = YOLO(str(weights))
    metrics = model.val(
        data=DATASET_YAML,
        split="val",
        imgsz=640,
        batch=16,
        device=0,
        plots=False,
        verbose=False,
        save_json=save_json,
        **({"project": project, "name": f"coco_{arm_key}", "exist_ok": True} if project else {}),
    )

    box = metrics.box
    names = model.names
    if isinstance(names, dict):
        ordered = [names[i] for i in sorted(names)]
    else:
        ordered = list(names)

    ap50 = as_list(getattr(box, "ap50", None))
    ap = as_list(getattr(box, "ap", None))
    p = as_list(getattr(box, "p", None))
    r = as_list(getattr(box, "r", None))
    # !!! Metric.mp / Metric.mr 是**平均精确率 / 平均召回率**, 不是 mAP !!!
    # mAP 是另外两个属性: map50 = all_ap[:,0].mean(), map = all_ap.mean()。
    # 2026-09-12 审计确认: 此前把 mp 标成 "mAP50"、mr 标成 "mAP50_95", 且 mAP 从未被读取,
    # 导致总体表四个数全错(mAP50-95 被单向虚高约 9 个点)。改回正确属性。
    mean_p = as_scalar(getattr(box, "mp", None))
    mean_r = as_scalar(getattr(box, "mr", None))
    map50_all = as_scalar(getattr(box, "map50", None))
    map_all = as_scalar(getattr(box, "map", None))

    # 关键: ap50/ap 是按 ap_class_index 索引的, 不保证与 0..nc-1 一一对应。
    # 若某类在 val 集中缺失, 直接按 enumerate 对齐会把 AP 错标到别的类别名上,
    # 且不会报错 —— 那属于数据错报。故显式按 ap_class_index 映射。
    ap_class_index = [int(i) for i in as_list(getattr(box, "ap_class_index", None))]
    if not ap_class_index:
        ap_class_index = list(range(len(ap50)))

    per_class = {}
    for j, class_idx in enumerate(ap_class_index):
        name = ordered[class_idx] if class_idx < len(ordered) else f"class_{class_idx}"
        per_class[name] = {
            "ap50": ap50[j] * 100 if j < len(ap50) else None,
            "ap50_95": ap[j] * 100 if j < len(ap) else None,
            "precision": p[j] * 100 if j < len(p) else None,
            "recall": r[j] * 100 if j < len(r) else None,
        }

    pred_json = None
    if save_json:
        save_dir = Path(getattr(metrics, "save_dir", project or "."))
        candidate = save_dir / "predictions.json"
        pred_json = str(candidate) if candidate.is_file() else None

    return {
        "arm": arm_key,
        "label": spec["label"],
        "weights": str(weights),
        "overall": {
            # 数据集层面 mAP -> 必须取 box.map50 / box.map
            "mAP50": (map50_all * 100) if map50_all is not None else None,
            "mAP50_95": (map_all * 100) if map_all is not None else None,
            # 数据集层面平均 P/R -> mp/mr 本身就是各类均值;
            # 不能再取 p[0]/r[0](那是 ap_class_index[0] 即 Pinhole 单类, 不是均值)。
            "precision": (mean_p * 100) if mean_p is not None else None,
            "recall": (mean_r * 100) if mean_r is not None else None,
        },
        "per_class": per_class,
        "class_order": ordered,
        "pred_json": pred_json,
    }


def main() -> None:
    results = []
    for key, spec in ARMS.items():
        print(f"[eval] {spec['label']} <- {spec['run']}/weights/best.pt", flush=True)
        results.append(evaluate(key, spec))

    by_arm = {r["arm"]: r for r in results}
    classes = results[0]["class_order"]

    lines = ["# Mosaic 消融 —— 逐类 AP 对比", ""]
    lines.append("单变量对照: 仅 mosaic 不同(1.0 vs 0.0), 其余训练设置完全一致。")
    lines.append("")
    lines.append("## 总体指标 (val, imgsz=640)")
    lines.append("")
    lines.append("| 指标 | YOLO11n (Mosaic 开) | YOLO11n (Mosaic 关) | 差值 (关 − 开) |")
    lines.append("|---|---|---|---|")
    for label, field in [
        ("mAP50", "mAP50"),
        ("mAP50-95", "mAP50_95"),
        ("Precision", "precision"),
        ("Recall", "recall"),
    ]:
        a = by_arm["on"]["overall"][field]
        b = by_arm["off"]["overall"][field]
        if a is None or b is None:
            lines.append(f"| {label} | n/a | n/a | n/a |")
        else:
            lines.append(f"| {label} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")

    def cell(arm_result: dict, cls: str, field: str):
        """某臂某类的某项指标; 该类在 val 中缺失时返回 None(渲染成 n/a, 不臆造数值)。"""
        entry = arm_result["per_class"].get(cls)
        return None if entry is None else entry[field]

    for title, field in [
        ("逐类 AP50", "ap50"),
        ("逐类 AP50-95", "ap50_95"),
        ("逐类 Recall", "recall"),
    ]:
        lines += ["", f"## {title}", "",
                  "| 类别 | Mosaic 开 | Mosaic 关 | 差值 (关 − 开) |", "|---|---|---|---|"]
        for cls in classes:
            a = cell(by_arm["on"], cls, field)
            b = cell(by_arm["off"], cls, field)
            if a is None or b is None:
                lines.append(f"| {cls} | n/a | n/a | n/a |")
            else:
                lines.append(f"| {cls} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")

    lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "perclass_eval.md"
    json_path = OUT_DIR / "perclass_eval.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"Saved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
