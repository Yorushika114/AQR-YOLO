# -*- coding: utf-8 -*-
"""Mosaic 消融 —— COCO 口径尺寸分层评测 (AP_small / AP_medium / AP_large)。

为什么需要这一步:
    "关闭 Mosaic 是否合理"的机理假设是 *Mosaic 把 4 图拼成 1 图 → 目标被缩小、
    缺陷形态被切割, 对小目标不利*。全类平均 mAP 会被大目标类稀释, 逐类 AP 又
    混入了类内尺寸分布(如 Surface_Crack 本身横跨小/中/大三档)。只有 COCO 口径
    的 AP_small 才直接回答"小目标到底有没有变好" —— 这是审稿人认的标准证据。

与 ultralytics 自带评测的关系(重要, 别混为一谈):
    ultralytics 的 results.csv / model.val() 用的是自研匹配策略(无 area range、
    无 maxDets 截断), 其 mAP50-95 与 COCO AP **不应指望相等**。本脚本产出的是
    独立第二口径, 用于交叉印证趋势, 不用于替换 results.csv 中的任何数值。

用法:
    python tools/eval_mosaic_coco_size.py [--rebuild-gt]

产出:
    runs/detect/mosaic_ablation/coco_gt_val.json          (GT, 可缓存复用)
    runs/detect/mosaic_ablation/coco_size_eval.json
    runs/detect/mosaic_ablation/coco_size_eval.md
    runs/detect/mosaic_ablation/val_runs/coco_<arm>/predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须用 ultralytics 自己的 reader: 裸 cv2.imread 在 Windows 上读不了含中文的
# 路径(本数据集根目录就是 "浏览器下载位置"), 会直接返回 None。
from ultralytics.utils.patches import imread  # noqa: E402

DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
DATASET_ROOT = Path(r"D:/浏览器下载位置/detection_yolo_v2")
VAL_IMAGES = DATASET_ROOT / "images/val"
VAL_LABELS = DATASET_ROOT / "labels/val"

OUT_DIR = ROOT / "runs/detect/mosaic_ablation"
GT_JSON = OUT_DIR / "coco_gt_val.json"

ARMS = {
    "on": {"run": "mosaic_on_e100", "label": "YOLO11n (Mosaic 开)"},
    "off": {"run": "mosaic_off_e100", "label": "YOLO11n (Mosaic 关)"},
}

# ultralytics detect/val.py:91 -> class_map = list(range(1, nc+1))  (非 COCO 数据集)
# 故 category_id = 类别下标 + 1。写死成常量, 与 pred_to_json 保持一致。
CLASS_NAMES = ["Pinhole", "Surface_Crack", "Delamination"]
CATEGORY_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}

# COCO 官方小/中/大目标面积阈值 (px^2)
AREA_SMALL_MAX = 32**2
AREA_MEDIUM_MAX = 96**2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 独立预算的 val 集真值(来自此前对 labels/val 的直接统计), 用于对拍验证 GT 构建。
# 对不上 => GT 建错, 必须中止而不是继续产数。
EXPECTED_BOXES_BY_CLASS = {"Pinhole": 776, "Surface_Crack": 3987, "Delamination": 3291}
EXPECTED_IMAGES = 1550


def label_signature() -> dict:
    """labels/val + images/val 的廉价指纹, 用于判断缓存的 GT 是否已过期。

    只读文件元数据(名/大小/mtime), 不读内容, 成本 O(目录) 级, 每次评测都能跑。
    """
    label_files = sorted(VAL_LABELS.glob("*.txt"))
    total_bytes = 0
    newest = 0.0
    for p in label_files:
        st = p.stat()
        total_bytes += st.st_size
        newest = max(newest, st.st_mtime)
    n_images = (
        sum(1 for p in VAL_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if VAL_IMAGES.is_dir()
        else 0
    )
    return {
        "label_files": len(label_files),
        "label_bytes": total_bytes,
        "label_mtime_max": round(newest, 3),
        "image_files": n_images,
    }


def build_gt(force: bool = False) -> dict:
    """把 YOLO 格式的 labels/val 转成 COCO GT json, 带自校验。

    缓存复用前会比对 labels/val 指纹: EXPECTED_* 自校验只长在**构建**路径上, 若标注
    在两次评测之间被改动过, 只凭"缓存文件存在"就会拿着过期 GT 出数 —— 而这恰恰是那个
    守卫本来要防的事。指纹不符则自动重建(重建路径会重跑自校验)。
    """
    if GT_JSON.is_file() and not force:
        gt = json.loads(GT_JSON.read_text(encoding="utf-8"))
        cached_sig = gt.get("info", {}).get("source_signature")
        current_sig = label_signature()
        if cached_sig == current_sig:
            print(
                f"[gt] 复用缓存 {GT_JSON} "
                f"({len(gt['images'])} 图 / {len(gt['annotations'])} 框, 指纹一致)"
            )
            return gt
        print(f"[gt] 缓存指纹不符, 自动重建。\n     缓存={cached_sig}\n     当前={current_sig}")
        force = True

    if not VAL_IMAGES.is_dir():
        raise FileNotFoundError(f"val 图像目录不存在: {VAL_IMAGES}")

    image_paths = sorted(
        p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"val 图像目录为空: {VAL_IMAGES}")

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0
    box_counter: Counter[str] = Counter()
    clamped = 0
    rounding_only = 0
    missing_label = 0
    bad_lines = 0

    for img_path in image_paths:
        # 与 ultralytics 的读图方式保持一致(IMREAD_COLOR), 否则 EXIF 旋转等
        # 边缘情况会让 GT 尺寸与实际推理时用的 ori_shape 不一致。
        img = imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        height, width = img.shape[:2]

        stem = img_path.stem
        # 与 val.py:433 `int(stem) if stem.isnumeric() else stem` 完全一致的 id 规则
        image_id = int(stem) if stem.isnumeric() else stem

        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": int(width),
                "height": int(height),
            }
        )

        label_path = VAL_LABELS / f"{stem}.txt"
        if not label_path.is_file():
            missing_label += 1
            continue

        for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split()
            if not parts:
                continue
            if len(parts) < 5:
                bad_lines += 1
                continue
            cls_idx = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])

            # 归一化 xywh -> 绝对 xywh (COCO bbox = 左上角 x, y, w, h)
            abs_w = bw * width
            abs_h = bh * height
            x = cx * width - abs_w / 2
            y = cy * height - abs_h / 2

            # 越界裁剪: 统计并如实上报, 不静默修改。
            nx0, ny0 = max(0.0, x), max(0.0, y)
            nx1, ny1 = min(float(width), x + abs_w), min(float(height), y + abs_h)
            overshoot = max(0.0, -x, -y, (x + abs_w) - width, (y + abs_h) - height)
            # 区分"真实裁剪"与"浮点末位": 本数据集实测最大越界仅 3.2e-4 px, 属纯舍入,
            # 两者混为一谈会让人误以为标注质量有问题。阈值取 1e-3 px。
            if overshoot > 1e-3:
                clamped += 1
            elif overshoot > 1e-9:
                rounding_only += 1
            w_clip, h_clip = max(0.0, nx1 - nx0), max(0.0, ny1 - ny0)
            if w_clip <= 0 or h_clip <= 0:
                bad_lines += 1
                continue

            name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            box_counter[name] += 1
            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID.get(name, cls_idx + 1),
                    "bbox": [nx0, ny0, w_clip, h_clip],
                    "area": w_clip * h_clip,
                    "iscrowd": 0,
                }
            )

    gt = {
        "info": {
            "description": "detection_yolo_v2 val split (YOLO->COCO)",
            # 供下次复用时比对, 见 label_signature()
            "source_signature": label_signature(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CATEGORY_ID[n], "name": n, "supercategory": "defect"} for n in CLASS_NAMES
        ],
    }

    # ---- 自校验: 与独立统计对拍 ----
    problems = []
    if len(images) != EXPECTED_IMAGES:
        problems.append(f"图像数 {len(images)} != 预期 {EXPECTED_IMAGES}")
    for name, expected in EXPECTED_BOXES_BY_CLASS.items():
        got = box_counter.get(name, 0)
        if got != expected:
            problems.append(f"{name} 框数 {got} != 预期 {expected}")
    if problems:
        raise SystemExit(
            "[gt] GT 构建自校验失败, 拒绝继续(避免产出错误指标):\n  - "
            + "\n  - ".join(problems)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GT_JSON.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    print(
        f"[gt] 已生成 {GT_JSON}: {len(images)} 图 / {len(annotations)} 框 "
        f"(自校验通过; 缺标注文件 {missing_label} 图, 废弃行 {bad_lines}, "
        f"真实越界裁剪 {clamped} 框, 浮点末位触边 {rounding_only} 框)"
    )
    if clamped:
        print(f"[gt] 注意: {clamped} 个框真实越界(超出 >1e-3 px)并被裁剪, 已计数未静默处理。")
    if rounding_only:
        print(
            f"[gt] 说明: {rounding_only} 个框恰好贴图像边缘、浮点末位落到边界外(<1e-3 px), "
            "裁剪量小于千分之一像素, 对面积与尺寸分档无影响。"
        )
    if bad_lines:
        print(f"[gt] 注意: {bad_lines} 行标注被跳过(格式异常)。")
    return gt


def summarize_gt_sizes(gt: dict) -> dict[str, Counter]:
    """按 COCO 面积阈值统计 GT 各类的尺寸分布(可用于核对小目标占比)。"""
    id2name = {c["id"]: c["name"] for c in gt["categories"]}
    dist: dict[str, Counter] = {n: Counter() for n in CLASS_NAMES}
    for ann in gt["annotations"]:
        area = ann["area"]
        bucket = "small" if area < AREA_SMALL_MAX else ("medium" if area < AREA_MEDIUM_MAX else "large")
        dist[id2name[ann["category_id"]]][bucket] += 1
    return dist


def run_arm(arm_key: str, spec: dict) -> tuple[Path, dict]:
    """跑**一次** val, 同时拿到逐类指标与 predictions.json。

    复用 eval_mosaic_perclass.evaluate(..., save_json=True): 一次推理同时喂给
    per-class 与 COCO 两套分析, 避免两次独立 val 的数值漂移, 也省一半 GPU 时间。
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_mosaic_perclass import evaluate  # noqa: PLC0415

    weights = ROOT / f"runs/detect/{spec['run']}/weights/best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"缺少权重: {weights}")

    print(f"[val] {spec['label']} <- {weights}", flush=True)
    result = evaluate(arm_key, spec, save_json=True, project=str(OUT_DIR / "val_runs"))

    pred_json = result.get("pred_json")
    if not pred_json or not Path(pred_json).is_file():
        raise FileNotFoundError(
            f"未生成 predictions.json(arm={arm_key})。"
            "检查 ultralytics validator 的 save_json 门控是否吞掉了写盘。"
        )
    n = len(json.loads(Path(pred_json).read_text(encoding="utf-8")))
    print(f"[val] {spec['label']}: predictions.json {n} 条 -> {pred_json}", flush=True)
    return Path(pred_json), result


def coco_eval(gt_json: Path, pred_json: Path) -> dict:
    """跑 pycocotools COCOeval, 取全档 + 小/中/大 + AR。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json))

    # COCOeval 只遍历 GT 的 catIds, loadRes 也不校验预测的 category_id —— 于是任何
    # category_id 不在 GT 类别里的检测都会被**静默丢弃**(既不算 TP 也不算 FP)。
    # 由于只会丢检测、不会丢 GT, 效果是 AP/AP_small 单向虚高且不报错。必须显式拦截。
    gt_cats = set(coco_gt.getCatIds())
    preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    stray = sorted({a["category_id"] for a in preds} - gt_cats)
    if stray:
        raise SystemExit(
            f"[coco] predictions.json 含 GT 中不存在的 category_id {stray} "
            f"(GT 类别为 {sorted(gt_cats)})。COCOeval 会静默丢弃这些检测并导致 AP 虚高, "
            "故在此中止。请检查 CLASS_NAMES / CATEGORY_ID 与 data.yaml 的类别数是否一致。"
        )

    coco_dt = coco_gt.loadRes(str(pred_json))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # ev.stats 是**固定的 12 元向量**, 不是 4 档面积的网格 —— 它把 maxDets 和 area 两个
    # 维度压在一维里, 槽位由 pycocotools/cocoeval.py 的 _summarizeDets() 写死:
    #   s[0..5]  = AP, AP50, AP75, AP_small, AP_medium, AP_large
    #   s[6..8]  = AR @ maxDets=1 / 10 / 100  (area=all)
    #   s[9..11] = AR @ maxDets=100, area=small / medium / large
    # 2026-09-12 审计确认: 此前把 s[6..9]/s[11] 当成了 AR_all/small/medium/large/100,
    # 五个 AR 行全部错位(s[10] 真正的 AR_medium 从未被读取)。AP 六行本来就是对的。
    s = [float(x) for x in ev.stats]
    if len(s) < 12:
        raise ValueError(f"COCOeval.stats 长度异常: {len(s)}")
    return {
        "AP": s[0] * 100,
        "AP50": s[1] * 100,
        "AP75": s[2] * 100,
        "AP_small": s[3] * 100,
        "AP_medium": s[4] * 100,
        "AP_large": s[5] * 100,
        "AR_maxDets1": s[6] * 100,    # AR @ maxDets=1,   area=all
        "AR_maxDets10": s[7] * 100,   # AR @ maxDets=10,  area=all
        "AR_all": s[8] * 100,         # AR @ maxDets=100, area=all
        "AR_maxDets100": s[8] * 100,  # 与 AR_all 同值, 保留该键名让 .md 行指向真正的 AR@100
        "AR_small": s[9] * 100,       # AR @ maxDets=100, area=small
        "AR_medium": s[10] * 100,     # AR @ maxDets=100, area=medium
        "AR_large": s[11] * 100,      # AR @ maxDets=100, area=large
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-gt", action="store_true", help="强制重建 GT json")
    args = ap.parse_args()

    gt = build_gt(force=args.rebuild_gt)

    dist = summarize_gt_sizes(gt)
    print("\n[gt] val 集尺寸分布 (COCO 阈值: small<32^2, medium<96^2, large>=96^2):")
    print("| 类别 | small | medium | large | 合计 | small 占比 |")
    print("|---|---|---|---|---|---|")
    for name in CLASS_NAMES:
        c = dist[name]
        tot = sum(c.values())
        pct = (c["small"] / tot * 100) if tot else 0.0
        print(f"| {name} | {c['small']} | {c['medium']} | {c['large']} | {tot} | {pct:.1f}% |")

    results: dict[str, dict] = {}
    perclass: dict[str, dict] = {}
    for key, spec in ARMS.items():
        pred_json, pc = run_arm(key, spec)
        perclass[key] = pc
        results[key] = coco_eval(GT_JSON, pred_json)
        print(f"[coco] {spec['label']}: AP={results[key]['AP']:.2f} "
              f"AP_small={results[key]['AP_small']:.2f} "
              f"AP_medium={results[key]['AP_medium']:.2f} "
              f"AP_large={results[key]['AP_large']:.2f}", flush=True)

    # 同一次 val 也产出逐类 AP, 与 per-class 脚本口径一致, 一并落盘。
    class_order = perclass["on"].get("class_order") or CLASS_NAMES
    pc_lines = ["# Mosaic 消融 —— 逐类 AP 对比", ""]
    pc_lines.append("数据来源: 与 COCO 尺寸分层评测**同一次** val(仅 mosaic 不同)。")
    pc_lines.append("")
    for title, field in [
        ("逐类 AP50", "ap50"),
        ("逐类 AP50-95", "ap50_95"),
        ("逐类 Recall", "recall"),
    ]:
        pc_lines += ["", f"## {title}", "",
                     "| 类别 | Mosaic 开 | Mosaic 关 | 差值 (关 − 开) |", "|---|---|---|---|"]
        for cls in class_order:
            a = perclass["on"]["per_class"].get(cls, {}).get(field)
            b = perclass["off"]["per_class"].get(cls, {}).get(field)
            if a is None or b is None:
                pc_lines.append(f"| {cls} | n/a | n/a | n/a |")
            else:
                pc_lines.append(f"| {cls} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
    pc_lines.append("")
    (OUT_DIR / "perclass_eval.md").write_text("\n".join(pc_lines), encoding="utf-8")
    (OUT_DIR / "perclass_eval.json").write_text(
        json.dumps(perclass, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print("\n".join(pc_lines))
    print(f"Saved: {OUT_DIR / 'perclass_eval.md'}")

    # ---- 汇总 ----
    fields = [
        ("AP (IoU .50:.95)", "AP"),
        ("AP50", "AP50"),
        ("AP75", "AP75"),
        ("AP_small", "AP_small"),
        ("AP_medium", "AP_medium"),
        ("AP_large", "AP_large"),
        ("AR (maxDets=100)", "AR_maxDets100"),
        ("AR_small", "AR_small"),
        ("AR_medium", "AR_medium"),
        ("AR_large", "AR_large"),
    ]
    lines = ["# Mosaic 消融 —— COCO 口径尺寸分层评测", ""]
    lines.append("单变量对照: 仅 mosaic 不同(1.0 vs 0.0), 其余训练设置完全一致。")
    lines.append("")
    lines.append("评测口径: pycocotools COCOeval(bbox), 面积阈值为 COCO 官方定义。")
    lines.append("注: 本表与 results.csv 的 mAP50-95 口径不同, **不应指望相等**, 仅作独立交叉印证。")
    lines.append("")
    lines.append("| 指标 | YOLO11n (Mosaic 开) | YOLO11n (Mosaic 关) | 差值 (关 − 开) |")
    lines.append("|---|---|---|---|")
    for label, key in fields:
        a, b = results["on"][key], results["off"][key]
        lines.append(f"| {label} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")

    md_path = OUT_DIR / "coco_size_eval.md"
    js_path = OUT_DIR / "coco_size_eval.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    js_path.write_text(
        json.dumps(
            {"arms": results, "gt_size_distribution": {k: dict(v) for k, v in dist.items()}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print("\n".join(lines))
    print(f"Saved: {md_path}")
    print(f"Saved: {js_path}")


if __name__ == "__main__":
    main()
