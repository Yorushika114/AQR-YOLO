# -*- coding: utf-8 -*-
"""逐类对比的可视化与图表 —— PR 曲线 + 典型检测样例（论文 3.3 节配图）。

产出（写入 桌面/新论文数据保存/逐类检测性能对比试验/）：
  pr_curves_per_class.png        逐类 PR 曲线（3 子图 + 总览）
  viz/pinhole.png                针孔缺陷典型样例（GT | YOLO11n | MSA-YOLO-RAP）
  viz/crack.png                  细长裂纹典型样例
  viz/delamination.png           分层缺陷典型样例
  viz/msa_hit_baseline_miss_*.png   本文检出、基线漏检的代表性样例
  viz/baseline_false_positive_*.png 基线误检、本文无误检的样例
  viz/manifest.json              每张图对应的图像文件名与逐类 TP/FP/FN

依赖 eval_perclass_defect.py 先跑出 perclass_results.json。

用法:
    python tools/viz_perclass_defect.py
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

DATASET_DIR = Path(r"D:/浏览器下载位置/detection_yolo_v2")
OUT_DIR = Path(r"C:/Users/23223/Desktop/新论文数据保存/逐类检测性能对比试验")
VIZ_DIR = OUT_DIR / "viz"
RESULTS_JSON = OUT_DIR / "perclass_results.json"

IMGSZ = 640
DEVICE = 0
BATCH = 8
CONF = 0.25
IOU_NMS = 0.5
IOU_MATCH = 0.5

# 类别配色（BGR）
CLS_COLOR = {0: (222, 134, 46), 1: (83, 82, 238), 2: (132, 172, 16)}  # 针孔 / 裂纹 / 分层
CLS_NAME = {0: "Pinhole", 1: "Surface_Crack", 2: "Delamination"}
MISS_COLOR = (60, 60, 255)    # 漏检 GT（红）
FP_COLOR = (0, 165, 255)      # 误检（橙）


# ---------------------------------------------------------------- 几何
def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def load_gt(label_path: Path, w: int, h: int):
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def match_image(preds, gts):
    """返回 (tp 对数, 每个GT是否命中, 每个pred是否命中)。贪心 + 类内 IoU>=0.5。"""
    matched_gt: set[int] = set()
    pred_hit = [False] * len(preds)
    for pi, (pc, _cf, pb) in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, (gc, gb) in enumerate(gts):
            if gc != pc or j in matched_gt:
                continue
            v = iou_xyxy(pb, gb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_MATCH:
            matched_gt.add(best_j)
            pred_hit[pi] = True
    return matched_gt, pred_hit


# ---------------------------------------------------------------- 绘制
def draw_gt(img, gts):
    for c, (x1, y1, x2, y2) in gts:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), CLS_COLOR.get(c, (200, 200, 200)), 2)
        cv2.putText(img, CLS_NAME.get(c, str(c)), (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLS_COLOR.get(c, (200, 200, 200)), 1, cv2.LINE_AA)


def draw_preds(img, preds, gts, matched_gt, pred_hit, title):
    """画预测框；漏检 GT 用红色虚线标签标出，误检预测标 FP。"""
    for j, (gc, gb) in enumerate(gts):
        if j not in matched_gt:
            x1, y1, x2, y2 = (int(v) for v in gb)
            cv2.rectangle(img, (x1, y1), (x2, y2), MISS_COLOR, 1)
            cv2.putText(img, "MISS", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MISS_COLOR, 1, cv2.LINE_AA)
    for pi, (pc, cf, pb) in enumerate(preds):
        x1, y1, x2, y2 = (int(v) for v in pb)
        if pred_hit[pi]:
            col = CLS_COLOR.get(pc, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{CLS_NAME.get(pc, pc)} {cf:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), FP_COLOR, 2)
            cv2.putText(img, f"FP {cf:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, FP_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)


def hstack(panels, gap=8):
    h = max(p.shape[0] for p in panels)
    fixed = []
    for p in panels:
        if p.shape[0] < h:
            p = cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        fixed.append(p)
    sep = np.full((h, gap, 3), 40, dtype=np.uint8)
    out = fixed[0]
    for p in fixed[1:]:
        out = np.hstack([out, sep, p])
    return out


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_preds(model, src, batch=BATCH):
    """返回 {stem: [(cls, conf, box), ...]}，按 conf 降序。

    src 必须是**目录或 glob 字符串**，不能是路径 list：
    predict 收到 list[str] 会走 LoadPilAndNumpy，它把整个列表当一个 batch
    （bs = len(列表)），1550 张 = (1550,3,640,640) ≈ 15.2 GiB，直接 OOM。
    传目录才走 LoadImagesAndVideos，batch= 才生效。
    用 r.path 取回每张图的实际路径，因此不依赖迭代顺序。
    """
    out = {}
    with torch.no_grad():
        stream = model.predict(source=src, imgsz=IMGSZ, conf=CONF,
                               iou=IOU_NMS, batch=batch, device=DEVICE, verbose=False,
                               stream=True, max_det=300)
        for r in stream:
            preds = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                preds = [(int(c), float(cf), tuple(float(x) for x in b))
                         for b, c, cf in zip(xyxy, cls, conf)]
            preds.sort(key=lambda t: -t[1])
            out[Path(r.path).stem] = preds
    return out


def main() -> None:
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    arms = payload["results"]
    order = arms["baseline"]["class_order"]

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    val_dir = DATASET_DIR / "images" / "val"
    img_paths = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print(f"[viz] val 图像 {len(img_paths)} 张", flush=True)

    # 一次只加载一个模型：8 GB 卡上同时驻留两个 + 推理缓冲会 OOM
    preds = {}
    for k in ("baseline", "msa"):
        print(f"[viz] 推理 {arms[k]['label']} ...", flush=True)
        m = YOLO(arms[k]["weights"])
        preds[k] = load_preds(m, str(val_dir))
        print(f"[viz]   {arms[k]['label']} 推理完成，{len(preds[k])} 张", flush=True)
        del m
        free_gpu()

    # ---------------------------------------------------------- 逐张逐类统计
    per_img = []
    for p in img_paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = load_gt(DATASET_DIR / "labels" / "val" / f"{p.stem}.txt", w, h)
        rec = {"stem": p.stem, "path": str(p), "n_gt": len(gts), "per_class": {}}
        for k in ("baseline", "msa"):
            pr = preds[k].get(p.stem, [])
            mg, ph = match_image(pr, gts)
            fc = {}
            for c in range(len(order)):
                fc[c] = {
                    "tp": sum(1 for j in mg if gts[j][0] == c),
                    "fn": sum(1 for j, (gc, _) in enumerate(gts) if gc == c and j not in mg),
                    "fp": sum(1 for pi, (pc, _, _) in enumerate(pr) if pc == c and not ph[pi]),
                }
            rec["per_class"][k] = fc
        per_img.append(rec)

    # ---------------------------------------------------------- 选样
    def score_hit(rec, c):
        b, m = rec["per_class"]["baseline"][c], rec["per_class"]["msa"][c]
        return (b["fn"] - m["fn"]) * 10 + (m["tp"] - b["tp"]) * 6 - m["fp"] * 2

    def score_fp(rec, c):
        b, m = rec["per_class"]["baseline"][c], rec["per_class"]["msa"][c]
        return (b["fp"] - m["fp"]) * 10 + (m["tp"] - b["tp"]) * 2

    def score_typical(rec, c):
        b, m = rec["per_class"]["baseline"][c], rec["per_class"]["msa"][c]
        if b["tp"] == 0 and m["tp"] == 0:
            return -999
        return b["tp"] + m["tp"] - (b["fn"] + m["fn"]) * 2 - (b["fp"] + m["fp"])

    manifest = {"samples": []}

    def render(rec, tag, note):
        p = Path(rec["path"])
        base = cv2.imread(str(p))
        h, w = base.shape[:2]
        gts = load_gt(DATASET_DIR / "labels" / "val" / f"{p.stem}.txt", w, h)

        p_gt = base.copy()
        draw_gt(p_gt, gts)
        cv2.putText(p_gt, "GT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)

        panels = [p_gt]
        detail = {}
        for k in ("baseline", "msa"):
            img = base.copy()
            pr = preds[k].get(p.stem, [])
            mg, ph = match_image(pr, gts)
            draw_preds(img, pr, gts, mg, ph, arms[k]["label"])
            panels.append(img)
            fc = rec["per_class"][k]
            detail[k] = {order[c]: fc[c] for c in range(len(order))}

        out = OUT_DIR / "viz" / f"{tag}.png"
        cv2.imwrite(str(out), hstack(panels))
        manifest["samples"].append({"file": out.name, "image": p.name, "note": note, "detail": detail})
        print(f"[viz] {out.name}  <- {p.name}  ({note})", flush=True)

    used: set[str] = set()

    def pick(score_fn, c: int):
        """取该类别得分最高、且尚未被其它图占用的样本（尽量让每张图只出现一次）。"""
        cand = sorted(per_img, key=lambda r: score_fn(r, c), reverse=True)
        for r in cand:
            if r["stem"] not in used:
                return r
        return cand[0] if cand else None

    for c in range(len(order)):
        name = order[c].lower()
        # 1) 典型样例（两模型均检出）
        r = pick(score_typical, c)
        if r is not None:
            used.add(r["stem"])
            render(r, f"{name}_typical", f"{order[c]} 典型样例（两模型均检出）")
        # 2) 本文检出、基线漏检
        cand = sorted(per_img, key=lambda x: score_hit(x, c), reverse=True)
        if cand and score_hit(cand[0], c) > 0:
            r = pick(score_hit, c)
            used.add(r["stem"])
            render(r, f"{name}_msa_hit_baseline_miss", f"{order[c]}：本文检出 / 基线漏检")
        # 3) 基线误检、本文无误检
        cand = sorted(per_img, key=lambda x: score_fp(x, c), reverse=True)
        if cand and score_fp(cand[0], c) > 0:
            r = pick(score_fp, c)
            used.add(r["stem"])
            render(r, f"{name}_baseline_fp", f"{order[c]}：基线误检 / 本文无误检")

    (OUT_DIR / "viz" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[viz] 完成，共 {len(manifest['samples'])} 张 -> {OUT_DIR / 'viz'}")


if __name__ == "__main__":
    main()
