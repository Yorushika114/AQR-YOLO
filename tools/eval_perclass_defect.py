# -*- coding: utf-8 -*-
"""不同缺陷类别检测性能对比试验 —— YOLO11n 基线 vs MSA-YOLO-RAP（论文 3.3 节）。

在**完全相同的验证集**上逐类统计 P / R / AP50 / AP50-95，并给出漏检数(FN) / 误检数(FP)。

两臂协议严格一致（仅网络结构与 TAL 分配不同，args.yaml 全量 diff 只差
model / name / mcsr / sa_tal / project / save_dir）：
  - YOLO11n      : runs/detect/mosaic_on_e100  (yolo11.yaml,                 mcsr=F, sa_tal=F)
  - MSA-YOLO-RAP : runs/detect/both_e100       (yolo11n-adaptive-defect.yaml, mcsr=T, sa_tal=T)

口径说明（重要，别踩历史坑）：
  * P / R      = ultralytics 在 **最大 F1 工作点** 的精度/召回（ap_per_class 的定义），
                 不是全阈值平均；也不是 Metric.mp / Metric.mr（那两个是各类均值）。
  * AP50       = box.ap50 (all_ap[:,0])；AP50-95 = box.ap (all_ap.mean(1))。
  * 逐类数值按 box.ap_class_index **位置对齐**，绝不按类 id 直接 enumerate ——
                 某类在 val 中缺失时后者会把 AP 错标到别的类名上且不报错。
  * TP/FP/FN   与上面的 P/R 自洽：TP = round(R*nt)，FP = round(TP/P - TP)，FN = nt - TP，
                 与 ultralytics metrics.py:931-932 同式。nt = 该类别 val 集 GT 框数。
  * 另附 conf=0.25 / IoU=0.5 固定阈值下的独立 IoU 匹配计数，供交叉核对。

显存注意（2026-09 踩坑）：本机 RTX 4060 Laptop 仅 8 GB。**同一进程内先 val 再 predict
会把显存耗干净**（Windows WDDM 下 PyTorch 回落到共享内存，报 allocated=16.61 GiB，
随后 predict 的 forward 触发 OutOfMemoryError）。因此本脚本改为**每个 (阶段 × 模型)
一个独立进程**运行，每次都是全新 CUDA 上下文，并在退出前显式释放。

只读运行：不修改任何 run 目录，不覆盖 results.csv。

用法（5 次独立进程调用）:
    python tools/eval_perclass_defect.py --stage official --arm baseline
    python tools/eval_perclass_defect.py --stage official --arm msa
    python tools/eval_perclass_defect.py --stage fixed    --arm baseline
    python tools/eval_perclass_defect.py --stage fixed    --arm msa
    python tools/eval_perclass_defect.py --stage merge
"""
from __future__ import annotations

import os

# 必须早于 torch 导入；缓解 8 GB 卡上的显存碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import collections
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

DATASET_YAML = r"D:/浏览器下载位置/detection_yolo_v2/data.yaml"
DATASET_DIR = Path(r"D:/浏览器下载位置/detection_yolo_v2")
OUT_DIR = Path(r"C:/Users/23223/Desktop/新论文数据保存/逐类检测性能对比试验")

IMGSZ = 640
BATCH = 16          # val 的 batch
BATCH_PRED = 8      # predict 交叉核对的 batch（显式限制，别用 default.yaml 的 16）
DEVICE = 0
CONF_FIXED = 0.25   # 固定阈值交叉核对用
IOU_NMS = 0.5       # 交叉核对时 NMS 的 IoU（也是匹配阈值）
IOU_MATCH = 0.5

ARMS = {
    "baseline": {
        "run": "mosaic_on_e100",
        "label": "YOLO11n",
        "arch": "yolo11.yaml",
    },
    "msa": {
        "run": "both_e100",
        "label": "MSA-YOLO-RAP",
        "arch": "yolo11n-adaptive-defect.yaml",
    },
    # --- 单因素消融臂（同协议 2x2 因子设计，用于机理归因）---
    # 四臂 args.yaml 除 model / mcsr / sa_tal 外逐字段相同（epochs=100, batch=16,
    # imgsz=640, seed=0, deterministic=True, mosaic=1.0, close_mosaic=10, lr0=0.01）
    "mcsr": {          # 只开结构级多尺度路由
        "run": "mcsr_e100",
        "label": "YOLO11n+MCSR",
        "arch": "yolo11n-adaptive-defect.yaml",
    },
    "sa_tal": {        # 只开尺寸感知正样本分配
        "run": "sa_tal_e100",
        "label": "YOLO11n+SA-TAL",
        "arch": "yolo11.yaml",
    },
}

# 主对比（论文 3.3 表）只用这两臂；其余进 2x2 因子分解
MAIN_ARMS = ("baseline", "msa")
FACTOR_ARMS = ("baseline", "mcsr", "sa_tal", "msa")


# ---------------------------------------------------------------- 基础工具
def as_list(v) -> list[float]:
    """tensor / ndarray / list -> python float 列表。"""
    if v is None:
        return []
    try:
        return [float(x) for x in v]
    except TypeError:
        return [float(v)]


def as_scalar(v) -> float | None:
    """标量属性安全取值；对长度 >1 的 ndarray 返回 None（不静默压扁）。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def free_gpu() -> None:
    """显式回收：Python 对象 -> CUDA 缓存。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def val_split_dir() -> Path:
    return DATASET_DIR / "images" / "val"


def val_image_paths() -> list[Path]:
    return sorted(p for p in val_split_dir().iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def ground_truth_counts() -> dict[int, int]:
    """统计 val 集逐类别 GT 框数（nt）。独立于 ultralytics 内部计数。"""
    cnt: collections.Counter = collections.Counter()
    for lab in (DATASET_DIR / "labels" / "val").glob("*.txt"):
        for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) >= 5:
                cnt[int(float(p[0]))] += 1
    return dict(cnt)


def load_gt_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, tuple]]:
    """读 YOLO 标签 -> [(cls, (x1,y1,x2,y2)), ...]（像素坐标）。"""
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        lo = ((cx - bw / 2) * w, (cy - bh / 2) * h)
        hi = ((cx + bw / 2) * w, (cy + bh / 2) * h)
        out.append((c, (lo[0], lo[1], hi[0], hi[1])))
    return out


def iou_xyxy(a: tuple, b: tuple) -> float:
    """两个 (x1,y1,x2,y2) 的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weights_of(spec: dict) -> Path:
    w = ROOT / f"runs/detect/{spec['run']}/weights/best.pt"
    if not w.is_file():
        raise FileNotFoundError(f"缺少权重: {w}")
    return w


def fmt(v, nd=2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------- 阶段 1：官方 val 指标
def run_official(arm_key: str, spec: dict, gt_counts: dict[int, int]) -> dict:
    """对某臂 best.pt 跑一次 val，返回总体 + 逐类指标 + PR 曲线数据。"""
    weights = weights_of(spec)

    model = YOLO(str(weights))
    head = type(model.model.model[-1]).__name__

    metrics = model.val(
        data=DATASET_YAML,
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=False,
        verbose=False,
        workers=4,
        # val 产物落到输出目录，不污染仓库 runs/
        project=str(OUT_DIR / "_val_runs"),
        name=spec["run"],
        exist_ok=True,
    )

    box = metrics.box
    names = model.names
    ordered = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    ap50 = as_list(box.ap50)
    ap = as_list(box.ap)
    p_arr = as_list(box.p)
    r_arr = as_list(box.r)
    ap_class_index = [int(i) for i in as_list(box.ap_class_index)]

    # 位置对齐（不按类 id 直接索引）
    per_class: dict[str, dict] = {}
    for j, cid in enumerate(ap_class_index):
        name = ordered[cid] if cid < len(ordered) else f"class_{cid}"
        p_i = p_arr[j] if j < len(p_arr) else None
        r_i = r_arr[j] if j < len(r_arr) else None
        nt = gt_counts.get(cid)

        # TP/FP/FN 与报告的 P/R 自洽（ultralytics metrics.py:931-932 同式）
        tp = fp = fn = None
        if p_i is not None and r_i is not None and nt:
            tp = int(round(r_i * nt))
            fp = int(round(tp / (p_i + 1e-16) - tp)) if p_i > 0 else None
            fn = int(nt - tp)

        per_class[name] = {
            "class_id": cid,
            "precision": p_i * 100 if p_i is not None else None,
            "recall": r_i * 100 if r_i is not None else None,
            "ap50": ap50[j] * 100 if j < len(ap50) else None,
            "ap50_95": ap[j] * 100 if j < len(ap) else None,
            "gt_count": nt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # PR 曲线：prec_values 行序 = unique_classes 序 = ap_class_index 序；
    # x = np.linspace(0,1,1000) 为召回轴
    px = as_list(getattr(box, "px", None))
    prec_values = getattr(box, "prec_values", None)
    pr_curves: dict[str, dict] = {}
    if prec_values is not None and len(px):
        pv = np.asarray(prec_values, dtype=float)
        for j, cid in enumerate(ap_class_index):
            name = ordered[cid] if cid < len(ordered) else f"class_{cid}"
            if j < pv.shape[0]:
                pr_curves[name] = {"recall": px, "precision": [float(x) for x in pv[j]]}

    out = {
        "arm": arm_key,
        "label": spec["label"],
        "arch": spec["arch"],
        "weights": str(weights),
        "head_class": head,
        "overall": {
            "mAP50": (as_scalar(box.map50) or 0.0) * 100,
            "mAP50_95": (as_scalar(box.map) or 0.0) * 100,
            "precision": (as_scalar(box.mp) or 0.0) * 100,   # 各类均值，不是 mAP
            "recall": (as_scalar(box.mr) or 0.0) * 100,
        },
        "per_class": per_class,
        "class_order": ordered,
        "pr_curves": pr_curves,
    }

    del metrics, box, model
    free_gpu()
    return out


# ---------------------------------------------------------------- 阶段 2：固定阈值交叉核对
def run_fixed(arm_key: str, spec: dict, class_order: list[str]) -> dict:
    """独立交叉核对：conf=0.25 / NMS iou=0.5 下按 IoU>=0.5 贪心匹配的 TP/FP/FN。

    source 必须传**目录字符串**，绝不能传路径 list —— 见下方 batch 陷阱说明。
    """
    import torch
    from PIL import Image

    model = YOLO(str(weights_of(spec)))
    n_expected = len(val_image_paths())

    res = {n: {"tp": 0, "fp": 0, "fn": 0} for n in class_order}
    n_seen = 0

    with torch.no_grad():
        # 【batch 陷阱 · 2026-09 实测】predict 的 source 若传 list[str]，check_source 会
        # 判为 from_img -> LoadPilAndNumpy，而它的 __next__ 是「整个列表一次性返回」
        # （loaders.py:546-551，注释原话 "loop only once as it's batch inference"），
        # 且 bs = len(列表)。于是 1550 张图被当成一个 batch 送进 preprocess，得到
        # (1550,3,640,640) float32 ≈ 15.2 GiB，8 GB 卡必然 OOM；batch= 参数对它无效
        # （LoadPilAndNumpy.__init__ 根本不收 batch）。
        # 传目录/glob 才会走 LoadImagesAndVideos，batch= 才生效。
        results = model.predict(
            source=str(val_split_dir()),
            imgsz=IMGSZ, conf=CONF_FIXED, iou=IOU_NMS,
            batch=BATCH_PRED, device=DEVICE, verbose=False, stream=True, max_det=300,
        )

        for i, r in enumerate(results):
            # 用 r.path（Results 自带源路径）定位，不依赖迭代顺序假设
            path = Path(r.path)
            n_seen += 1
            with Image.open(path) as im:
                w, h = im.size
            gts = load_gt_boxes(DATASET_DIR / "labels" / "val" / f"{path.stem}.txt", w, h)

            preds = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                preds = [(int(c), float(cf), tuple(float(x) for x in b))
                         for b, c, cf in zip(xyxy, cls, conf)]
            preds.sort(key=lambda t: -t[1])

            matched: set[int] = set()
            for pc, _cf, pb in preds:
                if pc >= len(class_order):
                    continue
                best_iou, best_j = 0.0, -1
                for j, (gc, gb) in enumerate(gts):
                    if gc != pc or j in matched:
                        continue
                    v = iou_xyxy(pb, gb)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= IOU_MATCH:
                    matched.add(best_j)
                    res[class_order[pc]]["tp"] += 1
                else:
                    res[class_order[pc]]["fp"] += 1

            for j, (gc, _gb) in enumerate(gts):
                if j in matched or gc >= len(class_order):
                    continue
                res[class_order[gc]]["fn"] += 1

            if n_seen % 250 == 0:
                print(f"    ..{n_seen}/{n_expected}", flush=True)

    del model
    free_gpu()

    if n_seen != n_expected:
        print(f"    [WARN] 实际推理 {n_seen} 张，目录图片 {n_expected} 张，不一致！", flush=True)

    return {"arm": arm_key, "label": spec["label"], "conf": CONF_FIXED,
            "iou": IOU_MATCH, "n_images": n_seen, "n_expected": n_expected, "per_class": res}


# ---------------------------------------------------------------- 阶段 3：合并
def run_merge() -> None:
    gt_counts = ground_truth_counts()
    all_arms, fixed = {}, {}
    for arm in ARMS:
        p = OUT_DIR / f"_official_{arm}.json"
        if not p.is_file():
            # 单因素消融臂缺失不阻断主表
            if arm in MAIN_ARMS:
                raise FileNotFoundError(f"缺少 {p}，请先跑 --stage official --arm {arm}")
            continue
        all_arms[arm] = json.loads(p.read_text(encoding="utf-8"))
        q = OUT_DIR / f"_fixed_{arm}.json"
        if q.is_file():
            fixed[arm] = json.loads(q.read_text(encoding="utf-8"))

    results = {a: all_arms[a] for a in MAIN_ARMS if a in all_arms}

    # 2x2 因子分解（mcsr × sa_tal），供 3.3 节机理归因
    factor = {}
    for a in FACTOR_ARMS:
        if a not in all_arms:
            continue
        r = all_arms[a]
        factor[a] = {
            "label": r["label"], "arch": r["arch"], "head_class": r["head_class"],
            "overall": r["overall"],
            "per_class": {k: {kk: vv for kk, vv in v.items()}
                          for k, v in r["per_class"].items()},
        }

    payload = {
        "config": {
            "dataset": DATASET_YAML,
            "imgsz": IMGSZ, "batch": BATCH, "batch_pred": BATCH_PRED, "device": DEVICE,
            "split": "val",
            "note": "P/R 为最大 F1 工作点；TP/FP/FN 与该 P/R 自洽（metrics.py:931-932）。",
        },
        "gt_counts": gt_counts,
        "results": results,
        "fixed_threshold_check": fixed,
        "factor_2x2": factor,
    }
    out = OUT_DIR / "perclass_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[save] {out}")
    print(f"[class order] {results['baseline']['class_order']}")
    print(f"[gt] {gt_counts}")
    print("\n===== 主对比 =====")
    for arm, r in results.items():
        ov = r["overall"]
        print(f"{r['label']:<14} mAP50={ov['mAP50']:.2f}  mAP50-95={ov['mAP50_95']:.2f}  head={r['head_class']}")
        for n, d in r["per_class"].items():
            print(f"  {n:<15} P={fmt(d['precision'])} R={fmt(d['recall'])} "
                  f"AP50={fmt(d['ap50'])} AP50-95={fmt(d['ap50_95'])} "
                  f"TP={d['tp']} FP={d['fp']} FN={d['fn']}")

    if len(factor) == 4:
        print("\n===== 2x2 因子分解 (mAP50-95 / 逐类 AP50-95) =====")
        cls = list(factor["baseline"]["per_class"])
        print(f"{'臂':<22}{'MCSR':>6}{'SA-TAL':>8}{'mAP50':>9}{'mAP50-95':>10}   "
              + "".join(f"{c[:13]:>15}" for c in cls))
        flag = {"baseline": (0, 0), "mcsr": (1, 0), "sa_tal": (0, 1), "msa": (1, 1)}
        for a in FACTOR_ARMS:
            f = factor[a]; ov = f["overall"]; m, s = flag[a]
            row = "".join(f"{f['per_class'][c]['ap50_95']:15.2f}" for c in cls)
            print(f"{f['label']:<22}{m:>6}{s:>8}{ov['mAP50']:9.2f}{ov['mAP50_95']:10.2f}   {row}")


# ---------------------------------------------------------------- entry
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["official", "fixed", "merge"])
    ap.add_argument("--arm", choices=list(ARMS))
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.stage == "merge":
        run_merge()
        return

    if not a.arm:
        raise SystemExit(f"--stage {a.stage} 需要 --arm")

    spec = ARMS[a.arm]
    print(f"[{a.stage}] {spec['label']}  <-  {spec['run']}/weights/best.pt", flush=True)

    if a.stage == "official":
        gt_counts = ground_truth_counts()
        print(f"[data] val GT 框数: {gt_counts}", flush=True)
        out = run_official(a.arm, spec, gt_counts)
        (OUT_DIR / f"_official_{a.arm}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        ov = out["overall"]
        print(f"  总体: mAP50={ov['mAP50']:.2f}  mAP50-95={ov['mAP50_95']:.2f}  head={out['head_class']}")
        for n, d in out["per_class"].items():
            print(f"  {n:<15} P={fmt(d['precision'])} R={fmt(d['recall'])} "
                  f"AP50={fmt(d['ap50'])} AP50-95={fmt(d['ap50_95'])} "
                  f"TP={d['tp']} FP={d['fp']} FN={d['fn']}")
        print(f"[save] {OUT_DIR / f'_official_{a.arm}.json'}")
    else:
        e = OUT_DIR / f"_official_{a.arm}.json"
        if not e.is_file():
            raise SystemExit(f"先跑 --stage official --arm {a.arm}（缺 {e}）")
        class_order = json.loads(e.read_text(encoding="utf-8"))["class_order"]
        out = run_fixed(a.arm, spec, class_order)
        (OUT_DIR / f"_fixed_{a.arm}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  固定阈值 conf={CONF_FIXED} IoU={IOU_MATCH} (n={out['n_images']}):")
        for n, d in out["per_class"].items():
            print(f"  {n:<15} TP={d['tp']:>5} FP={d['fp']:>5} FN={d['fn']:>5}")
        print(f"[save] {OUT_DIR / f'_fixed_{a.arm}.json'}")


if __name__ == "__main__":
    main()
