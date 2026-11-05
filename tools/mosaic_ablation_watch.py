# -*- coding: utf-8 -*-
"""Mosaic 消融双臂进度监控: 每 10 个 epoch 报一次, 两臂都到 100 才退出。

用法:
    python tools/mosaic_ablation_watch.py [poll_s]

stdout 每行 = 一个事件(供 Monitor 消费): 里程碑 / 单臂完成 / 全部完成 / 卡死。
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("on", "mosaic-on"), ("off", "mosaic-off")]
POLL = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MILESTONE = 10
TOTAL_EPOCHS = 100
# master.log 长时间无更新且两臂均未完成 => 视为异常退出。
STALL_S = 45 * 60


def read_rows(path: Path):
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
        return rows or None
    except Exception:
        return None


def master_all_done() -> bool:
    master = ROOT / "runs/mosaic_ablation_logs/master.log"
    if not master.is_file():
        return False
    try:
        # master.log 由 PowerShell Tee-Object 写出, 每字符间夹 NUL, 故先剥 NUL。
        text = master.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    except Exception:
        return False
    return "ALL_DONE" in text


def main() -> None:
    last = {key: 0 for key, _ in ARMS}
    done: set[str] = set()
    last_change = time.time()

    while True:
        progressed = False
        for key, label in ARMS:
            path = ROOT / f"runs/detect/mosaic_{key}_e100/results.csv"
            rows = read_rows(path)
            if not rows:
                continue
            try:
                epoch = int(float(rows[-1]["epoch"]))
            except Exception:
                continue

            if epoch >= last[key] + MILESTONE:
                last[key] = epoch
                progressed = True
                m50 = rows[-1].get("metrics/mAP50(B)", "?").strip()
                m5095 = rows[-1].get("metrics/mAP50-95(B)", "?").strip()
                print(f"{label} ep{epoch}/{TOTAL_EPOCHS} mAP50={m50} mAP50-95={m5095}", flush=True)

            if epoch >= TOTAL_EPOCHS and key not in done:
                done.add(key)
                progressed = True
                m50 = rows[-1].get("metrics/mAP50(B)", "?").strip()
                m5095 = rows[-1].get("metrics/mAP50-95(B)", "?").strip()
                print(f"DONE {label} epoch={epoch} mAP50={m50} mAP50-95={m5095}", flush=True)

        if progressed:
            last_change = time.time()

        if len(done) == len(ARMS):
            print("ALL_ARMS_DONE", flush=True)
            return

        if master_all_done() and len(done) < len(ARMS):
            missing = [lab for key, lab in ARMS if key not in done]
            print(f"ABNORMAL_EXIT missing={missing} master=ALL_DONE", flush=True)
            return

        if time.time() - last_change > STALL_S:
            print(
                f"STALL no progress for {STALL_S // 60}min "
                f"epochs={ {k: last[k] for k, _ in ARMS} }",
                flush=True,
            )
            return

        time.sleep(POLL)


if __name__ == "__main__":
    main()
