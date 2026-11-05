from __future__ import annotations

"""Resilient per-epoch watcher for a training run's results.csv.

Use as a harness Monitor (persistent). Emits one line per event:
  - ep<k> [label] <time>|<box>|<cls>|<dfl>|<prec>|<rec>|<mAP50>|<mAP50-95>   on each 10-epoch milestone
  - DONE [label] epoch=<k> mAP50-95=<v>                     once training reaches <total>: exits (end of watch)
  - STALL [label] ...                                       when no progress for >= stall_min minutes
Works even when the trainer was launched fully detached (no harness task notification).

usage: progress_watch.py <results.csv> <label> [total_epochs] [poll_s] [stall_min]
"""
import csv
import os
import sys
import time

csv_path = sys.argv[1]
label = sys.argv[2]
total = int(sys.argv[3]) if len(sys.argv) > 3 else 100
poll = int(sys.argv[4]) if len(sys.argv) > 4 else 40
stall_min = int(sys.argv[5]) if len(sys.argv) > 5 else 8

start = time.time()
last_ep = -1
last_reported = 0
last_progress = time.time()
stall_announced_at = 0.0


def emit(line: str) -> None:
    print(line)


while True:
    progressed = False
    if os.path.exists(csv_path):
        try:
            with open(csv_path, encoding="utf-8-sig", errors="replace") as fh:
                rows = [r for r in csv.DictReader(fh) if (r.get("epoch") or "").strip()]
        except Exception:
            rows = []

        if rows:
            last = rows[-1]
            try:
                ep = int(float(last["epoch"]))
            except ValueError:
                ep = -1
            if ep != last_ep:
                last_ep = ep
                last_progress = time.time()
                stall_announced_at = 0.0
                progressed = True

            if ep >= total:
                emit(f"DONE [{label}] epoch={ep} mAP50-95={last.get('metrics/mAP50-95(B)', '')}")
                sys.exit(0)

            if ep > 0 and ep % 10 == 0 and ep != last_reported:
                emit(
                    f"ep{ep} [{label}] "
                    + "|".join([
                        last.get("time", ""),
                        last.get("train/box_loss", ""),
                        last.get("train/cls_loss", ""),
                        last.get("train/dfl_loss", ""),
                        last.get("metrics/precision(B)", ""),
                        last.get("metrics/recall(B)", ""),
                        last.get("metrics/mAP50(B)", ""),
                        last.get("metrics/mAP50-95(B)", ""),
                    ])
                )
                last_reported = ep
        else:
            progressed = bool(os.path.getsize(csv_path) > 0)

    # Stall detection: no epoch advanced (and no data at all) for too long.
    idle = time.time() - last_progress
    if not progressed and idle > stall_min * 60:
        if time.time() - stall_announced_at > stall_min * 60:
            emit(f"STALL [{label}] no progress for {int(idle // 60)} min (last_ep={last_ep})")
            stall_announced_at = time.time()

    time.sleep(poll)
