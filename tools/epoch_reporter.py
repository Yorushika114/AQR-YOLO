from __future__ import annotations

"""Read the newest completed epoch from a run's results.csv and print its metrics.

Used by tools/epoch_monitor.sh for periodic (every 10 epochs) progress reports.
Prints pipe-delimited:  epoch|time|train/box_loss|train/cls_loss|train/dfl_loss|
                        recall(B)|mAP50(B)|mAP50-95(B)
Parses by header name so it is robust to column order changes. Empty output if
the CSV is mid-write or has no data yet.

Note: EP decision (milestone detection) is left to the shell monitor; this script
only reports the current newest epoch.
"""
import csv
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: epoch_reporter.py <results.csv>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            rows = [r for r in csv.DictReader(fh) if (r.get("epoch") or "").strip()]
    except FileNotFoundError:
        return 0  # training dir not created yet -> report nothing

    if not rows:
        return 0

    last = rows[-1]
    col = lambda k: (last.get(k) or "").strip()
    try:
        ep = int(float(col("epoch")))
    except ValueError:
        return 0

    fields = [
        str(ep), col("time"), col("train/box_loss"), col("train/cls_loss"),
        col("train/dfl_loss"), col("metrics/precision(B)"), col("metrics/recall(B)"),
        col("metrics/mAP50(B)"), col("metrics/mAP50-95(B)"),
    ]
    print("|".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
