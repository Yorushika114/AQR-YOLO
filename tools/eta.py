from __future__ import annotations

"""Estimate remaining training time and wall-clock completion time.

Reads the newest completed epoch from a run's results.csv and its cumulative
`time` column, extrapolates the per-epoch rate, and prints:
    remaining_duration_h  |  HH:MM (wall-clock completion estimate)
Prints "pending" when there is no measurable progress yet.

usage: eta.py <results.csv> [total_epochs]
"""
import csv
import sys
from datetime import datetime, timedelta


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: eta.py <results.csv> [total_epochs]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            rows = [r for r in csv.DictReader(fh) if (r.get("epoch") or "").strip()]
    except FileNotFoundError:
        print("pending")
        return 0
    if not rows:
        print("pending")
        return 0

    last = rows[-1]
    try:
        ep = int(float(last["epoch"]))
        t = float(last["time"])
    except (KeyError, ValueError):
        print("pending")
        return 0

    if ep < 1 or t <= 0:
        print("pending")
        return 0

    rate = t / ep                      # seconds per epoch (running average)
    remaining_s = max(0.0, (total - ep) * rate)
    eta = datetime.now() + timedelta(seconds=remaining_s)
    return f"{remaining_s/3600:.1f} h  ~ 预计 {eta:%H:%M} 完成"


if __name__ == "__main__":
    raise SystemExit(main())
