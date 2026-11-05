#!/usr/bin/env bash
# usage: epoch_monitor.sh <results.csv> <label> [poll_secs]
# Emits one line whenever the newest completed epoch reaches a multiple of 10.
# Each emitted line is used as a 10-epoch progress notification.
set -u
CSV="$1"
LABEL="$2"
POLL="${3:-40}"
PY="D:/anaconda/envs/test/python.exe"
REPORTER="D:/个人文件整理/02_开发项目/ultralytics-8.4.94/tools/epoch_reporter.py"
last=0

while true; do
  if [ -f "$CSV" ]; then
    out=$("$PY" "$REPORTER" "$CSV" 2>/dev/null)
    if [ -n "$out" ]; then
      ep=${out%%|*}
      if [ "$ep" -gt "$last" ] && [ $((ep % 10)) -eq 0 ]; then
        rest=${out#*|}
        echo "ep${ep} [${LABEL}] ${rest}"
        last=$ep
      fi
    fi
  fi
  sleep "$POLL"
done
