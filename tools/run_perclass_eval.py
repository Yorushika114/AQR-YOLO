# -*- coding: utf-8 -*-
"""逐类评测的调度器：把 4 个 (阶段 × 模型) 各起一个**独立进程**跑，最后合并。

为什么不用一个进程跑完：本机 RTX 4060 Laptop 只有 8 GB。同一进程里先 val 再
predict 会把显存耗干（Windows WDDM 回落到共享内存 -> forward 时 OOM）。
每个阶段独立进程 = 每次全新 CUDA 上下文，退出即回收。

日志打在 stdout，配合 PowerShell Start-Process 脱离启动 + 监控。

用法:
    python tools/run_perclass_eval.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval_perclass_defect.py"

STEPS = [
    ("official", "baseline"),
    ("official", "msa"),
    ("fixed", "baseline"),
    ("fixed", "msa"),
    ("merge", None),
]


def main() -> int:
    py = sys.executable
    t0 = time.time()
    for i, (stage, arm) in enumerate(STEPS, 1):
        cmd = [py, str(EVAL), "--stage", stage]
        if arm:
            cmd += ["--arm", arm]
        tag = f"{stage}/{arm}" if arm else stage
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {tag}   (t+{time.time() - t0:.0f}s)\n{'=' * 70}",
              flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\n[FAIL] {tag} 退出码 {r.returncode}", flush=True)
            return r.returncode
    print(f"\n[ALL DONE] 用时 {time.time() - t0:.0f}s", flush=True)
    print("[save] C:/Users/23223/Desktop/新论文数据保存/逐类检测性能对比试验/perclass_results.json",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
