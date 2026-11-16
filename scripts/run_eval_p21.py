#!/usr/bin/env python3
"""P2-1 评测 wrapper：复用 run_eval_ablation 的调度/校验/分卡逻辑。
4 臂 × 3 seed = 12 run，ADAPTER_DIR=outputs/p21, OUT=results_v2_p21。
评测基座 BF16（与主表/消融一致），seed42 固定。

环境变量:
  TASKS   逗号分隔，默认 mmlu,gsm8k（主判据 gsm8k-flex；mmlu 仅报告）
  GPUS    默认 0,1
  LIMIT   留空=全量
  FORCE   1=重跑
  DRYRUN  1=只打印 present adapters 后 exit0
"""
import os
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TASKS = os.environ.get("TASKS", "mmlu,gsm8k")
SEEDS3 = [42, 3407, 2025]
ARMS = ["cost_norm", "raw_score", "uniform", "uniform_pm"]

adir = "outputs/p21"
odir = "results_v2_p21"
runs = [f"{arm}_s{s}" for arm in ARMS for s in SEEDS3]

import run_eval_ablation as base  # noqa: E402

base.ADAPTER_DIR = os.path.join(ROOT, adir)
base.OUT = os.path.join(ROOT, odir)
base.STRATEGIES = runs
want = {t.strip() for t in TASKS.split(",") if t.strip()}
base.SPECS = {k: v for k, v in base.SPECS.items() if k in want}
if not base.SPECS:
    print(f"TASKS={TASKS} 过滤后 SPECS 为空", file=sys.stderr)
    sys.exit(2)

print(f"[p21-eval] ADAPTER_DIR={base.ADAPTER_DIR} OUT={base.OUT} "
      f"TASKS={sorted(base.SPECS)} runs={len(runs)}", flush=True)
if os.environ.get("DRYRUN"):
    print("[DRYRUN] STRATEGIES=", base.STRATEGIES)
    print("[DRYRUN] SPECS keys=", sorted(base.SPECS))
    present = [r for r in runs if os.path.isfile(os.path.join(base.ADAPTER_DIR, r, "adapter_config.json"))]
    print(f"[DRYRUN] adapters present {len(present)}/{len(runs)}: {present}")
    sys.exit(0)
sys.exit(base.main())
