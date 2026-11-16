#!/usr/bin/env python3
"""Phase-3 评测 wrapper：复用 run_eval_ablation 的调度/校验/分卡逻辑，按 PHASE 覆写
ADAPTER_DIR / OUT / STRATEGIES / SPECS。

环境变量:
  PHASE   D|B|E|C|A（必需）
  TASKS   逗号分隔，默认 mmlu,gsm8k；D 臂只评 gsm8k
  GPUS    默认 0,1
  LIMIT   留空=全量
  FORCE   1=重跑
"""
import os
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PHASE = os.environ.get("PHASE")
if not PHASE:
    print("need PHASE=D|B|E|C|A", file=sys.stderr)
    sys.exit(2)
TASKS = os.environ.get("TASKS", "mmlu,gsm8k")
SEEDS3 = [42, 3407, 2025]
SEEDS2 = [42, 3407]

# phase -> (adapter_dir 相对 ROOT, out_dir 相对 ROOT, run 子目录名列表)
if PHASE == "D":
    adir = "outputs/phase3_noise"
    odir = "results_v2_phase3_noise"
    runs = [f"product_s{s}_r{r}" for s in SEEDS3 for r in (0, 1, 2)]
elif PHASE == "B":
    adir = "outputs/phase3_main"
    odir = "results_v2_phase3_main"
    runs = [f"adalora_s{s}" for s in SEEDS3]
elif PHASE == "E":
    adir = "outputs/phase3_baselines"
    odir = "results_v2_phase3_baselines"
    runs = [f"{m}_s{s}" for s in SEEDS3 for m in ("loftq", "qlorapm")]
elif PHASE == "C":
    adir = "outputs/phase3_rcurve"
    odir = "results_v2_phase3_rcurve"
    runs = [f"{m}_R{R}_s{s}" for R in (128, 512, 1024) for s in SEEDS2 for m in ("adaq", "qlora")]
elif PHASE == "A":
    adir = "outputs/phase3_main"
    odir = "results_v2_phase3_main"
    runs = [f"{m}_s{s}" for s in SEEDS3 for m in ("qlora", "dora", "adaq", "adalora")]
else:
    print(f"unknown PHASE={PHASE}", file=sys.stderr)
    sys.exit(2)

import run_eval_ablation as base  # noqa: E402

base.ADAPTER_DIR = os.path.join(ROOT, adir)
base.OUT = os.path.join(ROOT, odir)
base.STRATEGIES = runs
want = {t.strip() for t in TASKS.split(",") if t.strip()}
base.SPECS = {k: v for k, v in base.SPECS.items() if k in want}
if not base.SPECS:
    print(f"TASKS={TASKS} 过滤后 SPECS 为空", file=sys.stderr)
    sys.exit(2)

print(f"[phase3-eval] PHASE={PHASE} ADAPTER_DIR={base.ADAPTER_DIR} OUT={base.OUT} "
      f"TASKS={sorted(base.SPECS)} runs={len(runs)}", flush=True)
if os.environ.get("DRYRUN"):
    print("[DRYRUN] STRATEGIES=", base.STRATEGIES)
    print("[DRYRUN] SPECS keys=", sorted(base.SPECS))
    present = [r for r in runs if os.path.isfile(os.path.join(base.ADAPTER_DIR, r, "adapter_config.json"))]
    print(f"[DRYRUN] adapters present {len(present)}/{len(runs)}: {present}")
    sys.exit(0)
sys.exit(base.main())
