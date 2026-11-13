#!/usr/bin/env python3
"""P1-D4: 方案A同配置消融多种子评测（继承 run_eval_ablation_schemeA，seed 化 adapter 目录）。

用法: SEED=3407 python scripts/run_eval_ablation_schemeA_multiseed.py
adapter 路径 = outputs/ablation_adapters_schemeA_s<SEED>/<strategy>
结果落 results_v2_ablation_schemeA_s<SEED>/<strategy>
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.environ.get("SEED")
if not SEED:
    print("ERROR: SEED env required (e.g. 3407)", file=sys.stderr)
    sys.exit(1)

# 直接 import base 模块后改全局变量，再调 main
sys.path.insert(0, ROOT + "/scripts")
import run_eval_ablation as base

base.ADAPTER_DIR = os.path.join(ROOT, "outputs", "ablation_adapters_schemeA_s" + SEED)
base.OUT = os.path.join(ROOT, "results_v2_ablation_schemeA_s" + SEED)

print(f"[multiseed] SEED={SEED} ADAPTER_DIR={base.ADAPTER_DIR} OUT={base.OUT}")
sys.exit(base.main())
