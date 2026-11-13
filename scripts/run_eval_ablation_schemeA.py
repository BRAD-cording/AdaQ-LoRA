#!/usr/bin/env python3
"""评测方案A同配置消融 6 策略（MMLU 5-shot + GSM8K 4-shot，全量）。

与 run_eval_ablation.py 唯一区别：adapter 路径 = outputs/ablation_adapters_schemeA/<strategy>，
结果落 results_v2_ablation_schemeA。其余口径完全一致（HF_HUB_OFFLINE=1 等）。

环境变量:
  OUT_DIR    默认 <root>/results_v2_ablation_schemeA
  STRATEGIES 逗号分隔，默认 "error_only,importance_only,product,uniform,equal_weight,random"
  GPUS       逗号分隔，默认 "0,1"
  LIMIT      每任务样本上限，留空=全量
  FORCE      1=重跑
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_eval_ablation as base

base.OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results_v2_ablation_schemeA"))
base.ADAPTER_DIR = os.path.join(ROOT, "outputs", "ablation_adapters_schemeA")

if __name__ == "__main__":
    sys.exit(base.main())
