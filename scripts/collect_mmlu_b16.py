#!/usr/bin/env python
"""Collect the uniform-protocol MMLU accuracies for the three INT4-RTN seeds.

Prints CSV: method,seed,mmlu_acc
compared later against the values used in the manuscript
(adaq 0.6107/0.6102/0.6089, qlora 0.6097/0.6079/0.6087 on seeds 2025/3407/42).
"""
import glob
import json
import os
import re
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA/results_v2_rtn_b16"
rows = []
for run in sorted(glob.glob(os.path.join(ROOT, "*_s*"))):
    m = re.match(r"^(adaq|qlora|adalora|dora)_s(\d+)$", os.path.basename(run))
    if not m:
        continue
    files = glob.glob(os.path.join(run, "mmlu", "**", "results_*.json"), recursive=True)
    if not files:
        continue
    latest = max(files, key=os.path.getmtime)
    d = json.load(open(latest))
    acc = d.get("results", {}).get("mmlu", {}).get("acc,none")
    cfg = list(d.get("configs", {}).values()) or [{}]
    print("%s,%d,%.6f,%s,%s" % (m.group(1), int(m.group(2)), acc,
                                cfg[0].get("num_fewshot"),
                                os.path.relpath(latest, ROOT)))
    rows.append(acc)
print("# runs=%d" % len(rows), file=sys.stderr)
