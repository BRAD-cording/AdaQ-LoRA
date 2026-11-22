#!/usr/bin/env python
"""Collect per-seed GSM8K/MMLU results for the INT4-RTN runs into CSV.

Scans results_v2_rtn/<method>_s<seed>/**/results_*.json (latest file per run),
extracts gsm8k strict/flexible exact-match and the aggregate mmlu accuracy,
and prints CSV: quantizer,method,seed,gsm8k_strict,gsm8k_flex,mmlu
"""
import glob
import json
import os
import re
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA/results_v2_rtn"
rows = []

for run_dir in sorted(glob.glob(os.path.join(ROOT, "*_s*"))):
    name = os.path.basename(run_dir)
    m = re.match(r"^(adaq|qlora|adalora|dora)_s(\d+)$", name)
    if not m:
        continue
    method, seed = m.group(1), m.group(2)
    files = glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)
    if not files:
        continue
    latest = max(files, key=os.path.getmtime)
    try:
        d = json.load(open(latest))
    except Exception as e:  # noqa: BLE001
        print("skip %s: %s" % (latest, e), file=sys.stderr)
        continue
    r = d.get("results", {})
    gsm = r.get("gsm8k", {})
    mmlu = r.get("mmlu", {})
    rows.append({
        "quantizer": "INT4-RTN",
        "method": method,
        "seed": int(seed),
        "gsm8k_strict": gsm.get("exact_match,strict-match"),
        "gsm8k_flex": gsm.get("exact_match,flexible-extract"),
        "mmlu": mmlu.get("acc,none"),
        "n_fewshot": (list(d.get("configs", {}).values()) or [{}])[0].get("num_fewshot"),
        "file": os.path.relpath(latest, ROOT),
    })

rows.sort(key=lambda x: (x["method"], x["seed"]))
print("quantizer,method,seed,gsm8k_strict,gsm8k_flex,mmlu,n_fewshot,file")
for x in rows:
    print("%s,%s,%d,%s,%s,%s,%s,%s" % (
        x["quantizer"], x["method"], x["seed"],
        "" if x["gsm8k_strict"] is None else "%.6f" % x["gsm8k_strict"],
        "" if x["gsm8k_flex"] is None else "%.6f" % x["gsm8k_flex"],
        "" if x["mmlu"] is None else "%.6f" % x["mmlu"],
        x["n_fewshot"], x["file"]))
print("# runs=%d" % len(rows), file=sys.stderr)
