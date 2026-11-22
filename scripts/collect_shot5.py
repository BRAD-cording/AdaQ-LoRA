"""Collect the 5-shot GSM8K re-scores (results_shot5) into one CSV.

Directory names carry the adapter-root slug, e.g. phase3_main_adaq_s42, because
NF4 and INT8 share run names.
"""
import glob
import json
import os
import re

ROOT = "/home/ubuntu/AdaQ-LoRA/results_shot5"
rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "*_s*"))):
    name = os.path.basename(d)
    m = re.match(r"^(phase3_main|phase3_int8)_(adaq|qlora|adalora|dora)_s(\d+)$", name)
    if not m:
        continue
    files = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    if not files:
        continue
    p = max(files, key=os.path.getmtime)
    dd = json.load(open(p))
    r = dd.get("results", {}).get("gsm8k", {})
    cfg = list(dd.get("configs", {}).values()) or [{}]
    rows.append((m.group(1), m.group(2), int(m.group(3)),
                 r.get("exact_match,strict-match"), r.get("exact_match,flexible-extract"),
                 cfg[0].get("num_fewshot")))

print("quantizer,method,seed,gsm8k_strict,gsm8k_flex,fewshot")
for q, meth, s, st, fx, fs in rows:
    print("%s,%s,%d,%.6f,%.6f,%s" % (q, meth, s, st, fx, fs))
print("# rows=%d" % len(rows))
