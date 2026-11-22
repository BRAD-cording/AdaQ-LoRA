#!/usr/bin/env python
"""Collect the uniform-protocol INT4-RTN results (results_v2_rtn_b16).

Prints CSV: method,seed,gsm8k_strict,gsm8k_flex,n_fewshot,batch,file
Also cross-checks the same (method, seed) against the legacy results dir so the
effect of the batch-size change can be verified run by run.
"""
import glob
import json
import os
import re
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA"
NEW = os.path.join(ROOT, "results_v2_rtn_b16")


def load_latest(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    try:
        d = json.load(open(latest))
    except Exception:  # noqa: BLE001
        return None
    cfg = list(d.get("configs", {}).values()) or [{}]
    gsm = d.get("results", {}).get("gsm8k", {})
    return {
        "strict": gsm.get("exact_match,strict-match"),
        "flex": gsm.get("exact_match,flexible-extract"),
        "fewshot": cfg[0].get("num_fewshot"),
        "batch": cfg[0].get("batch_size"),
        "file": os.path.relpath(latest, ROOT),
    }


print("method,seed,gsm8k_strict,gsm8k_flex,n_fewshot,batch,legacy_strict,legacy_fewshot,file")
rows = 0
for run in sorted(glob.glob(os.path.join(NEW, "*_s*"))):
    m = re.match(r"^(adaq|qlora|adalora|dora)_s(\d+)$", os.path.basename(run))
    if not m:
        continue
    method, seed = m.group(1), int(m.group(2))
    new = load_latest(os.path.join(run, "gsm8k", "**", "results_*.json"))
    if not new:
        continue
    legacy = load_latest(os.path.join(ROOT, "results_v2_rtn",
                                      "%s_s%d" % (method, seed), "**", "results_*.json"))
    print("%s,%d,%.6f,%.6f,%s,%s,%s,%s,%s" % (
        method, seed, new["strict"], new["flex"], new["fewshot"], new["batch"],
        "" if not legacy else "%.6f" % legacy["strict"],
        "" if not legacy else legacy["fewshot"],
        new["file"]))
    rows += 1
print("# runs=%d" % rows, file=sys.stderr)
