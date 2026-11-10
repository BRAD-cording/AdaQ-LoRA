#!/usr/bin/env python3
"""一键核对 results_v2 完成度与 summary.json 可用性。只读，exit 0=全部达标。"""
import glob, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_v2")
METHODS = ("qlora", "dora", "adalora", "adaq")
SEEDS = (42, 3407, 2025)
TASKS = {"mmlu": ("mmlu",), "gsm8k": ("gsm8k",),
         "safety": ("truthfulqa_mc1", "truthfulqa_mc2", "ifeval")}
def have(run_dir):
    keys = set()
    for f in glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True):
        try:
            keys |= set((json.load(open(f)).get("results") or {}).keys())
        except Exception:
            pass
    return keys
missing = []
for m in METHODS:
    for s in SEEDS:
        r = f"{m}_seed{s}"; d = os.path.join(OUT, r); k = have(d)
        for marker, need in TASKS.items():
            if not all(t in k for t in need):
                missing.append(f"{r}/{marker}")
print(f"run 完整度: {12*3-len(missing)}/36 任务组齐全")
for x in missing: print("  MISSING", x)
sp = os.path.join(OUT, "summary.json")
ok = os.path.isfile(sp)
if ok:
    d = json.load(open(sp)); a = d.get("_audit", {})
    print("summary._audit.mismatch_task_count =", a.get("mismatch_task_count"))
    print("summary._audit.n_runs =", a.get("n_runs"), " n_files =", a.get("n_files"))
    bad = []
    for m in METHODS:
        for t in ("mmlu/acc", "gsm8k/exact_match,strict-match",
                  "truthfulqa_mc1/acc", "truthfulqa_mc2/acc",
                  "ifeval/inst_level_strict_acc"):
            n = (d.get(m, {}).get(t) or {}).get("n_seeds", 0)
            if n != 3: bad.append(f"{m}/{t}=n{n}")
    if a.get("mismatch_task_count") or bad:
        ok = False; print("  不达标:", bad or a.get("sample_len_mismatch"))
else:
    print("summary.json 尚未生成")
print("VERDICT:", "PASS 可交付" if ok and not missing else "INCOMPLETE 继续等待/补跑")
sys.exit(0 if ok and not missing else 1)
