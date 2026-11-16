#!/usr/bin/env python3
"""Phase-3 聚合：按 PHASE 读 results_v2_phase3_<phase>/<run>/results_*.json，
抽取 mmlu(acc,none) / gsm8k(strict,flex)，按臂分组算 mean±std(n)，写 summary.json + REPORT.md。

用法: PHASE=D|B|E|C|A python scripts/aggregate_phase3.py
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA"
PHASE = os.environ.get("PHASE")
if not PHASE:
    print("need PHASE", file=sys.stderr)
    sys.exit(2)

SEEDS3 = [42, 3407, 2025]
SEEDS2 = [42, 3407]

# phase -> (results 相对目录, run 名 -> 组键 的解析)
def group_key(phase, run):
    if phase == "D":
        return "product"                      # 9 run 同配置 → 噪声底
    if phase in ("B", "A"):
        return run.rsplit("_s", 1)[0]         # {method}_s{seed} -> method
    if phase == "E":
        return run.rsplit("_s", 1)[0]         # loftq / qlorapm
    if phase == "C":
        # {method}_R{R}_s{seed} -> {method}_R{R}
        return run.rsplit("_s", 1)[0]
    return run


def runs_for(phase):
    if phase == "D":
        return [f"product_s{s}_r{r}" for s in SEEDS3 for r in (0, 1, 2)]
    if phase == "B":
        return [f"adalora_s{s}" for s in SEEDS3]
    if phase == "E":
        return [f"{m}_s{s}" for s in SEEDS3 for m in ("loftq", "qlorapm")]
    if phase == "C":
        return [f"{m}_R{R}_s{s}" for R in (128, 512, 1024) for s in SEEDS2 for m in ("adaq", "qlora")]
    if phase == "A":
        return [f"{m}_s{s}" for s in SEEDS3 for m in ("qlora", "dora", "adaq", "adalora")]
    return []


def read_run(run_dir):
    mmlu = gstrict = gflex = None
    for f in glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True):
        try:
            j = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        res = j.get("results", {}) or {}
        if "mmlu" in res and "acc,none" in res["mmlu"]:
            mmlu = res["mmlu"]["acc,none"]
        if "gsm8k" in res:
            g = res["gsm8k"]
            if "exact_match,strict-match" in g:
                gstrict = g["exact_match,strict-match"]
            if "exact_match,flexible-extract" in g:
                gflex = g["exact_match,flexible-extract"]
    return {"mmlu": mmlu, "gsm8k_strict": gstrict, "gsm8k_flex": gflex}


# phase -> results 目录，必须与 run_eval_phase3.py 的 OUT 完全一致
RESDIR_MAP = {
    "D": "results_v2_phase3_noise",
    "B": "results_v2_phase3_main",
    "E": "results_v2_phase3_baselines",
    "C": "results_v2_phase3_rcurve",
    "A": "results_v2_phase3_main",
}
if PHASE not in RESDIR_MAP:
    print(f"unknown PHASE={PHASE}", file=sys.stderr)
    sys.exit(2)
resdir = os.path.join(ROOT, RESDIR_MAP[PHASE])

per_run = {}
groups = {}
for run in runs_for(PHASE):
    rd = os.path.join(resdir, run)
    m = read_run(rd) if os.path.isdir(rd) else {"mmlu": None, "gsm8k_strict": None, "gsm8k_flex": None}
    per_run[run] = m
    gk = group_key(PHASE, run)
    groups.setdefault(gk, []).append((run, m))


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    out = {"n": len(vals), "mean": round(st.mean(vals), 5)}
    if len(vals) > 1:
        out["std"] = round(st.stdev(vals), 5)
        out["min"] = round(min(vals), 5)
        out["max"] = round(max(vals), 5)
        out["range"] = round(max(vals) - min(vals), 5)
    return out


summary = {"phase": PHASE, "results_dir": os.path.relpath(resdir, ROOT), "groups": {}, "per_run": per_run}
for gk, items in sorted(groups.items()):
    summary["groups"][gk] = {
        "runs": [r for r, _ in items],
        "mmlu": agg([m["mmlu"] for _, m in items]),
        "gsm8k_strict": agg([m["gsm8k_strict"] for _, m in items]),
        "gsm8k_flex": agg([m["gsm8k_flex"] for _, m in items]),
    }

os.makedirs(resdir, exist_ok=True)
with open(os.path.join(resdir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---- REPORT.md ----
L = []
L.append(f"# Phase-3 {PHASE} 聚合报告")
L.append("")
L.append(f"结果目录：`{os.path.relpath(resdir, ROOT)}`　run 数：{len(per_run)}　组数：{len(groups)}")
L.append("")


def fmt(a):
    if not a:
        return "—"
    s = f"{a['mean']:.4f}"
    if "std" in a:
        s += f" ±{a['std']:.4f}"
    s += f" (n={a['n']}"
    if "range" in a:
        s += f", range {a['range']:.4f}"
    s += ")"
    return s


L.append("| 组 | MMLU acc | GSM8K strict | GSM8K flex |")
L.append("|---|---|---|---|")
for gk in sorted(groups):
    g = summary["groups"][gk]
    L.append(f"| {gk} | {fmt(g['mmlu'])} | {fmt(g['gsm8k_strict'])} | {fmt(g['gsm8k_flex'])} |")
L.append("")

if PHASE == "D":
    fl = [m["gsm8k_flex"] for m in per_run.values() if m["gsm8k_flex"] is not None]
    sr = [m["gsm8k_strict"] for m in per_run.values() if m["gsm8k_strict"] is not None]
    L.append("## 噪声底（同 product 配置 9 次独立训练）")
    if fl:
        L.append(f"- GSM8K flex：mean {st.mean(fl):.4f}，std {st.stdev(fl) if len(fl)>1 else 0:.4f}，"
                 f"min {min(fl):.4f}，max {max(fl):.4f}，**极差 {max(fl)-min(fl):.4f}**（n={len(fl)}）")
    if sr:
        L.append(f"- GSM8K strict：mean {st.mean(sr):.4f}，极差 {max(sr)-min(sr):.4f}（n={len(sr)}）")
    L.append("- 判定：任何方法间 GSM8K 差异 ≤ 此极差者，不可主张方向性优劣（F8 噪声底）。")
    # 同种子跨 repeat（非确定性）vs 跨种子
    by_seed = {}
    for run, m in per_run.items():
        if m["gsm8k_flex"] is None:
            continue
        sd = run.split("_s")[1].split("_")[0]
        by_seed.setdefault(sd, []).append(m["gsm8k_flex"])
    L.append("")
    L.append("### 方差分解（flex）")
    for sd, vs in sorted(by_seed.items()):
        if len(vs) > 1:
            L.append(f"- seed{sd} 内 3 repeat 极差 {max(vs)-min(vs):.4f}（非确定性分量）")
    seed_means = [st.mean(v) for v in by_seed.values() if v]
    if len(seed_means) > 1:
        L.append(f"- 跨 seed 均值极差 {max(seed_means)-min(seed_means):.4f}（种子分量）")
L.append("")

report = "\n".join(L)
rp = os.path.join(resdir, f"REPORT_phase3_{PHASE}.md")
with open(rp, "w") as f:
    f.write(report + "\n")
print(report)
print(f"\n[aggregate] wrote {os.path.relpath(rp, ROOT)} + summary.json")
