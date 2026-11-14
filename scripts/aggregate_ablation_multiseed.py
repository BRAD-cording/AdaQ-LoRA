#!/usr/bin/env python3
"""汇总 P1-D4 方案A多种子消融（results_v2_ablation_schemeA_s<seed>/<strategy>/**/results*.json）。

口径与 scripts/aggregate_results.py 一致：复用其 norm_keys/pick_metric/find_result_files，
但目录布局是 s<seed> 而非 _seed<seed>，故自带解析。

输出:
  results_v2_ablation_multiseed/summary.json
  results_v2_ablation_multiseed/REPORT.md

诚实口径: n=2 种子只报描述性 trend，不做显著性检验（n<5 无统计意义）。
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import aggregate_results as agg

SEEDS = [3407, 2025]
STRATEGIES = ["error_only", "importance_only", "product", "uniform", "equal_weight", "random"]
TASKS = ["mmlu", "gsm8k"]


def seed_dir(seed):
    return os.path.join(ROOT, "results_v2_ablation_schemeA_s%d" % seed)


def collect(seed):
    """{(strategy, task): (value, path, metric_name, sample_len)} 取每 (run,task) 最新一份。"""
    out = {}
    for strat in STRATEGIES:
        d = os.path.join(seed_dir(seed), strat)
        files = agg.find_result_files(d) if os.path.isdir(d) else []
        latest = {}
        for f in files:
            try:
                data = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            for task, metrics in data.get("results", {}).items():
                if task not in TASKS or not isinstance(metrics, dict):
                    continue
                m = agg.norm_keys(metrics)
                name, val = agg.pick_metric(task, m)
                if val is None:
                    continue
                key = (strat, task)
                row = (os.path.getmtime(f), f, val, m.get("sample_len"), name)
                if key not in latest or row[0] > latest[key][0]:
                    latest[key] = row
        for key, row in latest.items():
            # row = (mtime, file, value, sample_len, metric_name) -> 按语义重排
            _mtime, f, val, sample_len, name = row
            out[key] = (val, f, name, sample_len)
    return out


def main():
    per = {s: collect(s) for s in SEEDS}
    summary = {
        "seeds": SEEDS,
        "strategies": STRATEGIES,
        "tasks": TASKS,
        "per_seed": {},
        "aggregate": {},
        "_audit": {"warnings": [], "n_seeds": len(SEEDS)},
    }
    for s in SEEDS:
        summary["per_seed"][str(s)] = {
            "%s/%s" % (k[0], k[1]): {
                "value": v[0],
                "file": v[1],
                "metric": v[2],
                "sample_len": v[3],
            }
            for k, v in per[s].items()
        }
    for strat in STRATEGIES:
        for task in TASKS:
            vals = [per[s][(strat, task)][0] for s in SEEDS if (strat, task) in per[s]]
            if not vals:
                continue
            entry = {
                "mean": round(statistics.mean(vals), 4),
                "n_seeds": len(vals),
                "values": [round(v, 4) for v in vals],
            }
            if len(vals) > 1:
                entry["std"] = round(statistics.stdev(vals), 4)
                entry["range"] = round(max(vals) - min(vals), 4)
            summary["aggregate"]["%s/%s" % (strat, task)] = entry

    for task in TASKS:
        lens = {
            per[s][(st, task)][3]
            for s in SEEDS
            for st in STRATEGIES
            if (st, task) in per[s]
        }
        lens.discard(None)
        if len(lens) > 1:
            summary["_audit"]["warnings"].append(
                "%s: sample_len 跨策略不一致 -> %s" % (task, sorted(lens))
            )

    outdir = os.path.join(ROOT, "results_v2_ablation_multiseed")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# P1-D4 方案A多种子消融汇总",
        "",
        "- 种子: %s  (n=%d，仅描述性 trend，不做显著性检验)" % (SEEDS, len(SEEDS)),
        "",
        "## 主表 (mean across seeds)",
        "",
        "| strategy | MMLU | GSM8K |",
        "|---|---|---|",
    ]
    for strat in STRATEGIES:
        cells = []
        for task in TASKS:
            e = summary["aggregate"].get("%s/%s" % (strat, task))
            cells.append(
                "%.4f±%.4f" % (e["mean"], e.get("std", 0.0)) if e else "—"
            )
        lines.append("| %s | %s | %s |" % (strat, cells[0], cells[1]))
    lines += ["", "## 逐种子明细", ""]
    for strat in STRATEGIES:
        for task in TASKS:
            row = []
            for s in SEEDS:
                d = summary["per_seed"][str(s)].get("%s/%s" % (strat, task))
                row.append("%.4f" % d["value"] if d else "—")
            lines.append("- %s/%s: %s" % (strat, task, " / ".join(row)))
    if summary["_audit"]["warnings"]:
        lines += ["", "## 审计警告", ""] + ["- %s" % w for w in summary["_audit"]["warnings"]]
    with open(os.path.join(outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n写入 %s/summary.json 与 REPORT.md" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
