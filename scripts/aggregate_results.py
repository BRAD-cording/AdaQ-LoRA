#!/usr/bin/env python3
"""汇总评测结果：从 results/<method>_seed<seed>/*.json 提取分数，
计算 mean ± std，并做 Wilcoxon 符号秩检验。

用法: python aggregate_results.py [--baseline qlora] [--out summary.json]
"""
import argparse
import json
import os
import glob
import sys
from collections import defaultdict


def find_result_files(results_dir):
    """递归查找所有 lm-eval 结果 JSON 文件。"""
    import glob as g
    # Pattern 1: results.json (standard)
    files = g.glob(os.path.join(results_dir, "**", "results.json"), recursive=True)
    # Pattern 2: any *results*.json
    if not files:
        files = g.glob(os.path.join(results_dir, "**", "*results*.json"), recursive=True)
    # Pattern 3: any *.json in subdirs (excluding logs)
    if not files:
        all_json = g.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
        files = [f for f in all_json if "eval.log" not in f and "summary" not in f]
    # Verify files contain lm-eval results format
    valid = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "results" in data or any(k in str(data) for k in ["acc", "mc1", "strict_match"]):
                valid.append(f)
        except (json.JSONDecodeError, KeyError):
            continue
    return valid


def extract_scores(file_path):
    """从 lm-eval 输出 JSON 提取任务分数。"""
    with open(file_path) as f:
        data = json.load(f)

    scores = {}
    # lm-eval output format: {"results": {"task_name": {"acc": 0.75, "acc_stderr": 0.02}}}
    results = data.get("results", data)
    for task, metrics in results.items():
        if isinstance(metrics, dict):
            for metric_key in ["acc", "acc_norm", "mc1", "mc2", "prompt_level_strict_acc"]:
                if metric_key in metrics:
                    scores[f"{task}/{metric_key}"] = metrics[metric_key]
                    if f"{metric_key}_stderr" in metrics:
                        scores[f"{task}/{metric_key}_stderr"] = metrics[f"{metric_key}_stderr"]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="qlora", help="Baseline method for significance test")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()

    # 收集所有结果
    # structure: {method: {task_metric: [scores across seeds]}}
    all_scores = defaultdict(lambda: defaultdict(list))

    result_files = find_result_files(args.results_dir)
    if not result_files:
        print(f"No result files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    for f in result_files:
        # Parse method and seed from path: results/<method>_seed<seed>/...
        parts = f.split(os.sep)
        dir_name = None
        for p in parts:
            if "_seed" in p:
                dir_name = p
                break
        if dir_name is None:
            continue
        # Extract method and seed
        tokens = dir_name.split("_seed")
        if len(tokens) != 2:
            continue
        method = tokens[0]
        scores = extract_scores(f)
        for metric, val in scores.items():
            if isinstance(val, (int, float)):
                all_scores[method][metric].append(val)

    # 计算 mean ± std
    import statistics

    summary = {}
    for method, metrics in sorted(all_scores.items()):
        summary[method] = {}
        for metric, values in sorted(metrics.items()):
            if len(values) == 0:
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[method][metric] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n_seeds": len(values),
                "all_values": [round(v, 4) for v in values],
            }

    # Wilcoxon 符号秩检验（AdaQ vs baseline）
    baseline = args.baseline
    if baseline in all_scores and "adaq" in all_scores:
        try:
            from scipy.stats import wilcoxon
            sig_tests = {}
            for metric in all_scores.get("adaq", {}):
                adaq_vals = all_scores["adaq"][metric]
                base_vals = all_scores[baseline].get(metric, [])
                if len(adaq_vals) == len(base_vals) and len(adaq_vals) >= 3:
                    try:
                        stat, pval = wilcoxon(adaq_vals, base_vals)
                        sig_tests[metric] = {"statistic": stat, "p_value": pval, "significant": pval < 0.05}
                    except Exception:
                        pass
            summary["significance_test"] = {
                "test": "Wilcoxon signed-rank",
                "comparison": f"adaq vs {baseline}",
                "results": sig_tests,
            }
        except ImportError:
            summary["significance_test"] = {"error": "scipy not available"}

    # 输出主表
    print("\n=== 评测结果汇总 (mean ± std) ===\n")
    methods = sorted(all_scores.keys())
    metrics_set = set()
    for m in all_scores.values():
        metrics_set.update(m.keys())
    metrics_sorted = sorted(metrics_set)

    # 表头
    header = f"{'Method':<12s}" + "".join(f"{m:<25s}" for m in metrics_sorted)
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<12s}"
        for metric in metrics_sorted:
            if metric in summary.get(method, {}):
                s = summary[method][metric]
                row += f"{s['mean']:.4f}±{s['std']:.4f} (n={s['n_seeds']})".ljust(25)
            else:
                row += "—".ljust(25)
        print(row)

    # 保存
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已写入 {args.out}")
    print("\n注意: 论文数值只能从此 JSON 汇总生成，禁止手抄。")


if __name__ == "__main__":
    main()
