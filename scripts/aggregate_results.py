#!/usr/bin/env python3
"""汇总评测结果：从 results/<method>_seed<seed>/**/*.json 提取分数，
计算 mean ± std，并做 Wilcoxon 符号秩检验。

用法: python aggregate_results.py [--baseline qlora] [--out summary.json]
      python aggregate_results.py --results-dir results_v2 --out results_v2/summary.json

口径审计（v2 新增）：
  * lm-eval 0.4.x 的 metric 键形如 "acc,none" / "acc_stderr,none"，本脚本会归一化。
  * 同一 (run, task) 出现多份结果 JSON 时只取 mtime 最新的一份（避免重跑后新旧混杂）。
  * 逐任务检查 sample_len 是否跨方法一致；不一致会写进 _audit.warnings 并在终端高亮，
    因为"不同方法评测样本数不同"会使对比结论无效（项目铁律 2：公平先于领先）。
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

# 任务 -> 主指标优先级（lm-eval 归一化后的键名）
TASK_METRIC_PRIORITY = {
    "mmlu": ["acc", "acc_norm"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match"],
    "truthfulqa_mc1": ["acc", "mc1"],
    "truthfulqa_mc2": ["acc", "mc2"],
    "ifeval": ["inst_level_strict_acc", "prompt_level_strict_acc", "inst_level_loose_acc"],
}
DEFAULT_PRIORITY = ["acc", "acc_norm", "mc1", "mc2",
                    "inst_level_strict_acc", "prompt_level_strict_acc",
                    "exact_match,strict-match", "exact_match,flexible-extract", "exact_match"]

# 兼容：不带 filter 后缀的别名
ALIASES = {
    "exact_match,strict-match": ["exact_match", "strict_match"],
    "exact_match,flexible-extract": ["exact_match"],
}


def norm_keys(metrics):
    """把 'acc,none' / 'acc_stderr,none' 之类的键归一化为 'acc' / 'acc_stderr'。"""
    out = {}
    for k, v in metrics.items():
        base = k.split(",")[0] if "," in k else k
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.setdefault(base, v)
    return out


def pick_metric(task, metrics):
    """按任务优先级选主指标，返回 (metric_name, value) 或 (None, None)。"""
    prio = TASK_METRIC_PRIORITY.get(task.lower(), DEFAULT_PRIORITY)
    for name in prio:
        if name in metrics:
            return name, metrics[name]
        for alias in ALIASES.get(name, []):
            if alias in metrics:
                return name, metrics[alias]
    # 兜底：优先 acc 类
    cands = [k for k in metrics if k.startswith(("acc", "mc", "exact_match", "inst_level", "prompt_level"))]
    if cands:
        return cands[0], metrics[cands[0]]
    return None, None


def find_result_files(results_dir):
    """递归查找 lm-eval 结果 JSON。"""
    files = glob.glob(os.path.join(results_dir, "**", "results*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    valid = []
    for f in files:
        if "summary" in os.path.basename(f):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "results" in data:
            valid.append(f)
    return valid


def parse_run(path):
    """从路径中的 results/<method>_seed<seed>/ 解析 (method, seed)。"""
    for part in path.split(os.sep):
        if "_seed" not in part or os.sep in part:
            continue
        tokens = part.split("_seed")
        if len(tokens) == 2 and tokens[0] and tokens[1].isdigit():
            return tokens[0], int(tokens[1])
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="qlora", help="Baseline method for significance test")
    ap.add_argument("--out", default="results/summary.json")
    ap.add_argument("--allow-mixed-samples", action="store_true",
                    help="即使 sample_len 不一致也继续（不推荐，会在 _audit 中记录）")
    ap.add_argument("--all-tasks", action="store_true",
                    help="终端表格同时打印 MMLU 各子科等细粒度任务（默认只打主指标）")
    args = ap.parse_args()

    files = find_result_files(args.results_dir)
    if not files:
        print(f"No result files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # 1) 收集：{(method, seed, task): [(mtime, path, value, sample_len, metric_name)]}
    raw = defaultdict(list)
    for f in files:
        method, seed = parse_run(f)
        if method is None:
            continue
        try:
            data = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        mtime = os.path.getmtime(f)
        for task, metrics in data.get("results", {}).items():
            if not isinstance(metrics, dict):
                continue
            m = norm_keys(metrics)
            name, val = pick_metric(task, m)
            if val is None:
                continue
            raw[(method, seed, task)].append((mtime, f, val, m.get("sample_len"), name))

    # 2) 同一 (run, task) 只保留最新一份
    chosen = {}
    dup_warnings = []
    for key, rows in raw.items():
        rows.sort(key=lambda r: r[0])
        if len(rows) > 1:
            dup_warnings.append(
                f"{key[0]}_seed{key[1]} / {key[2]}: 发现 {len(rows)} 份结果，取最新 "
                f"{os.path.basename(rows[-1][1])}"
            )
        chosen[key] = rows[-1]

    # 3) 口径审计：同一 task 跨方法/种子的 sample_len 必须一致
    by_task = defaultdict(list)
    for (method, seed, task), (_, path, val, sample_len, name) in chosen.items():
        by_task[task].append((method, seed, sample_len, val, name))

    sample_warnings = []
    mismatch_tasks = []
    sig_groups = defaultdict(list)
    for task, rows in sorted(by_task.items()):
        lens = {r[2] for r in rows if r[2] is not None}
        if len(lens) > 1:
            mismatch_tasks.append(task)
            sig = tuple(sorted((f"{m}_s{s}", sl) for m, s, sl, _, _ in rows))
            sig_groups[sig].append(task)
    for sig, tasks in sig_groups.items():
        sample_warnings.append(
            f"{len(tasks)} 个任务样本数跨方法不一致（例: {', '.join(tasks[:4])}）-> "
            + ", ".join(f"{k}={v}" for k, v in sig)
        )

    # 4) 汇总 mean ± std
    agg = defaultdict(lambda: defaultdict(list))   # method -> "task/metric" -> values
    lens = defaultdict(lambda: defaultdict(list))  # method -> "task/metric" -> sample_len
    for (method, seed, task), (_, path, val, sample_len, name) in chosen.items():
        key = f"{task}/{name}"
        agg[method][key].append(val)
        if sample_len is not None:
            lens[method][key].append(sample_len)

    summary = {}
    for method, metrics in sorted(agg.items()):
        summary[method] = {}
        for metric, values in sorted(metrics.items()):
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[method][metric] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n_seeds": len(values),
                "all_values": [round(v, 4) for v in values],
                "sample_len": sorted(set(lens[method][metric])),
            }

    # 5) Wilcoxon（adaq vs baseline），逐 task 对齐种子
    baseline = args.baseline
    sig = {}
    if "adaq" in agg:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            sig = {"error": "scipy not available"}
            wilcoxon = None
        if wilcoxon is not None:
            for metric in sorted(agg["adaq"]):
                diff = [k for k in agg if k != "adaq"]
                base_vals = agg.get(baseline, {}).get(metric)
                if not base_vals:
                    continue
                pairs = []
                for m in sorted(agg):
                    if m in ("adaq", baseline):
                        continue
                    pairs.append(m)
                # 按 seed 配对
                a = {s: v for (mm, s, t), (_, _, v, _, n) in chosen.items()
                     if mm == "adaq" and f"{t}/{n}" == metric}
                b = {s: v for (mm, s, t), (_, _, v, _, n) in chosen.items()
                     if mm == baseline and f"{t}/{n}" == metric}
                common = sorted(set(a) & set(b))
                if len(common) >= 6:
                    try:
                        stat, pval = wilcoxon([a[s] for s in common], [b[s] for s in common])
                        sig[metric] = {"statistic": float(stat), "p_value": float(pval),
                                       "significant": bool(pval < 0.05), "n_pairs": len(common)}
                    except Exception as e:  # noqa: BLE001
                        sig[metric] = {"error": str(e), "n_pairs": len(common)}
                elif common:
                    sig[metric] = {"note": f"仅 {len(common)} 对配对值；Wilcoxon 需 >=6 对才有效", "n_pairs": len(common)}

    summary["_audit"] = {
        "results_dir": args.results_dir,
        "n_files": len(files),
        "n_runs": len({(m, s) for (m, s, _t) in chosen}),
        "tasks": sorted(by_task),
        "duplicate_files": dup_warnings,
        "sample_len_mismatch": sample_warnings,
        "mismatch_task_count": len(mismatch_tasks),
    }
    if sig:
        summary["significance_test"] = {
            "test": "Wilcoxon signed-rank (paired by seed)",
            "comparison": f"adaq vs {baseline}",
            "results": sig,
        }

    # 6) 打印主表
    print("\n=== 评测结果汇总 (mean ± std) ===\n")
    HEADLINE = ("mmlu", "gsm8k", "truthfulqa_mc1", "truthfulqa_mc2", "ifeval")

    def is_headline(metric):
        if args.all_tasks:
            return True
        task = metric.split("/")[0]
        if task in HEADLINE or task.startswith(("truthfulqa", "ifeval")):
            return True
        return not task.startswith("mmlu_")

    metrics_all = sorted({k for m in agg.values() for k in m if is_headline(k)})
    width = max([len(m) for m in agg] + [8]) + 2
    colw = max([len(m) for m in metrics_all] + [18]) + 2
    header = f"{'Method':<{width}s}" + "".join(f"{m:<{colw}s}" for m in metrics_all)
    print(header)
    print("-" * len(header))
    for method in sorted(agg):
        row = f"{method:<{width}s}"
        for metric in metrics_all:
            s = summary.get(method, {}).get(metric)
            cell = f"{s['mean']:.4f}±{s['std']:.4f}(n={s['n_seeds']})" if s else "—"
            row += cell.ljust(colw)
        print(row)

    if dup_warnings:
        print("\n[去重] 同一 run/task 存在多份结果，已取最新：")
        for w in dup_warnings:
            print("  - " + w)
    if sample_warnings:
        print("\n[口径告警] 以下任务的评测样本数跨方法不一致，对比结论不可用：")
        for w in sample_warnings:
            print("  - " + w)
        if not args.allow_mixed_samples:
            print("\n提示：这是预期内的拦截——请统一样本数后重跑，或显式加 --allow-mixed-samples。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已写入 {args.out}")
    print("\n注意: 论文数值只能从此 JSON 汇总生成，禁止手抄。")

    # 退出码：口径不一致 -> 2（可被 CI/总控当机读门用）
    if sample_warnings and not args.allow_mixed_samples:
        sys.exit(2)


if __name__ == "__main__":
    main()
