#!/usr/bin/env python3
"""生成消融实验的不同秩分配配置（实验二）。

五种策略：
  error_only:     r_l ∝ ‖E_l‖_F
  importance_only: r_l ∝ ‖W_l‖_F
  product:        r_l ∝ ‖E_l‖_F · ‖W_l‖_F  (完整 AdaQ)
  uniform:        r_l = R / N
  equal_weight:   r_l ∝ (‖E_l‖_F + ‖W_l‖_F)  (等权消融)

用法: python generate_ablation_configs.py --base-ranks ranks.json --budget 512
"""
import argparse
import json
import os
import sys

# 复用 pipeline 中的分配逻辑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adaq-lora-research-pipeline", "scripts"))
from adaq_precompute import allocate_ranks


def generate_strategy(stats, strategy, budget, r_min=4, r_max=128):
    """根据策略从 layer_stats 生成 score，再分配秩。"""
    if strategy == "uniform":
        names = list(stats)
        n = len(names)
        scores = {name: 1.0 for name in names}
    elif strategy == "error_only":
        scores = {n: s["rel_error"] for n, s in stats.items()}
    elif strategy == "importance_only":
        scores = {n: s["importance"] for n, s in stats.items()}
    elif strategy == "product":
        scores = {n: s["score"] for n, s in stats.items()}
    elif strategy == "equal_weight":
        # 归一化后等权相加
        import statistics
        errors = [s["rel_error"] for s in stats.values()]
        imps = [s["importance"] for s in stats.values()]
        err_mean, err_std = statistics.mean(errors), statistics.stdev(errors) or 1
        imp_mean, imp_std = statistics.mean(imps), statistics.stdev(imps) or 1
        scores = {
            n: (s["rel_error"] - err_mean) / err_std + (s["importance"] - imp_mean) / imp_std
            for n, s in stats.items()
        }
        # 确保非负
        min_score = min(scores.values())
        if min_score < 0:
            scores = {n: v - min_score + 1e-8 for n, v in scores.items()}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranks = allocate_ranks(scores, budget, r_min, r_max)
    return ranks, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ranks", default="ranks.json", help="adaq_precompute.py 输出")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--r-min", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=128)
    ap.add_argument("--out-dir", default="configs/ablation_ranks")
    args = ap.parse_args()

    if not os.path.exists(args.base_ranks):
        print(f"ERROR: {args.base_ranks} not found. Run adaq_precompute.py first.", file=sys.stderr)
        sys.exit(1)

    with open(args.base_ranks) as f:
        base = json.load(f)

    stats = base["layer_stats"]
    strategies = ["error_only", "importance_only", "product", "uniform", "equal_weight"]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== 消融配置生成 (budget={args.budget}) ===\n")

    for strategy in strategies:
        ranks, scores = generate_strategy(stats, strategy, args.budget, args.r_min, args.r_max)

        # 方案 A 分组（按模块类型聚合）
        import statistics as stat_mod
        KNOWN_MODS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        type_ranks = {}
        for name, r in ranks.items():
            parts = name.split(".")
            mod = next((p for p in reversed(parts) if p in KNOWN_MODS), parts[-2] if name.endswith(".weight") else parts[-1])
            type_ranks.setdefault(mod, []).append(r)
        type_ranks = {k: int(stat_mod.median(v)) for k, v in type_ranks.items()}

        out = {
            "strategy": strategy,
            "budget": args.budget,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "per_layer_ranks": ranks,
            "per_module_type_ranks (方案A, PEFT 直接使用)": type_ranks,
            "scores": {n: round(s, 6) for n, s in scores.items()},
        }

        out_file = os.path.join(args.out_dir, f"ranks_{strategy}.json")
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        rank_vals = list(ranks.values())
        print(f"  {strategy:<18s}: total={sum(rank_vals)}, min={min(rank_vals)}, max={max(rank_vals)}, "
              f"types={ {k: v for k, v in sorted(type_ranks.items(), key=lambda x: -x[1])} }")
        print(f"    → {out_file}")

    print(f"\n所有消融配置已生成到 {args.out_dir}/")
    print("用法: python train.py --method adaq --ranks-file configs/ablation_ranks/ranks_error_only.json")


if __name__ == "__main__":
    main()
