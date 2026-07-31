#!/usr/bin/env python3
"""生成合成 ranks.json 用于无 GPU 环境下的管道测试。

模拟 adaq_precompute.py 在真实模型上的输出：
- 64 个适配位点（8 模块类型 × 8 层）
- q_proj/v_proj 量化误差高（模拟 NF4 对注意力层敏感性）
- 输出格式与 adaq_precompute.py 完全一致

用法: python generate_synthetic_ranks.py --budget 512 --out ranks.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adaq-lora-research-pipeline", "scripts"))
from adaq_precompute import allocate_ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--r-min", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=128)
    ap.add_argument("--out", default="ranks.json")
    args = ap.parse_args()

    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    num_layers = 8

    # 模拟逐层统计
    stats = {}
    for layer in range(num_layers):
        for mod in modules:
            name = f"model.layers.{layer}.self_attn.{mod}.weight" if mod in ("q_proj", "k_proj", "v_proj", "o_proj") \
                else f"model.layers.{layer}.mlp.{mod}.weight"

            # q/v_proj 量化误差高 (ε≈0.025), 其余低 (ε≈0.015)
            if mod in ("q_proj", "v_proj"):
                rel_error = 0.025 + 0.003 * layer / num_layers
            elif mod in ("k_proj", "o_proj"):
                rel_error = 0.018 + 0.002 * layer / num_layers
            else:
                rel_error = 0.015 + 0.001 * layer / num_layers

            # 权重范数模拟 (gate/up/down 更大)
            if mod in ("gate_proj", "up_proj", "down_proj"):
                norm_w = 120.0 + 5.0 * layer
            else:
                norm_w = 80.0 + 3.0 * layer

            score = rel_error * norm_w
            stats[name] = {
                "rel_error": round(rel_error, 6),
                "importance": round(norm_w, 4),
                "score": round(score, 4),
            }

    scores = {n: s["score"] for n, s in stats.items()}
    ranks = allocate_ranks(scores, args.budget, args.r_min, args.r_max)

    # 方案 A 分组
    import statistics
    type_ranks = {}
    for name, r in ranks.items():
        mod = name.split(".")[-2] if not name.endswith(".weight") else name.split(".")[-2]
        type_ranks.setdefault(mod, []).append(r)
    type_ranks = {k: int(statistics.median(v)) for k, v in type_ranks.items()}

    out = {
        "model": "synthetic-test",
        "budget": args.budget,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "per_layer_ranks": ranks,
        "per_module_type_ranks (方案A, PEFT 直接使用)": type_ranks,
        "layer_stats": stats,
        "_note": "Synthetic data for pipeline testing without GPU. Replace with real adaq_precompute.py output.",
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Synthetic ranks written to {args.out}")
    print(f"  Layers: {len(ranks)}, Total rank: {sum(ranks.values())}")
    print(f"  Module type ranks (方案A):")
    for k, v in sorted(type_ranks.items(), key=lambda x: -x[1]):
        print(f"    {k}: r = {v}")


if __name__ == "__main__":
    main()
