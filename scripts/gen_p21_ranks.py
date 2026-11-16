#!/usr/bin/env python3
"""P2-1 成本归一化秩分配：生成 cost_norm / raw_score / uniform 三臂 ranks（预注册 4107552）。

cost_norm: r_l ∝ score_l / c_l，c_l = d_in+d_out（q_proj:8192, v_proj:5120）—— 成本归一化修正
raw_score: r_l ∝ score_l（= ‖E_l‖_F，原 AdaQ per-layer，对照）
uniform:   r_l ∝ 1.0（等秩基线）

三臂均 allocate_ranks(·, budget=512, r_min=4, r_max=128) 守恒 sum-of-ranks=512。
参数量 = sum(r_l * c_l) 须实测报告（等秩和≠等参数，fair-baseline F6/M3）。
用法: python gen_p21_ranks.py --base-ranks ranks.json --config models/Meta-Llama-3-8B/config.json --budget 512 --out-dir configs/p21
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adaq-lora-research-pipeline", "scripts"))
from adaq_precompute import allocate_ranks

KNOWN_MODS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def mod_of(name):
    parts = name.split(".")
    return next((p for p in reversed(parts) if p in KNOWN_MODS), parts[-2] if name.endswith(".weight") else parts[-1])


def site_costs(config):
    """从模型 config 算每个 target module 的 LoRA 成本因子 c_l = d_in + d_out。"""
    h = config["hidden_size"]
    nh = config.get("num_attention_heads")
    nkv = config.get("num_key_value_heads", nh)
    head_dim = h // nh
    q_out = nh * head_dim          # 4096
    kv_out = nkv * head_dim        # 1024 (GQA)
    # d_in = hidden（q/v 投影输入均为 hidden）；d_out 见上
    return {
        "q_proj": h + q_out,        # 4096 + 4096 = 8192
        "v_proj": h + kv_out,       # 4096 + 1024 = 5120
        "k_proj": h + kv_out,
        "o_proj": q_out + h,        # 4096 + 4096 = 8192
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ranks", default="ranks.json")
    ap.add_argument("--config", default="models/Meta-Llama-3-8B/config.json")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--r-min", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=128)
    ap.add_argument("--out-dir", default="configs/p21")
    args = ap.parse_args()

    with open(args.base_ranks) as f:
        base = json.load(f)
    with open(args.config) as f:
        config = json.load(f)

    stats = base["layer_stats"]
    costs = site_costs(config)
    print(f"=== 成本因子 c_l=d_in+d_out: {costs} ===")

    # 每 site 的成本
    c_of = {}
    for name in stats:
        m = mod_of(name)
        if m not in costs:
            print(f"ERROR: site {name} mod {m} 无成本因子（target 非 q/v/k/o）", file=sys.stderr)
            sys.exit(2)
        c_of[name] = costs[m]

    # 三主臂 budget=512；uniform_pm 是预注册 §2 条件臂（cost_norm params>uniform×1.05 触发），
    # 等秩 r=9 → budget=576，params≈3.83M 上夹逼 cost_norm（保守：胜过更多参数的 flat 臂才算强阳性）。
    arms = {
        "cost_norm": ({n: stats[n]["score"] / c_of[n] for n in stats}, args.budget),   # r ∝ score/c
        "raw_score": ({n: stats[n]["score"] for n in stats}, args.budget),             # r ∝ score
        "uniform":   ({n: 1.0 for n in stats}, args.budget),                           # r ∝ 1, budget 512 → r=8
        "uniform_pm": ({n: 1.0 for n in stats}, 576),                                  # param-matched, budget 576 → r=9
    }

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n=== P2-1 四臂生成 (主臂 budget={args.budget}, uniform_pm budget=576, r_min={args.r_min}, r_max={args.r_max}) ===\n")

    summary = {}
    for arm, (scores, arm_budget) in arms.items():
        ranks = allocate_ranks(scores, arm_budget, args.r_min, args.r_max)
        sum_ranks = sum(ranks.values())
        params = sum(ranks[n] * c_of[n] for n in ranks)
        # 按模块类型统计秩
        q_ranks = [ranks[n] for n in ranks if mod_of(n) == "q_proj"]
        v_ranks = [ranks[n] for n in ranks if mod_of(n) == "v_proj"]
        type_ranks = {}
        for n, r in ranks.items():
            type_ranks.setdefault(mod_of(n), []).append(r)
        type_med = {k: sorted(v)[len(v) // 2] for k, v in type_ranks.items()}

        out = {
            "arm": arm,
            "score_formula": {
                "cost_norm": "score_l / c_l  (c_l=d_in+d_out)",
                "raw_score": "score_l (= rel_error_l * importance_l = ||E_l||_F)",
                "uniform": "1.0",
                "uniform_pm": "1.0 (param-matched control, budget 576 -> r=9)",
            }[arm],
            "budget": arm_budget,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "scheme": "per_layer (Scheme B, 守恒)",
            "sum_ranks": sum_ranks,
            "params": params,
            "per_layer_ranks": ranks,
            "per_module_type_ranks_median": type_med,
            "q_rank_min_max": [min(q_ranks), max(q_ranks)] if q_ranks else None,
            "v_rank_min_max": [min(v_ranks), max(v_ranks)] if v_ranks else None,
            "scores": {n: round(s, 8) for n, s in scores.items()},
        }
        out_file = os.path.join(args.out_dir, f"ranks_{arm}.json")
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        summary[arm] = {"sum_ranks": sum_ranks, "params": params, "type_med": type_med,
                        "q_mm": out["q_rank_min_max"], "v_mm": out["v_rank_min_max"]}
        print(f"  {arm:<10s}: sum_ranks={sum_ranks}  params={params:,}  "
              f"q[min,max]={out['q_rank_min_max']} v[min,max]={out['v_rank_min_max']}  type_med={type_med}")
        print(f"    → {out_file}")

    # 参数对照（R1 判据：cost_norm params ≤ uniform×1.05；超则用 uniform_pm 等参数对照）
    pu = summary["uniform"]["params"]
    ppm = summary["uniform_pm"]["params"]
    pc = summary["cost_norm"]["params"]
    print(f"\n=== 参数对照（uniform r8={pu:,} / uniform_pm r9={ppm:,} / cost_norm={pc:,}）===")
    for arm in ("cost_norm", "raw_score"):
        pa = summary[arm]["params"]
        ratio = pa / pu
        flag = "≤1.05 vs uniform：R1 可直判" if ratio <= 1.05 else f">1.05 vs uniform：须用 uniform_pm(r9,{ppm:,}) 等参数对照"
        print(f"  {arm:<10s}: params={pa:,}  ratio_vs_uniform={ratio:.4f}  {flag}")
    print(f"  cost_norm vs uniform_pm: {pc/ppm:.4f}（cost_norm 更省参数 → 若性能≥uniform_pm 即等参数下分配生效，强阳性）")

    print(f"\n所有 P2-1 配置已生成到 {args.out_dir}/")
    print("用法: python train_phase3.py --method adaq --rank-scheme per_layer --ranks-file configs/p21/ranks_cost_norm.json --budget 512 ...")


if __name__ == "__main__":
    main()
