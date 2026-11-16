#!/usr/bin/env python3
"""C 臂（R-曲线）离线重生成 adaq 秩分配。

用 ranks.json 的 layer_stats score（= rel_error × importance，即完整 AdaQ 信号），
经 allocate_ranks 精确守恒分配到预算 R ∈ {128, 512, 1024}，输出 per_layer 秩文件。
纯 CPU，不加载模型。R=128 需 r_min=2（64×2=128）。

产物：configs/rcurve/ranks_adaq_R{R}.json，结构含 per_layer_ranks（sum==R），
train.py 以 --rank-scheme per_layer --ranks-file 此文件 --budget R 消费。
"""
import json
import os
import sys

ROOT = "/home/ubuntu/AdaQ-LoRA"
sys.path.insert(0, os.path.join(ROOT, "adaq-lora-research-pipeline", "scripts"))
from adaq_precompute import allocate_ranks  # noqa: E402

base = json.load(open(os.path.join(ROOT, "ranks.json")))
stats = base["layer_stats"]
scores = {k: float(v["score"]) for k, v in stats.items()}
n = len(scores)
print(f"layer_stats n={n} score range=[{min(scores.values()):.4f}, {max(scores.values()):.4f}]")

outdir = os.path.join(ROOT, "configs", "rcurve")
os.makedirs(outdir, exist_ok=True)

for R, rmin in [(128, 2), (512, 4), (1024, 4)]:
    ranks = allocate_ranks(scores, R, rmin, 128)
    s = sum(ranks.values())
    assert s == R, f"R={R} 守恒失败 sum={s}"
    out = {
        "model": base.get("model"),
        "budget": R,
        "r_min": rmin,
        "r_max": 128,
        "per_layer_ranks": ranks,
        "source": "gen_rcurve_ranks.py: allocate_ranks(layer_stats.score, R, r_min, 128)",
        "note": "R-曲线 adaq 臂用 per_layer(方案B) 精确守恒分配；score=rel_error×importance",
    }
    p = os.path.join(outdir, f"ranks_adaq_R{R}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {p}  sum={s} min={min(ranks.values())} max={max(ranks.values())} "
          f"distinct={len(set(ranks.values()))}")
print("OK")
