#!/usr/bin/env python3
"""P2-D2: 误差分布图（量化误差 ε_l 逐层分布 + 幂律谱 + 零均值）。

数据源：results_v2/epsilon_l_report.json（P1 已测，measure_epsilon_l.py 产出）。
程序化生成，绝不 AI 生图。产出 3 张子图合一的 results_v2/epsilon_distribution.png：
  1. 逐层 ε_l（q_proj vs v_proj 分层着色）
  2. 幂律谱拟合 α 的逐层分布（H1 假设核验）
  3. 零均值验证（逐层 mean_e，对数坐标看是否 ~0）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "results_v2", "epsilon_l_report.json")
OUT = os.path.join(ROOT, "results_v2", "epsilon_distribution.png")


def main():
    with open(REPORT) as fh:
        d = json.load(fh)
    pl = d["per_layer"]
    layers_q = [pl[k] for k in sorted(pl) if "q_proj" in k]
    layers_v = [pl[k] for k in sorted(pl) if "v_proj" in k]

    eps_q = [e["eps"] for e in layers_q]
    eps_v = [e["eps"] for e in layers_v]
    alpha_q = [e["alpha"] for e in layers_q]
    alpha_v = [e["alpha"] for e in layers_v]
    mean_e_q = [e["mean_e"] for e in layers_q]
    mean_e_v = [e["mean_e"] for e in layers_v]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 子图1：逐层 ε_l
    ax = axes[0]
    ax.plot(range(32), eps_q, "o-", color="#1f77b4", label="q_proj", ms=4)
    ax.plot(range(32), eps_v, "s-", color="#d62728", label="v_proj", ms=4)
    ax.axhline(d["eps"]["median"], color="gray", ls="--", lw=1, label=f"median={d['eps']['median']:.4f}")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$\varepsilon_l = \|E_l\|_F / \|W_l\|_F$")
    ax.set_title(r"$\varepsilon_l$ per layer (flat, ~8.7% spread)")
    ax.legend(fontsize=8)

    # 子图2：幂律谱 α 分布
    ax = axes[1]
    ax.plot(range(32), alpha_q, "o-", color="#1f77b4", label="q_proj", ms=4)
    ax.plot(range(32), alpha_v, "s-", color="#d62728", label="v_proj", ms=4)
    ax.axhline(d["powerlaw"]["alpha_median"], color="gray", ls="--", lw=1, label=f"median α={d['powerlaw']['alpha_median']:.3f}")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"power-law exponent $\alpha$ ($\sigma_i = c\,i^{-\alpha}$)")
    ax.set_title(r"Singular-value power-law $\alpha$ per layer (H1)")
    ax.legend(fontsize=8)

    # 子图3：零均值验证（对数坐标）
    ax = axes[2]
    ax.semilogy(range(32), [abs(m) for m in mean_e_q], "o-", color="#1f77b4", label="q_proj |mean_e|", ms=4)
    ax.semilogy(range(32), [abs(m) for m in mean_e_v], "s-", color="#d62728", label="v_proj |mean_e|", ms=4)
    ax.axhline(d["zero_mean"]["max_abs_layer_mean_e"], color="gray", ls="--", lw=1, label=f"max |mean_e|={d['zero_mean']['max_abs_layer_mean_e']:.2e}")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$|\mathrm{mean}(E_l)|$ (log scale)")
    ax.set_title(r"Zero-mean check: $|\mathrm{mean}(E_l)| \sim 10^{-5}$")
    ax.legend(fontsize=8)

    fig.suptitle("AdaQ-LoRA: NF4 quantization error $\\varepsilon_l$ diagnostics (64 layers)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"图已生成 -> {OUT}")


if __name__ == "__main__":
    main()
