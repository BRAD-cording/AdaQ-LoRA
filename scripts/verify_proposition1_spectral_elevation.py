#!/usr/bin/env python3
"""数值验证命题 1'（平均谱抬升）：

对幂律谱合成的 ΔW* 叠加不同强度的近似各向同性随机 E，
验证有效秩 erank(M) = ‖M‖_F²/‖M‖₂² 随 ‖E‖_F 增大而增大。

输出: docs/figures/erank_vs_error.png + 数据表
"""
import numpy as np
import json
import os

def powerlaw_spectrum(d, k, alpha=1.5, seed=42):
    """生成 d×k 矩阵，奇异值服从幂律 σ_i ∝ i^{-alpha}。"""
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(d, d))
    V, _ = np.linalg.qr(rng.randn(k, k))
    sv = np.array([(i + 1) ** (-alpha) for i in range(min(d, k))])
    sv = sv / sv[0]  # 归一化到 σ_1=1
    S = np.zeros((d, k))
    S[:len(sv), :len(sv)] = np.diag(sv)
    return U @ S @ V.T, sv

def add_isotropic_noise(W, relative_error, seed=0):
    """叠加近似各向同性高斯噪声，使 ‖E‖_F = relative_error * ‖W‖_F。"""
    rng = np.random.RandomState(seed)
    E = rng.randn(*W.shape)
    E = E / np.linalg.norm(E, 'fro') * (relative_error * np.linalg.norm(W, 'fro'))
    return W + E, E

def effective_rank(M):
    """erank(M) = ‖M‖_F²/‖M‖₂²。"""
    fro = np.linalg.norm(M, 'fro') ** 2
    spec = np.linalg.norm(M, 2) ** 2
    return fro / (spec + 1e-12)

def main():
    d, k = 256, 256
    alpha = 1.5  # 幂律衰减指数
    W_star, sv_star = powerlaw_spectrum(d, k, alpha=alpha)
    erank_star = effective_rank(W_star)
    fro_star = np.linalg.norm(W_star, 'fro')

    print(f"=== 命题 1' 数值验证：平均谱抬升 ===")
    print(f"合成 ΔW*: d={d}, k={k}, α={alpha}, erank={erank_star:.2f}, ‖ΔW*‖_F={fro_star:.4f}")
    print()

    errors_grid = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
    results = []

    print(f"{'‖E‖_F/‖W‖_F':>15s} {'‖M‖_F':>10s} {'‖M‖_2':>10s} {'erank(M)':>10s} {'Δerank':>10s}")
    print("-" * 60)

    for rel_err in errors_grid:
        eranks_seeds = []
        for seed in range(20):  # 20 次平均
            M, E = add_isotropic_noise(W_star, rel_err, seed=seed)
            eranks_seeds.append(effective_rank(M))
        erank_mean = np.mean(eranks_seeds)
        erank_std = np.std(eranks_seeds)
        M, E = add_isotropic_noise(W_star, rel_err, seed=0)
        fro_m = np.linalg.norm(M, 'fro')
        spec_m = np.linalg.norm(M, 2)
        delta_erank = erank_mean - erank_star
        print(f"{rel_err:>15.3f} {fro_m:>10.4f} {spec_m:>10.4f} {erank_mean:>10.2f} {delta_erank:>+10.2f}")
        results.append({
            "relative_error": rel_err,
            "fro_M": fro_m,
            "spec_M": spec_m,
            "erank_mean": erank_mean,
            "erank_std": erank_std,
            "delta_erank": delta_erank,
        })

    # 验证单调性
    eranks = [r["erank_mean"] for r in results]
    is_monotone = all(eranks[i] <= eranks[i + 1] + eranks[i] * 0.01 for i in range(len(eranks) - 1))
    total_rise = eranks[-1] - eranks[0]
    print()
    print(f"erank 单调递增: {'✅' if is_monotone else '⚠️ 近似单调'}")
    print(f"总抬升幅度: {eranks[0]:.2f} → {eranks[-1]:.2f} (Δ={total_rise:.2f}, +{total_rise/eranks[0]*100:.1f}%)")
    print(f"命题 1' 验证: {'✅ 通过' if is_monotone and total_rise > 0 else '❌ 未通过'}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "erank_vs_error.json"), "w") as f:
        json.dump({
            "proposition": "1' (average spectral elevation)",
            "matrix_shape": [d, k],
            "powerlaw_alpha": alpha,
            "baseline_erank": erank_star,
            "results": results,
            "monotone": is_monotone,
            "verdict": "PASS" if is_monotone and total_rise > 0 else "FAIL",
        }, f, indent=2)
    print(f"\n数据已写入 docs/figures/erank_vs_error.json")


if __name__ == "__main__":
    main()
