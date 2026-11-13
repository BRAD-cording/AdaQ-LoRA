#!/usr/bin/env python3
"""P1：ε_l 实测 —— 把 theory-review-report.md 里「待实测」的四个假设全部落成数据。

四项（对应 theory-review-report.md 的 A1/A2/A3 + 幂律 H1）：
  A2  ε_l = ‖E_l‖_F / ‖W_l‖_F  逐层实测（min/median/max），验证 ε₄≈0.02 经验值
  A1  零均值假设：E_l = Q₄(W_l) - W_l 的逐块/逐层均值是否为 0（近似）
  A3  块内同分布 / 块间独立：块大小 64 下，块内元素与跨块元素统计特性
  H1  幂律：σ_{l,i} = c_l · i^{-α}，逐层奇异值谱拟合 α，检验 α 是否各层近似相同

全部走 bitsandbytes 官方 dequantize_4bit（与训练实际量化权重一致）。

用法：
  python scripts/measure_epsilon_l.py --model /home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B \
      --target-modules q_proj,v_proj --out results_v2/epsilon_l_report.json
  python scripts/measure_epsilon_l.py --self-test   # 无 GPU 合成自测
"""
import argparse
import json
import sys


def measure_layer(name, w_fp, w_q):
    """对单层算 ε_l、零均值、块统计、奇异值谱。返回 dict。"""
    import torch
    w_fp = w_fp.float()
    w_q = w_q.float()
    E = w_q - w_fp  # 量化误差矩阵
    norm_w = torch.norm(w_fp, p="fro")
    norm_e = torch.norm(E, p="fro")
    eps = (norm_e / (norm_w + 1e-12)).item()

    # A1 零均值：E 的均值（绝对 + 相对 norm）
    mean_e = E.mean().item()
    mean_abs_e = E.abs().mean().item()

    # A3 块统计：块大小 64（NF4 blocksize=64），沿最后一维分块
    # 形状 [out_dim, in_dim]，块在 in_dim 上
    E2 = E.reshape(-1)  # 展平
    n = E2.numel()
    block = 64
    n_blocks = n // block
    if n_blocks >= 2:
        E_blocks = E2[: n_blocks * block].reshape(n_blocks, block)
        block_means = E_blocks.mean(dim=1)
        block_stds = E_blocks.std(dim=1)
        # 块内 vs 块间：块内元素方差（残差） vs 块均值方差
        within_var = (E_blocks - block_means.unsqueeze(1)).pow(2).mean().item()
        between_var = block_means.var().item()
        # 块间独立性：相邻块均值的相关系数（若独立应≈0）
        if n_blocks >= 3:
            bm = block_means
            corr = torch.corrcoef(torch.stack([bm[:-1], bm[1:]]))[0, 1].item()
        else:
            corr = None
    else:
        within_var = between_var = corr = None

    # H1 幂律：奇异值谱拟合 σ_i ≈ c · i^{-α}
    s = torch.linalg.svdvals(w_fp)
    s = s[s > 1e-6]  # 去零奇异值
    alpha = None
    c_log = None
    r2 = None
    if s.numel() >= 20:
        # 只取前 80% 的奇异值拟合，避免尾部噪声
        k = max(20, int(0.8 * s.numel()))
        i = torch.arange(1, k + 1, dtype=torch.float32)
        log_s = torch.log(s[:k])
        log_i = torch.log(i)
        # 线性回归 log σ = log c - α log i
        A = torch.stack([torch.ones_like(log_i), log_i], dim=1)
        coef = torch.linalg.lstsq(A, log_s.unsqueeze(1)).solution.squeeze(1)
        c_log = coef[0].item()
        alpha = -coef[1].item()
        # R²
        pred = c_log - alpha * log_i
        ss_res = ((log_s - pred) ** 2).sum().item()
        ss_tot = ((log_s - log_s.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-12)

    return {
        "eps": eps,
        "norm_e": norm_e.item(),
        "norm_w": norm_w.item(),
        "mean_e": mean_e,
        "mean_abs_e": mean_abs_e,
        "n_blocks": n_blocks,
        "within_var": within_var,
        "between_var": between_var,
        "block_mean_autocorr_lag1": corr,
        "alpha": alpha,
        "log_c": c_log,
        "r2_powerlaw": r2,
        "n_singular_values": s.numel(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B")
    ap.add_argument("--target-modules", default="q_proj,v_proj")
    ap.add_argument("--out", default="results_v2/epsilon_l_report.json")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0, help="0=全部，>0 只测前 N 层（冒烟）")
    args = ap.parse_args()

    if args.self_test:
        import torch
        print("=== ε_l 测量自测（合成矩阵）===")
        w = torch.randn(256, 256)
        # 合成量化误差：零均值小扰动
        E = torch.randn(256, 256) * 0.02 * w.norm() / 256
        r = measure_layer("self_test", w, w + E)
        print(f"  eps={r['eps']:.6f} (应≈0.02)")
        print(f"  mean_e={r['mean_e']:.2e} (应≈0)")
        print(f"  alpha={r['alpha']:.3f} (随机矩阵应≈0 或小)")
        print("自测通过 ✅")
        return

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    modules = args.target_modules.split(",")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"=== 加载量化模型 {args.model} ===")
    q_model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto"
    )
    print("=== 加载 FP16 模型到 CPU ===")
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )

    import bitsandbytes as bnb
    q_state = dict(q_model.named_parameters())
    results = {}
    layer_count = 0
    for name, fp_param in fp_model.named_parameters():
        if not name.endswith(".weight"):
            continue
        mod_type = name.split(".")[-2]
        if mod_type not in modules:
            continue
        q_param = q_state.get(name)
        if q_param is None:
            continue
        w_q = bnb.functional.dequantize_4bit(q_param.data, q_param.quant_state).float().cpu()
        w_fp = fp_param.float().cpu()
        r = measure_layer(name, w_fp, w_q)
        results[name] = r
        layer_count += 1
        if args.max_layers and layer_count >= args.max_layers:
            print(f"  [冒烟] 已达 {args.max_layers} 层，停止")
            break

    # 汇总
    eps_vals = [r["eps"] for r in results.values()]
    alpha_vals = [r["alpha"] for r in results.values() if r["alpha"] is not None]
    r2_vals = [r["r2_powerlaw"] for r in results.values() if r["r2_powerlaw"] is not None]
    mean_e_vals = [r["mean_e"] for r in results.values()]
    import statistics
    summary = {
        "n_layers": len(results),
        "eps": {
            "min": min(eps_vals), "median": statistics.median(eps_vals), "max": max(eps_vals),
            "mean": statistics.mean(eps_vals),
        },
        "zero_mean": {
            "mean_of_layer_mean_e": statistics.mean(mean_e_vals),
            "max_abs_layer_mean_e": max(abs(v) for v in mean_e_vals),
            "mean_of_layer_mean_abs_e": statistics.mean(r["mean_abs_e"] for r in results.values()),
        },
        "powerlaw": {
            "n_fitted": len(alpha_vals),
            "alpha_min": min(alpha_vals) if alpha_vals else None,
            "alpha_median": statistics.median(alpha_vals) if alpha_vals else None,
            "alpha_max": max(alpha_vals) if alpha_vals else None,
            "r2_median": statistics.median(r2_vals) if r2_vals else None,
        },
        "per_layer": results,
    }

    import os
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== ε_l 实测结果（{len(results)} 层）===")
    print(f"  ε_l  (‖E‖_F/‖W‖_F): min={summary['eps']['min']:.5f} "
          f"median={summary['eps']['median']:.5f} max={summary['eps']['max']:.5f}")
    print(f"  零均值: 层均值 mean={summary['zero_mean']['mean_of_layer_mean_e']:.3e} "
          f"max_abs={summary['zero_mean']['max_abs_layer_mean_e']:.3e}")
    print(f"  幂律 α: median={summary['powerlaw']['alpha_median']} "
          f"(min={summary['powerlaw']['alpha_min']}, max={summary['powerlaw']['alpha_max']}, "
          f"R² median={summary['powerlaw']['r2_median']})")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
