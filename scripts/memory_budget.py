#!/usr/bin/env python3
"""A10 显存预算计算器（Llama-3-8B QLoRA 级训练）。

估算不同配置的显存占用，辅助 batch×seq 调参。
依据：training-optimization.md §1 显存预算模型。

用法：
  python memory_budget.py
  python memory_budget.py --batch 4 --seq 2048 --grad-checkpoint
"""
import argparse
import json
import sys


def estimate_memory(
    model_name="Llama-3-8B",
    batch_size=4,
    seq_length=2048,
    gradient_checkpointing=True,
    packing=True,
    lora_rank=512,
    num_lora_sites=64,
    vocab_size=128256,
    hidden_size=4096,
    num_layers=32,
    optimizer_bits=8,
):
    """估算训练峰值显存（GB）。"""

    # --- 4-bit 基座权重 ---
    # Llama-3-8B: ~8B params × 0.5 bytes/param (4-bit + double quant overhead ~10%)
    base_params = 8.03e9
    base_4bit_gb = (base_params * 0.5 * 1.1) / 1e9  # ~4.4 GB
    # With double quant metadata: ~5.5 GB
    base_4bit_gb = 5.5

    # --- LoRA 参数 (bf16 = 2 bytes) ---
    # Per site: 2 × hidden × rank (A and B matrices)
    lora_params = num_lora_sites * 2 * hidden_size * lora_rank
    lora_gb = (lora_params * 2) / 1e9  # bf16

    # --- 优化器状态 ---
    # AdamW: 2× params (m, v), at optimizer_bits
    optimizer_gb = (lora_params * 2 * optimizer_bits / 8) / 1e9

    # --- 梯度 (same size as LoRA params, bf16) ---
    gradient_gb = lora_gb  # ~same as lora params

    # --- 激活显存 ---
    # Per layer per token: ~6 × hidden_size × seq_length (for attention + MLP activations)
    # Without gradient checkpointing: batch × seq × num_layers × 6 × hidden × 2 bytes (bf16)
    # With gradient checkpointing: ~1/3
    # Liger CE fusion saves ~vocab_size × batch × seq × 2 bytes

    attn_activation = batch_size * seq_length * hidden_size * 2  # bf16
    mlp_activation = batch_size * seq_length * hidden_size * 4 * 2  # intermediate 4x, bf16
    per_layer_activation = attn_activation + mlp_activation
    total_activation = per_layer_activation * num_layers

    # Packing: ~1.5-2x more tokens per batch, so multiply by ~1.5
    packing_factor = 1.5 if packing else 1.0
    total_activation *= packing_factor

    # Gradient checkpointing: only store ~sqrt(num_layers) layers worth
    if gradient_checkpointing:
        # Approximate: store 1 layer's activation + recompute
        total_activation = per_layer_activation * (num_layers ** 0.5) * packing_factor

    # CE loss activation (vocab 128k is huge)
    ce_activation = batch_size * seq_length * vocab_size * 2 * packing_factor  # bf16
    if not gradient_checkpointing:
        ce_total = ce_activation
    else:
        # With Liger CE fusion, this is 0; without, it's ~1 layer's worth
        ce_total = batch_size * seq_length * vocab_size * 2 * 0.1  # reduced by fusion

    activation_gb = (total_activation + ce_total) / 1e9

    # --- 碎片余量 ---
    fragmentation = 1.0  # GB

    total = base_4bit_gb + lora_gb + optimizer_gb + gradient_gb + activation_gb + fragmentation

    return {
        "model": model_name,
        "config": {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "gradient_checkpointing": gradient_checkpointing,
            "packing": packing,
            "lora_rank": lora_rank,
            "num_lora_sites": num_lora_sites,
            "optimizer_bits": optimizer_bits,
        },
        "components_gb": {
            "base_4bit_weights": round(base_4bit_gb, 2),
            "lora_params_bf16": round(lora_gb, 4),
            "optimizer_state": round(optimizer_gb, 4),
            "gradient": round(gradient_gb, 4),
            "activations": round(activation_gb, 2),
            "fragmentation_margin": fragmentation,
        },
        "total_gb": round(total, 2),
        "within_23gb_budget": total <= 23.0,
        "headroom_gb": round(24.0 - total, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="A10 VRAM budget calculator")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--packing", action="store_true", default=True)
    ap.add_argument("--no-packing", dest="packing", action="store_false")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--sites", type=int, default=64)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        print("=== A10 显存预算交互式估算 ===")
        print("扫描 batch_size × seq_length 网格：\n")
        for batch in [1, 2, 4, 8]:
            for seq in [512, 1024, 2048, 4096]:
                result = estimate_memory(batch_size=batch, seq_length=seq)
                status = "✅" if result["within_23gb_budget"] else "❌"
                print(f"  {status} batch={batch} seq={seq:4d} → {result['total_gb']:.1f} GB (headroom {result['headroom_gb']:+.1f})")
        return

    result = estimate_memory(
        batch_size=args.batch,
        seq_length=args.seq,
        gradient_checkpointing=args.grad_checkpoint,
        packing=args.packing,
        lora_rank=args.rank,
        num_lora_sites=args.sites,
    )

    print(f"=== {result['model']} 显存预算 ===")
    print(f"配置: batch={args.batch}, seq={args.seq}, "
          f"grad_ckpt={args.grad_checkpoint}, packing={args.packing}, "
          f"rank={args.rank}, sites={args.sites}")
    print(f"\n组件明细:")
    for comp, val in result["components_gb"].items():
        print(f"  {comp:<25s}: {val:.2f} GB")
    print(f"  {'─' * 35}")
    print(f"  {'Total':<25s}: {result['total_gb']:.2f} GB")
    print(f"  {'A10 limit (24GB)':<25s}: 24.00 GB")
    print(f"  {'Headroom':<25s}: {result['headroom_gb']:+.2f} GB")
    print(f"\n  {'✅ 在 23GB 预算内' if result['within_23gb_budget'] else '❌ 超预算，需减小 batch 或 seq'}")

    print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
