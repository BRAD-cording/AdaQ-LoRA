#!/usr/bin/env python3
"""数值语义不变性检查（铁律3 执行手段）。

每项优化启用后，同一批 100 条数据、固定种子跑 100 步，
与基线配置对比 loss 序列，相对偏差 >1% 即回退并记录。

用法：
  # 基线配置
  python numerical_invariance_check.py --config baseline --steps 100

  # 启用 FlashAttention-2 后
  python numerical_invariance_check.py --config flash_attn --steps 100

  # 对比两个配置
  python numerical_invariance_check.py --compare baseline flash_attn
"""
import argparse
import json
import os
import sys
import numpy as np


CONFIGS = {
    "baseline": {
        "attn_implementation": "sdpa",
        "gradient_checkpointing": False,
        "packing": False,
        "liger_kernel": False,
        "description": "基线（无优化）"
    },
    "flash_attn": {
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": False,
        "packing": False,
        "liger_kernel": False,
        "description": "+ FlashAttention-2"
    },
    "packing": {
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": False,
        "packing": True,
        "liger_kernel": False,
        "description": "+ FlashAttention-2 + Packing"
    },
    "grad_ckpt": {
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": True,
        "packing": True,
        "liger_kernel": False,
        "description": "+ FlashAttention-2 + Packing + Gradient Checkpoint"
    },
    "liger": {
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": True,
        "packing": True,
        "liger_kernel": True,
        "description": "+ FlashAttention-2 + Packing + Gradient Checkpoint + Liger"
    },
    "full": {
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": True,
        "packing": True,
        "liger_kernel": True,
        "paged_adamw_8bit": True,
        "description": "全部优化（完整配置）"
    },
}


def run_config(config_name, num_steps=100, seed=42):
    """运行指定配置 num_steps 步，记录 loss 序列和显存。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    cfg = CONFIGS[config_name]
    log_dir = f"logs/invariance/{config_name}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"=== Running config: {config_name} ({cfg['description']}) ===")

    # Liger kernel
    if cfg.get("liger_kernel"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama()
        except ImportError:
            print("  [WARN] Liger kernel not available")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:100]")
    def fmt(ex):
        return {"text": ex.get("conversations", [{}])[0].get("value", ex.get("text", ""))}
    dataset = dataset.map(fmt, remove_columns=dataset.column_names)

    training_args = SFTConfig(
        output_dir=log_dir,
        max_steps=num_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing") else None,
        optim="paged_adamw_8bit" if cfg.get("paged_adamw_8bit") else "adamw_torch",
        max_seq_length=2048,
        packing=cfg.get("packing", False),
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset, processing_class=tokenizer,
    )

    losses = []
    for step in range(num_steps):
        out = trainer.train_step() if hasattr(trainer, "train_step") else None
        # Fallback: use callback
    # Full train (collect from log)
    result = trainer.train()
    # Extract losses from log history
    for entry in trainer.state.log_history:
        if "loss" in entry:
            losses.append(entry["loss"])

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    record = {
        "config": config_name,
        "description": cfg["description"],
        "losses": losses[:num_steps],
        "peak_vram_gb": peak_vram,
        "num_steps": len(losses),
    }
    with open(f"{log_dir}/record.json", "w") as f:
        json.dump(record, f, indent=2)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Losses: {len(losses)} steps, final={losses[-1]:.4f}" if losses else "  No losses recorded")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")
    return record


def compare_configs(config_a, config_b):
    """对比两个配置的 loss 序列。"""
    rec_a = json.load(open(f"logs/invariance/{config_a}/record.json"))
    rec_b = json.load(open(f"logs/invariance/{config_b}/record.json"))

    losses_a = rec_a["losses"]
    losses_b = rec_b["losses"]
    n = min(len(losses_a), len(losses_b))

    if n == 0:
        print("ERROR: No loss data to compare")
        return

    # 相对偏差
    rel_diffs = []
    for i in range(n):
        if losses_a[i] != 0:
            rel = abs(losses_b[i] - losses_a[i]) / losses_a[i]
            rel_diffs.append(rel)

    max_rel = max(rel_diffs) if rel_diffs else 0
    mean_rel = np.mean(rel_diffs) if rel_diffs else 0
    threshold = 0.01  # 1%

    print(f"\n=== 不变性检查: {config_a} vs {config_b} ===")
    print(f"对比步数: {n}")
    print(f"最大相对偏差: {max_rel:.6f} ({'✅' if max_rel < threshold else '❌'} 阈值 {threshold})")
    print(f"平均相对偏差: {mean_rel:.6f}")
    print(f"显存: {rec_a['peak_vram_gb']:.2f} → {rec_b['peak_vram_gb']:.2f} GB "
          f"({rec_b['peak_vram_gb'] - rec_a['peak_vram_gb']:+.2f})")

    verdict = "PASS" if max_rel < threshold else "FAIL"
    print(f"\n判定: {verdict}")
    if verdict == "FAIL":
        print(f"⚠️ 相对偏差 > 1%，建议回退 {config_b} 的优化或调查原因")

    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS.keys()))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--compare", nargs=2, metavar=("CONFIG_A", "CONFIG_B"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.compare:
        compare_configs(args.compare[0], args.compare[1])
    elif args.config:
        run_config(args.config, args.steps, args.seed)
    else:
        print("Available configs:")
        for name, cfg in CONFIGS.items():
            print(f"  {name:<15s} {cfg['description']}")
        print("\nUsage:")
        print("  python numerical_invariance_check.py --config baseline --steps 100")
        print("  python numerical_invariance_check.py --compare baseline liger")


if __name__ == "__main__":
    main()
