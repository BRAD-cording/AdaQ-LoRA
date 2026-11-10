#!/usr/bin/env python3
"""训练 profiling：50 步快速 profile，记录 tokens/s、显存峰值、数据加载占比。

用法: python profile_training.py --method adaq --steps 50
"""
import argparse
import json
import os
import time
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adaq")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3-8B")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # Liger
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama()
        print("Liger kernel ✅")
    except ImportError:
        print("Liger kernel not available")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Load ranks for AdaQ
    peft_kwargs = dict(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
        task_type="CAUSAL_LM",
    )
    if args.method == "adaq" and os.path.exists("ranks.json"):
        with open("ranks.json") as f:
            ranks = json.load(f)
        type_ranks = ranks.get("per_module_type_ranks (方案A, PEFT 直接使用)", {})
        peft_kwargs["rank_pattern"] = type_ranks
        print(f"Loaded AdaQ rank pattern: {type_ranks}")

    peft_config = LoraConfig(**peft_config_kwargs if (peft_config_kwargs := {}) else peft_kwargs)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:200]")
    def fmt(ex):
        convs = ex.get("conversations", [{}])
        return {"text": convs[0].get("value", "") if convs else ex.get("text", "")}
    dataset = dataset.map(fmt, remove_columns=dataset.column_names)

    log_dir = f"logs/profile_{args.method}_seed{args.seed}"
    os.makedirs(log_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=log_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_seq_length=args.seq_length,
        packing=True,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset, processing_class=tokenizer,
    )

    print(f"\n=== Profiling {args.method} ({args.steps} steps) ===")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Sequence length: {args.seq_length}")
    print()

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    total_tokens = args.steps * args.batch_size * args.seq_length
    tokens_per_sec = total_tokens / total_time
    steps_per_sec = args.steps / total_time

    # Data loading overhead
    dataloader_stats = trainer.get_train_dataloader().num_workers if hasattr(trainer, 'get_train_dataloader') else 4

    profile = {
        "method": args.method,
        "model": args.model,
        "config": {
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seq_length": args.seq_length,
            "effective_batch": args.batch_size * args.grad_accum,
            "steps": args.steps,
        },
        "results": {
            "total_time_s": round(total_time, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "steps_per_sec": round(steps_per_sec, 3),
            "peak_vram_gb": round(peak_vram, 2),
            "dataloader_workers": dataloader_stats,
        },
        "loss_history": [e["loss"] for e in trainer.state.log_history if "loss" in e],
    }

    print(f"  Total time: {total_time:.1f}s")
    print(f"  Throughput: {tokens_per_sec:.1f} tok/s ({steps_per_sec:.3f} it/s)")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")
    print(f"  Loss range: {min(profile['loss_history']):.4f} - {max(profile['loss_history']):.4f}")

    out_file = os.path.join(log_dir, "profile.json")
    with open(out_file, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved to {out_file}")


if __name__ == "__main__":
    main()
