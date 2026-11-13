#!/usr/bin/env python3
"""AdaQ-LoRA 统一训练脚本（支持 qlora / dora / adalora / adaq 四种方法）。

铁律2（公平先于领先）：所有方法共享同一数据、epoch、lr 范围、量化配置。
唯一差异：秩分配方式。

用法：
  python train.py --method adaq --seed 42
  python train.py --method qlora --seed 42
  python train.py --method dora  --seed 42
  python train.py --method adalora --budget 512 --seed 42
"""
import argparse
import json
import os
import sys

MODEL_PATH = os.environ.get(
    "ADAQ_MODEL_PATH",
    "/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B"
)


def build_lora_config(method, args, ranks_file=None, rank_scheme="auto"):
    """根据方法构建 PEFT 配置。"""
    from peft import LoraConfig

    target_modules = args.target_modules.split(",")

    if method == "qlora":
        return LoraConfig(
            r=args.base_rank, lora_alpha=args.base_rank * 2, lora_dropout=0.05,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
    elif method == "dora":
        return LoraConfig(
            r=args.base_rank, lora_alpha=args.base_rank * 2, lora_dropout=0.05,
            target_modules=target_modules, bias="none", use_dora=True,
            task_type="CAUSAL_LM",
        )
    elif method == "adalora":
        try:
            from peft import AdaLoraConfig
        except ImportError:
            print("ERROR: AdaLoraConfig not available in this PEFT version", file=sys.stderr)
            sys.exit(1)
        return AdaLoraConfig(
            init_r=args.base_rank * 2, target_r=args.base_rank,
            tinit=20, tfinal=60, deltaT=5, beta1=0.3, beta2=0.3,
            orth_reg_weight=0.1, lora_alpha=args.base_rank * 2, lora_dropout=0.05,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
            total_step=200,
        )
    elif method == "adaq":
        if not ranks_file or not os.path.exists(ranks_file):
            print(f"ERROR: ranks.json not found at {ranks_file}. Run adaq_precompute.py first.", file=sys.stderr)
            sys.exit(1)
        with open(ranks_file) as f:
            ranks_data = json.load(f)

        def _to_pattern_key(k):
            # ranks.json 的 per_layer_ranks key 形如 model.layers.N.self_attn.q_proj.weight
            # PEFT rank_pattern 用后缀正则 (.*\.)?({key})$ 匹配 current_key
            # （current_key = named_modules 的模块名，形如 base_model.model.model.layers.N.self_attn.q_proj，不含 .weight）
            # 故需去掉 .weight 后缀 + 去掉前导 "model."，才能 64/64 精确匹配逐层。
            k2 = k.replace(".weight", "")
            if k2.startswith("model."):
                k2 = k2[len("model."):]
            return k2

        # rank_scheme 控制秩分配粒度：
        #   auto             —— 优先逐层（方案B），无逐层才回退模块类型（方案A）
        #   per_layer        —— 强制逐层（方案B）
        #   per_module_type  —— 强制模块类型（方案A），复刻主实验早期行为
        per_layer = ranks_data.get("per_layer_ranks", {}) if rank_scheme != "per_module_type" else {}
        if per_layer:
            rank_pattern = {_to_pattern_key(k): int(v) for k, v in per_layer.items()}
            alpha_pattern = {k: 2 * v for k, v in rank_pattern.items()}
            scheme = "per_layer"
        else:
            # 回退：模块类型聚合（方案A），兼容旧 ranks.json
            type_ranks = ranks_data.get("per_module_type_ranks (方案A, PEFT 直接使用)", {})
            if not type_ranks:
                print("WARNING: No per_layer_ranks / type_ranks in ranks.json, falling back to uniform", file=sys.stderr)
                type_ranks = None
            rank_pattern = type_ranks
            if rank_scheme == "per_module_type":
                # 严格复刻主实验早期行为：只用 rank_pattern，不写 alpha_pattern
                alpha_pattern = None
            else:
                alpha_pattern = (
                    {mod: 2 * r for mod, r in type_ranks.items() if mod in target_modules}
                    if type_ranks else None
                )
            scheme = "per_module_type" if type_ranks else "uniform"

        config_kwargs = dict(
            r=args.base_rank, lora_alpha=args.base_rank * 2, lora_dropout=0.05,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        if rank_pattern:
            config_kwargs["rank_pattern"] = rank_pattern
            # 非均匀秩下 α 须随 r 缩放（α_l = 2·r_l），保证各层有效学习率统一，
            # 与 qlora/dora 的 α=2r（铁律2：公平先于领先）保持一致。
            if alpha_pattern:
                config_kwargs["alpha_pattern"] = alpha_pattern
        print(f"[AdaQ] rank scheme={scheme}, rank_pattern entries={len(rank_pattern) if rank_pattern else 0}", file=sys.stderr)
        return LoraConfig(**config_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser(description="AdaQ-LoRA unified training")
    ap.add_argument("--method", required=True, choices=["qlora", "dora", "adalora", "adaq"])
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--base-rank", type=int, default=8, help="Uniform rank for baselines")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--target-modules", default="q_proj,v_proj")
    ap.add_argument("--ranks-file", default="ranks.json")
    ap.add_argument("--rank-scheme", default="auto", choices=["auto", "per_layer", "per_module_type"],
                    help="秩分配粒度：auto=优先逐层回退模块类型；per_layer=强制逐层；per_module_type=强制模块类型(方案A)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    ap.add_argument("--dataset-size", type=int, default=2000)
    ap.add_argument("--no-packing", dest="packing", action="store_false", default=True)
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/adapters/{args.method}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # --- 量化配置（所有方法统一） ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"=== Loading model: {args.model} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- PEFT 配置 ---
    peft_config = build_lora_config(args.method, args, ranks_file=args.ranks_file, rank_scheme=args.rank_scheme)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 数据 ---
    print(f"=== Loading dataset: {args.dataset} (first {args.dataset_size}) ===")
    dataset = load_dataset(args.dataset, split=f"train[:{args.dataset_size}]")

    # Format alpaca: use 'text' field if available, otherwise format
    if "text" in dataset.column_names:
        dataset = dataset.select_columns(["text"])
    elif "instruction" in dataset.column_names:
        def format_alpaca(ex):
            text = ex.get("instruction", "")
            if ex.get("input"):
                text += "\n" + ex["input"]
            text += "\n" + ex.get("output", "")
            return {"text": text}
        dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    elif "conversations" in dataset.column_names:
        # OpenHermes-2.5 等 chat 格式：拼接多轮 human/gpt 为单个 text 序列
        def format_chat(ex):
            parts = []
            for turn in ex["conversations"]:
                role = turn.get("from", "")
                value = turn.get("value", "")
                if role == "human":
                    parts.append(f"### Human:\n{value}")
                elif role == "gpt":
                    parts.append(f"### Assistant:\n{value}")
                else:
                    parts.append(value)
            return {"text": "\n\n".join(parts)}
        dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    print(f"Dataset: {len(dataset)} samples")

    # --- 训练参数 ---
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        save_only_model=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        seed=args.seed,
        max_seq_length=args.max_length,
        packing=args.packing,
        dataset_text_field="text",
        use_liger_kernel=False,
        report_to="none",
    )

    # --- 训练 ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n=== Training {args.method} (seed={args.seed}, budget={args.budget}) ===")

    # --- 断点续训：检测 output_dir 下最近的 checkpoint-*，自动 resume ---
    resume_ckpt = None
    if os.path.isdir(args.output_dir):
        ckpts = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
            key=lambda d: int(d.split("-")[1]),
        )
        if ckpts:
            resume_ckpt = os.path.join(args.output_dir, ckpts[-1])
            print(f"[resume] 发现断点 {resume_ckpt}，从断点续训", file=sys.stderr)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- 保存 adapter ---
    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

    # --- 记录显存峰值 ---
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_vram:.2f} GB")
        with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
            json.dump({
                "method": args.method, "seed": args.seed, "budget": args.budget,
                "peak_vram_gb": round(peak_vram, 2),
            }, f, indent=2)

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
