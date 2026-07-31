#!/usr/bin/env python3
"""合并 4-bit 训练后的 adapter 到 bf16 基座模型。

训练完成后 4-bit 反量化 → merge adapter → 保存 bf16 完整模型用于评测。
merge 后约 15GB，A10 单卡可推理。

用法: python merge_adapter.py --method adaq --seed 42
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["qlora", "dora", "adalora", "adaq"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base-model", default=os.environ.get(
        "ADAQ_MODEL_PATH",
        "/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B"
    ))
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.adapter_dir is None:
        args.adapter_dir = f"outputs/adapters/{args.method}_seed{args.seed}"
    if args.output_dir is None:
        args.output_dir = f"outputs/merged/{args.method}_seed{args.seed}"

    if not os.path.exists(args.adapter_dir):
        print(f"ERROR: adapter dir not found: {args.adapter_dir}", file=sys.stderr)
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # 加载 4-bit 基座并反量化到 bf16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # 加载 adapter
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    print(f"Merging {args.method} adapter (seed={args.seed})...")

    # 合并
    model = model.merge_and_unload()

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {args.output_dir}")
    print(f"Ready for evaluation: bash scripts/run_eval_batch.sh {args.method} {args.seed}")


if __name__ == "__main__":
    main()
