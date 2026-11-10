#!/usr/bin/env bash
# AdaQ-LoRA 标准评测命令生成（lm-evaluation-harness）
# 用法: bash run_benchmarks.sh <model_or_merged_path> <method_tag> <seed> [tasks]
# 示例: bash run_benchmarks.sh ./merged/llama3-8b-adaq adaq 42 "mmlu,gsm8k"
set -euo pipefail

MODEL=${1:?需要模型路径}
TAG=${2:?需要方法标签（qlora/dora/adalora/adaq）}
SEED=${3:?需要种子}
TASKS=${4:-"mmlu,gsm8k,truthfulqa_mc1,truthfulqa_mc2,ifeval"}
OUT="results/${TAG}_seed${SEED}"
mkdir -p "${OUT}"

echo "== 评测 ${TAG} (seed=${SEED}) 任务: ${TASKS} =="
echo "== 输出目录: ${OUT}（原始 json 必须保留，论文数值只能由 json 汇总生成）=="

# A10 24GB：bf16 推理 batch 4-8；GSM8K 为生成式评测，耗时约 1-2h
python -m lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks "${TASKS}" \
    --num_fewshot 5 \
    --batch_size 4 \
    --device cuda:0 \
    --seed "${SEED}" \
    --output_path "${OUT}"

# 若只给 adapter（未 merge），改用：
# python -m lm_eval --model hf \
#     --model_args "pretrained=<base_model>,peft=<adapter_path>,dtype=bfloat16" ...

# 注意: --num_fewshot 对 mmlu=5, gsm8k=4；混合任务时请分两次运行以使用各自 shot 数:
#   mmlu -> --num_fewshot 5 ; gsm8k -> --num_fewshot 4 ; truthfulqa/ifeval -> 0-shot
echo "== 完成。下一步: 用 json 汇总均值±std，禁止手抄数值 =="
