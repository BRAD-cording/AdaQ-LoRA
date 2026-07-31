#!/usr/bin/env bash
# AdaQ-LoRA 批量评测脚本（多方法 × 多种子 × 多任务）
# 依据：sota-comparison-protocol.md §4 评测协议
#
# 用法:
#   bash run_eval_batch.sh           # 全部方法 × 全部种子 × 全部任务
#   bash run_eval_batch.sh adaq      # 仅 AdaQ
#   bash run_eval_batch.sh adaq 42   # 仅 AdaQ seed=42
#
# 注意: 混合 shot 数需分批运行（MMLU=5-shot, GSM8K=4-shot, TruthfulQA/IFEval=0-shot）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/adaq-lora-research-pipeline/scripts/run_benchmarks.sh"

METHOD_FILTER="${1:-}"
SEED_FILTER="${2:-}"

METHODS=("qlora" "dora" "adalora" "adaq")
SEEDS=(42 3407 2025)

# 评测任务分批（因 shot 数不同）
# 批次1: MMLU 5-shot
# 批次2: GSM8K 4-shot
# 批次3: TruthfulQA + IFEval 0-shot
declare -A TASK_SHOTS=(
    ["mmlu"]="5"
    ["gsm8k"]="4"
    ["truthfulqa_mc1,truthfulqa_mc2,ifeval"]="0"
)

MERGED_DIR="${PROJECT_ROOT}/outputs/merged"
ADAPTER_DIR="${PROJECT_ROOT}/outputs/adapters"
BASE_MODEL="${ADAQ_MODEL_PATH:-/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B}"
RESULTS_DIR="${PROJECT_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

# Quick mode: limit samples per task for faster evaluation
QUICK_FLAG=""
if [[ "${QUICK:-0}" == "1" ]]; then
    QUICK_LIMIT="${QUICK_LIMIT:-100}"
    QUICK_FLAG="--limit ${QUICK_LIMIT}"
    echo "[QUICK MODE] Limiting to ${QUICK_LIMIT} samples per task"
fi

# Evaluation mode: 'merge' uses merged bf16 models, 'adapter' loads PEFT adapter directly
EVAL_MODE="${EVAL_MODE:-adapter}"

echo "============================================"
echo " AdaQ-LoRA 批量评测"
echo "============================================"

for method in "${METHODS[@]}"; do
    # 方法过滤
    if [[ -n "${METHOD_FILTER}" && "${method}" != "${METHOD_FILTER}" ]]; then
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        # 种子过滤
        if [[ -n "${SEED_FILTER}" && "${seed}" != "${SEED_FILTER}" ]]; then
            continue
        fi

        # Build model path based on eval mode
        if [[ "${EVAL_MODE}" == "merge" ]]; then
            model_path="${MERGED_DIR}/${method}_seed${seed}"
            model_args="pretrained=${model_path},dtype=bfloat16"
            if [[ ! -d "${model_path}" ]]; then
                echo "[SKIP] ${method} seed=${seed}: merged model not found at ${model_path}"
                echo "       Run merge first: python scripts/merge_adapter.py --method ${method} --seed ${seed}"
                continue
            fi
        else
            # Adapter mode: load base model (bf16) + PEFT adapter
            # Note: lm_eval doesn't support load_in_4bit; adapter compensates for quant errors
            adapter_path="${ADAPTER_DIR}/${method}_seed${seed}"
            model_args="pretrained=${BASE_MODEL},peft=${adapter_path},dtype=bfloat16"
            if [[ ! -d "${adapter_path}" ]]; then
                echo "[SKIP] ${method} seed=${seed}: adapter not found at ${adapter_path}"
                continue
            fi
        fi

        echo ""
        echo "--- ${method} (seed=${seed}) ---"

        # In quick mode, skip TruthfulQA/IFEval (only run MMLU+GSM8K main metrics)
        if [[ "${QUICK:-0}" == "1" ]]; then
            TASK_LIST=("mmlu" "gsm8k")
        else
            TASK_LIST=("mmlu" "gsm8k" "truthfulqa_mc1,truthfulqa_mc2,ifeval")
        fi

        for tasks in "${TASK_LIST[@]}"; do
            shots="${TASK_SHOTS[$tasks]}"
            out_dir="${RESULTS_DIR}/${method}_seed${seed}"
            mkdir -p "${out_dir}"

            echo "  Tasks: ${tasks} (${shots}-shot)"

            python -m lm_eval \
                --model hf \
                --model_args "${model_args}" \
                --tasks "${tasks}" \
                --num_fewshot "${shots}" \
                --batch_size auto \
                --device cuda:0 \
                --seed "${seed}" \
                ${QUICK_FLAG} \
                --output_path "${out_dir}" \
                2>&1 | tee -a "${out_dir}/eval.log" || \
                echo "  [WARN] Task ${tasks} failed, continuing..."
        done
    done
done

echo ""
echo "============================================"
echo " 评测完成。结果在 ${RESULTS_DIR}/<method>_seed<seed>/"
echo " 下一步: 用 scripts/aggregate_results.py 汇总均值±std"
echo "============================================"
