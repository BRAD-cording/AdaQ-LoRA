#!/usr/bin/env bash
# AdaQ-LoRA 统一口径评测（v2）
#
# 解决的问题：历史结果分布在 /mnt 与 /home 两套环境，且 MMLU 样本数不一致
#   （1425 = 每科 25 题 vs 5700 = 每科 100 题），跨方法对比无效。
# 本脚本保证：同一 OUT_DIR 下所有 run 用同一个 LIMIT、同一批 shot 数、同一个 base model。
#
# 用法:
#   OUT_DIR=results_v2 LIMIT= bash run_eval_unified.sh            # 全量（推荐论文主表）
#   OUT_DIR=results_q25 LIMIT=25 bash run_eval_unified.sh          # 冒烟/联调
#   GPU=1 METHODS="adaq" SEEDS="2025 3407" bash run_eval_unified.sh
#
# 环境变量:
#   OUT_DIR     结果目录（默认 results_v2）
#   LIMIT       每任务样本上限；留空=全量
#   METHODS     空格分隔，默认 "qlora dora adalora adaq"
#   SEEDS       空格分隔，默认 "42 3407 2025"
#   TASKSETS    main | safety | all（默认 main；safety=TruthfulQA+IFEval 0-shot）
#   GPU         CUDA_VISIBLE_DEVICES（默认 0）
#   FORCE       1 = 忽略已完成标记重跑
#   BASE_MODEL  基座模型路径
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Meta-Llama-3-8B}"
OUT_DIR="${OUT_DIR:-results_v2}"
LIMIT="${LIMIT:-}"
METHODS="${METHODS:-qlora dora adalora adaq}"
SEEDS="${SEEDS:-42 3407 2025}"
TASKSETS="${TASKSETS:-main}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"

ABS_OUT="${OUT_DIR}"
[[ "${ABS_OUT}" != /* ]] && ABS_OUT="${PROJECT_ROOT}/${OUT_DIR}"
mkdir -p "${ABS_OUT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export ADAQ_MODEL_PATH="${BASE_MODEL}"

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "[FATAL] base model not found: ${BASE_MODEL}" >&2
  exit 1
fi

LIMIT_FLAG=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_FLAG="--limit ${LIMIT}"
  echo "[INFO] LIMIT=${LIMIT} (每任务样本上限；仅用于冒烟，论文主表须留空)"
else
  echo "[INFO] LIMIT=full (全量)"
fi

# 任务集定义：任务名|num_fewshot|marker
case "${TASKSETS}" in
  main)   SETS=("mmlu|5|main" "gsm8k|4|main") ;;
  safety) SETS=("truthfulqa_mc1,truthfulqa_mc2,ifeval|0|safety") ;;
  all)    SETS=("mmlu|5|main" "gsm8k|4|main" "truthfulqa_mc1,truthfulqa_mc2,ifeval|0|safety") ;;
  *) echo "[FATAL] unknown TASKSETS=${TASKSETS}" >&2; exit 1 ;;
esac

echo "============================================"
echo " AdaQ-LoRA 统一评测 v2"
echo " GPU=${GPU}  OUT=${ABS_OUT}  METHODS=[${METHODS}]  SEEDS=[${SEEDS}]  SETS=${TASKSETS}"
echo " $(date '+%F %T')"
echo "============================================"

FAILED=()
for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    run="${method}_seed${seed}"
    adapter="${PROJECT_ROOT}/outputs/adapters/${run}"
    run_dir="${ABS_OUT}/${run}"
    mkdir -p "${run_dir}"

    if [[ ! -f "${adapter}/adapter_config.json" ]]; then
      echo "[SKIP] ${run}: adapter missing"
      FAILED+=("${run}:adapter-missing")
      continue
    fi

    model_args="pretrained=${BASE_MODEL},peft=${adapter},dtype=bfloat16"

    for spec in "${SETS[@]}"; do
      IFS='|' read -r tasks shots marker <<< "${spec}"
      marker_file="${run_dir}/.done_${marker}"
      if [[ -f "${marker_file}" && "${FORCE}" != "1" ]]; then
        echo "[SKIP] ${run} ${marker}: already done"
        continue
      fi

      echo ""
      echo "--- ${run} | ${tasks} (${shots}-shot) | $(date '+%T') ---"
      start_ts=$(date +%s)
      if "${PY}" -m lm_eval \
            --model hf \
            --model_args "${model_args}" \
            --tasks "${tasks}" \
            --num_fewshot "${shots}" \
            --batch_size auto \
            --device cuda:0 \
            --seed "${seed}" \
            ${LIMIT_FLAG} \
            --output_path "${run_dir}" \
            >> "${run_dir}/eval_${marker}.log" 2>&1; then
        touch "${marker_file}"
        echo "[ OK ] ${run} ${marker} in $(( $(date +%s) - start_ts ))s"
      else
        echo "[FAIL] ${run} ${marker} — see ${run_dir}/eval_${marker}.log"
        FAILED+=("${run}:${marker}")
      fi
    done
  done
done

echo ""
echo "============================================"
if (( ${#FAILED[@]} > 0 )); then
  echo " 有失败项: ${FAILED[*]}"
  exit 2
fi
echo " 全部完成 $(date '+%F %T')"
echo " 下一步: ${PY} scripts/aggregate_results.py --results-dir ${OUT_DIR} --out ${OUT_DIR}/summary.json"
echo "============================================"
