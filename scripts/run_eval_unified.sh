#!/usr/bin/env bash
# AdaQ-LoRA 统一口径评测（v2.1）
#
# 解决的问题：历史结果分布在 /mnt 与 /home 两套环境，且 MMLU 样本数不一致
#   （1425 = 每科 25 题 vs 5700 = 每科 100 题），跨方法对比无效。
# 本脚本保证：同一 OUT_DIR 下所有 run 用同一个 LIMIT、同一批 shot 数、同一个 base model。
#
# v2.1 修复（2026-11-11，根因见 docs/eval-protocol-v2.md）:
#   1) v2.0 把 main 的 mmlu 与 gsm8k 写成同一个 marker=main，第一个任务
#      touch .done_main 后第二个任务被 SKIP -> 12 个 run 无一同时拥有两项。
#      改为每任务独立 marker：.done_mmlu / .done_gsm8k / .done_safety。
#   2) v2.0 中 mmlu 遇 hf-mirror 429 失败后循环继续，gsm8k 成功反过来
#      touch 同一个 .done_main，把 mmlu 的失败彻底掩盖。现在每任务独立
#      marker + touch 前用 verify_eval_artifact.py 机读校验产物确实含该任务指标。
#   3) 默认 HF_HUB_OFFLINE=1（mmlu/gsm8k/truthfulqa/ifeval 数据集已缓存），
#      绕开 hf-mirror 的 429 Too Many Requests。
#   4) 任一任务失败即记入 FAILED 并 exit 2，不静默继续。
#
# 用法:
#   OUT_DIR=results_v2 LIMIT= bash run_eval_unified.sh            # 全量（论文主表）
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
#   OFFLINE     0 = 允许联网（数据集未缓存时需要）；默认 1
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
OFFLINE="${OFFLINE:-1}"

ABS_OUT="${OUT_DIR}"
[[ "${ABS_OUT}" != /* ]] && ABS_OUT="${PROJECT_ROOT}/${OUT_DIR}"
mkdir -p "${ABS_OUT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export ADAQ_MODEL_PATH="${BASE_MODEL}"
# 本机直连 huggingface.co 不通（Network is unreachable），hf-mirror.com 可达但会 429。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "${OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

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

# 任务集定义：任务名|num_fewshot|marker —— marker 必须每任务唯一（v2.1 铁律）
case "${TASKSETS}" in
  main)   SETS=("mmlu|5|mmlu" "gsm8k|4|gsm8k") ;;
  safety) SETS=("truthfulqa_mc1,truthfulqa_mc2,ifeval|0|safety") ;;
  all)    SETS=("mmlu|5|mmlu" "gsm8k|4|gsm8k" "truthfulqa_mc1,truthfulqa_mc2,ifeval|0|safety") ;;
  *) echo "[FATAL] unknown TASKSETS=${TASKSETS}" >&2; exit 1 ;;
esac

echo "============================================"
echo " AdaQ-LoRA 统一评测 v2.1"
echo " GPU=${GPU}  OUT=${ABS_OUT}  METHODS=[${METHODS}]  SEEDS=[${SEEDS}]  SETS=${TASKSETS}"
echo " OFFLINE=${OFFLINE}  $(date '+%F %T')"
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
        # 命令返回 0 不代表产物到了：429/配置错误可能让结果 JSON 只含别的任务。
        if "${PY}" "${PROJECT_ROOT}/scripts/verify_eval_artifact.py" "${run_dir}" "${marker}"; then
          touch "${marker_file}"
          echo "[ OK ] ${run} ${marker} in $(( $(date +%s) - start_ts ))s"
        else
          echo "[FAIL] ${run} ${marker}: artifact verification failed — see ${run_dir}/eval_${marker}.log"
          FAILED+=("${run}:${marker}:artifact-missing")
        fi
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
