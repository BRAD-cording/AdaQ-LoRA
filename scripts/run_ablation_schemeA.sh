#!/usr/bin/env bash
# AdaQ-LoRA 方案A同配置消融训练：6 策略 × seed42 × 3 epoch × 2000 样本
# 与主实验 adaq_seed42 完全同配置：rank_pattern 模块类型(方案A) + alpha_pattern 空 {}
# 通过 --rank-scheme per_module_type 强制走方案A，忽略 per_layer_ranks。
# seq 1024 + batch 4 + grad_accum 4；断点续训 + 断电自动恢复（@reboot cron）。
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python}"
MODEL="${MODEL:-${PROJECT_ROOT}/models/Meta-Llama-3-8B}"
export HF_ENDPOINT=https://hf-mirror.com

SEED=42
EPOCHS=3
SIZE=2000
OUTBASE="${PROJECT_ROOT}/outputs/ablation_adapters_schemeA"
ROUNDS=(
  "error_only importance_only"
  "product uniform"
  "equal_weight random"
)

train_one() {
  local strat=$1 gpu=$2
  local out="${OUTBASE}/${strat}"
  local log="${PROJECT_ROOT}/logs/ablation_schemeA_${strat}.log"
  mkdir -p "${out}"
  if [[ -f "${out}/adapter_config.json" ]]; then
    echo "  [SKIP] ${strat} 已完成（adapter_config.json 存在）"
    return 0
  fi
  echo "  [TRAIN] ${strat} on gpu${gpu} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${PROJECT_ROOT}/scripts/train.py" \
    --method adaq --seed "${SEED}" --epochs "${EPOCHS}" --dataset-size "${SIZE}" \
    --max-length 1024 --batch-size 4 \
    --rank-scheme per_module_type \
    --ranks-file "${PROJECT_ROOT}/configs/ablation_ranks/ranks_${strat}.json" \
    --output-dir "${out}" --model "${MODEL}" >> "${log}" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]] && [[ -f "${out}/adapter_config.json" ]]; then
    echo "  [DONE] ${strat} rc=0 $(date '+%F %T')"
  else
    echo "  [FAIL] ${strat} rc=${rc} $(date '+%F %T')"
  fi
  return $rc
}

echo "=== AdaQ-LoRA 方案A同配置消融训练（rank_pattern 模块类型 + alpha 空）==="
round_idx=0
for pair in "${ROUNDS[@]}"; do
  round_idx=$((round_idx + 1))
  read -r s0 s1 <<< "${pair}"
  echo ""
  echo "===== 第 ${round_idx} 轮：GPU0=${s0}, GPU1=${s1} $(date '+%F %T') ====="
  train_one "${s0}" 0 &
  pid0=$!
  train_one "${s1}" 1 &
  pid1=$!
  wait ${pid0} || echo "  [FAIL] ${s0}"
  wait ${pid1} || echo "  [FAIL] ${s1}"
done

echo ""
echo "=== 方案A消融训练全部完成 $(date '+%F %T') ==="
for s in error_only importance_only product uniform equal_weight random; do
  f="${OUTBASE}/${s}/adapter_config.json"
  [[ -f "$f" ]] && echo "  ✅ ${s}" || echo "  ❌ ${s} (missing)"
done
