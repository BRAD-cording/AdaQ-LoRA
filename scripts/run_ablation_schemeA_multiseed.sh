#!/usr/bin/env bash
# P1-D4: 方案A同配置消融多种子训练（SEED env 化，复用 run_ablation_schemeA.sh 结构）
# SEED=3407 或 2025；输出到 outputs/ablation_adapters_schemeA_s<seed>/
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python}"
MODEL="${MODEL:-${PROJECT_ROOT}/models/Meta-Llama-3-8B}"
export HF_ENDPOINT=https://hf-mirror.com

SEED="${SEED:?SEED env required (e.g. 3407)}"
EPOCHS=3
SIZE=2000
OUTBASE="${PROJECT_ROOT}/outputs/ablation_adapters_schemeA_s${SEED}"
ROUNDS=(
  "error_only importance_only"
  "product uniform"
  "equal_weight random"
)

train_one() {
  local strat=$1 gpu=$2
  local out="${OUTBASE}/${strat}"
  local log="${PROJECT_ROOT}/logs/ablation_schemeA_s${SEED}_${strat}.log"
  mkdir -p "${out}"
  if [[ -f "${out}/adapter_config.json" ]]; then
    echo "  [SKIP] ${strat} 已完成"
    return 0
  fi
  echo "  [TRAIN] seed${SEED} ${strat} on gpu${gpu} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${PROJECT_ROOT}/scripts/train.py" \
    --method adaq --seed "${SEED}" --epochs "${EPOCHS}" --dataset-size "${SIZE}" \
    --max-length 1024 --batch-size 4 \
    --rank-scheme per_module_type \
    --ranks-file "${PROJECT_ROOT}/configs/ablation_ranks/ranks_${strat}.json" \
    --output-dir "${out}" --model "${MODEL}" >> "${log}" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]] && [[ -f "${out}/adapter_config.json" ]]; then
    echo "  [DONE] seed${SEED} ${strat} rc=0 $(date '+%F %T')"
  else
    echo "  [FAIL] seed${SEED} ${strat} rc=${rc} $(date '+%F %T')"
  fi
  return $rc
}

echo "=== 方案A消融多种子训练 seed=${SEED} ==="
for pair in "${ROUNDS[@]}"; do
  read -r s0 s1 <<< "${pair}"
  echo ""
  echo "===== GPU0=${s0}, GPU1=${s1} $(date '+%F %T') ====="
  train_one "${s0}" 0 &
  pid0=$!
  train_one "${s1}" 1 &
  pid1=$!
  wait ${pid0} || echo "  [FAIL] ${s0}"
  wait ${pid1} || echo "  [FAIL] ${s1}"
done
echo "=== seed=${SEED} 全部完成 $(date '+%F %T') ==="
