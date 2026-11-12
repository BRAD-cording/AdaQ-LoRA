#!/usr/bin/env bash
# AdaQ-LoRA 消融训练：6 策略 × seed42 × 3 epoch × 2000 样本
# seq 1024 + batch 4 + grad_accum 4（有效 batch 16，显存安全）
# 每卡 1 个，2 卡并行，3 轮跑完
# 断点续训：train.py 已支持 resume_from_checkpoint，每 10 step 存 checkpoint-*
# 断电恢复：每次重启后由 @reboot cron 调 resume_ablation.sh 自动续跑
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python}"
MODEL="${MODEL:-${PROJECT_ROOT}/models/Meta-Llama-3-8B}"
export HF_ENDPOINT=https://hf-mirror.com

SEED=42
EPOCHS=3
SIZE=2000
ROUNDS=(
  "error_only importance_only"
  "product uniform"
  "equal_weight random"
)

# 单个策略训练函数：已完成（有 adapter_config.json）则跳过
train_one() {
  local strat=$1 gpu=$2
  local out="${PROJECT_ROOT}/outputs/ablation_adapters/${strat}"
  local log="${PROJECT_ROOT}/logs/ablation_${strat}.log"
  mkdir -p "${out}"
  if [[ -f "${out}/adapter_config.json" ]]; then
    echo "  [SKIP] ${strat} 已完成（adapter_config.json 存在）"
    return 0
  fi
  echo "  [TRAIN] ${strat} on gpu${gpu} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${PROJECT_ROOT}/scripts/train.py" \
    --method adaq --seed "${SEED}" --epochs "${EPOCHS}" --dataset-size "${SIZE}" \
    --max-length 1024 --batch-size 4 \
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

echo "=== AdaQ-LoRA 消融训练：6 策略 × 2 卡 × 3 轮（seq 1024 batch 4，断点续训）==="
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
echo "=== 消融训练全部完成 $(date '+%F %T') ==="
for s in error_only importance_only product uniform equal_weight random; do
  f="outputs/ablation_adapters/${s}/adapter_config.json"
  [[ -f "$f" ]] && echo "  ✅ ${s}" || echo "  ❌ ${s} (missing)"
done
