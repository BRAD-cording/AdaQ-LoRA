#!/bin/bash
# P1-B4 补齐种子：Qwen2.5-14B 上 adaq vs qlora，seed 2025 / 3407（seed42 已有）
# 双卡并行：adaq->GPU0, qlora->GPU1；按种子顺序推进（每卡单进程，避免 14B OOM）
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=/home/ubuntu/AdaQ-LoRA/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
OUTBASE=outputs/adapters_qwen14b
RANKS=ranks_qwen14b.json
SEEDS="${SEEDS:-2025 3407}"
LOGDIR=logs
mkdir -p "$OUTBASE" "$LOGDIR"

echo "=== [B4-seeds] 开始 $(date -u) SEEDS=$SEEDS ==="

train_one() {
  local method=$1 seed=$2 gpu=$3
  local out="$OUTBASE/${method}_seed${seed}"
  local log="$LOGDIR/b4_${method}_s${seed}.log"
  if [ -f "$out/adapter_config.json" ] && [ -f "$out/adapter_model.safetensors" ]; then
    echo "[SKIP] $method seed$seed 已完成"
    return 0
  fi
  echo "=== [B4-seeds] train $method seed$seed on GPU$gpu $(date -u) ==="
  if [ "$method" = "adaq" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train.py --method adaq --model "$MODEL" --seed "$seed" \
      --budget 512 --base-rank 8 --ranks-file "$RANKS" --rank-scheme per_module_type \
      --target-modules q_proj,v_proj --max-length 512 --batch-size 2 --grad-accum 8 \
      --epochs 3 --lr 2e-4 --dataset-size 2000 \
      --output-dir "$out" > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train.py --method qlora --model "$MODEL" --seed "$seed" \
      --base-rank 8 --target-modules q_proj,v_proj --max-length 512 --batch-size 2 --grad-accum 8 \
      --epochs 3 --lr 2e-4 --dataset-size 2000 \
      --output-dir "$out" > "$log" 2>&1
  fi
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [B4-seeds] DONE $method seed$seed $(date -u) ==="
  else
    echo "=== [B4-seeds] FAIL $method seed$seed rc=$rc (log $log) ==="
  fi
  return $rc
}

for s in $SEEDS; do
  train_one adaq  "$s" 0 &
  pid_a=$!
  train_one qlora "$s" 1 &
  pid_q=$!
  wait $pid_a || true
  wait $pid_q || true
  echo "=== [B4-seeds] seed $s 这一轮结束 $(date -u) ==="
done

echo "=== [B4-seeds] 全部完成 $(date -u) ==="
ls -1 "$OUTBASE"/*/adapter_config.json 2>/dev/null
