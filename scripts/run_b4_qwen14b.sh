#!/bin/bash
# P1-B4: Qwen2.5-14B 规模泛化验证（adaq vs qlora 两方法，同预算 512 rank）
set -euo pipefail
cd /home/ubuntu/AdaQ-LoRA
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=/home/ubuntu/AdaQ-LoRA/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
OUTBASE=outputs/adapters_qwen14b
RANKS=ranks_qwen14b.json

mkdir -p "$OUTBASE"

# Step 1: precompute ranks（用 Qwen 的 NF4 量化误差）
if [ ! -f "$RANKS" ]; then
  echo "=== [B4] precompute ranks for Qwen2.5-14B ==="
  $PY adaq-lora-research-pipeline/scripts/adaq_precompute.py \
    --model "$MODEL" \
    --budget 512 --r-min 4 --r-max 128 \
    --target-modules q_proj,v_proj \
    --out "$RANKS" 2>&1 | tail -20
else
  echo "=== [B4] ranks 已存在，跳过 precompute ==="
fi

train_one() {
  local method=$1 seed=$2
  local out="$OUTBASE/${method}_seed${seed}"
  if [ -f "$out/adapter_config.json" ]; then
    echo "=== [B4] $method seed$seed 已完成，SKIP ==="
    return 0
  fi
  echo "=== [B4] 训练 $method seed$seed (Qwen2.5-14B) ==="
  if [ "$method" = "adaq" ]; then
    $PY scripts/train.py --method adaq --model "$MODEL" --seed "$seed" \
      --budget 512 --base-rank 8 --ranks-file "$RANKS" --rank-scheme per_module_type \
      --target-modules q_proj,v_proj --max-length 512 --batch-size 2 --grad-accum 8 \
      --epochs 3 --lr 2e-4 --dataset-size 2000 \
      --output-dir "$out" 2>&1 | tail -15
  else
    $PY scripts/train.py --method qlora --model "$MODEL" --seed "$seed" \
      --base-rank 8 --target-modules q_proj,v_proj --max-length 512 --batch-size 2 --grad-accum 8 \
      --epochs 3 --lr 2e-4 --dataset-size 2000 \
      --output-dir "$out" 2>&1 | tail -15
  fi
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== [B4] $method seed$seed DONE ==="; else echo "=== [B4] $method seed$seed FAIL rc=$rc ==="; fi
  return $rc
}

# 顺序跑（14B 双卡，一次一个方法避免 OOM；两方法各 1 种子做规模泛化）
train_one adaq 42
train_one qlora 42

echo "=== [B4] 全部完成 ==="
