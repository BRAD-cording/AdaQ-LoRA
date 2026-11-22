#!/bin/bash
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs outputs/phase3_int8
LOG=logs/int8_pipeline.log

# Wait for GPU1 to free
echo "=== [$(date -u +%FT%T)] Waiting for GPU1..." | tee "$LOG"
while true; do
  mem=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader | grep -oP '\d+')
  if [ "$mem" -lt 2000 ]; then echo "GPU1 free (${mem} MiB)"; break; fi
  sleep 10
done

# Step 1: epsilon INT8
echo "=== [$(date -u +%FT%T)] STEP 1: measure epsilon INT8 ===" | tee -a "$LOG"
env CUDA_VISIBLE_DEVICES=1 python scripts/measure_epsilon_int8.py 2>&1 | tee -a "$LOG"
CV=$(python3 -c "import json; d=json.load(open('results_v2/epsilon_report_int8.json')); print(round(d['cv_pct'],2))")
echo "INT8 epsilon CV = ${CV}%" | tee -a "$LOG"

# Step 2: quantize
echo "=== [$(date -u +%FT%T)] STEP 2: quantize INT8 model ===" | tee -a "$LOG"
env CUDA_VISIBLE_DEVICES=1 python scripts/quantize_int8.py 2>&1 | tee -a "$LOG"

# Step 3: train (adaq + qlora × 3 seeds)
export ADAQ_MODEL_PATH=models/Meta-Llama-3-8B-int8rtn
echo "=== [$(date -u +%FT%T)] STEP 3: INT8 training (6 jobs) ===" | tee -a "$LOG"

for method in adaq qlora; do
  for seed in 42 3407 2025; do
    OUT="outputs/phase3_int8/${method}_s${seed}"
    if [ -f "${OUT}/adapter_model.safetensors" ]; then
      echo "$(date -u +%FT%T) SKIP ${method}_s${seed}" | tee -a "$LOG"
      continue
    fi
    mkdir -p "$OUT"
    echo "$(date -u +%FT%T) TRAIN ${method}_s${seed}" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
      --method "$method" --seed "$seed" --epochs 3 --dataset-size 2000 \
      --output-dir "$OUT" --batch-size 1 --grad-accum 16 \
      --base-rank 8 --budget 512 --ranks-file ranks.json --no-quant \
      >> "${OUT}/train.log" 2>&1
    rc=$?; rflag="OK"; [ "$rc" != 0 ] && rflag="FAIL rc=$rc"
    [ -f "${OUT}/adapter_model.safetensors" ] || rflag="FAIL no checkpoint"
    echo "$(date -u +%FT%T) $rflag ${method}_s${seed}" | tee -a "$LOG"
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    sleep 3
  done
done
echo "=== [$(date -u +%FT%T)] INT8 pipeline DONE ===" | tee -a "$LOG"