#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu

cd /home/ubuntu/AdaQ-LoRA

LOG="logs/int8_seeds_p2.log"
echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) START P2 INT8 seed supplement (adaq/qlora x s7/s100)" | tee "$LOG"

# Ensure GPU1 is left alone for RTN
export CUDA_VISIBLE_DEVICES=0

MODEL="models/Meta-Llama-3-8B-int8rtn"
BATCH=1

for method in adaq qlora; do
  for seed in 7 100; do
    OUTDIR="outputs/phase3_int8/${method}_s${seed}"
    echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) TRAIN ${method}_s${seed} on GPU0" | tee -a "$LOG"
    python -u scripts/train.py \
      --model "$MODEL" \
      --method "$method" \
      --seed "$seed" \
      --output-dir "$OUTDIR" \
      --no-quant \
      --batch-size "$BATCH" \
      >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) FAIL ${method}_s${seed} rc=$rc" | tee -a "$LOG"
    else
      echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) OK ${method}_s${seed}" | tee -a "$LOG"
    fi
  done
done

echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) ALL DONE" | tee -a "$LOG"
